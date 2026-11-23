"""Export a fresh 512x384 NCNN model from YOLOE-26s and a fixed class list.

Run on the export workstation using Ultralytics 8.4.215, PNNX 20260526 and
NCNN 1.0.20261020. Put the original checkpoint and text encoder in the
working directory. Supply a separate output directory for every export.
"""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLOE


def defer_sigmoid(path):
    lines = path.read_text(encoding="ascii").splitlines()
    sigmoid, concat = map(str.split, lines[-2:])
    if (sigmoid[:1] != ["Sigmoid"] or sigmoid[2:4] != ["1", "1"]
            or len(sigmoid) != 6 or concat[:1] != ["Concat"]
            or concat[2:4] != ["2", "1"] or concat[5] != sigmoid[5]
            or concat[7:] != ["0=0"]
            or sum(sigmoid[5] in row.split()[4:] for row in lines[2:]) != 2):
        raise ValueError("Unexpected classification head; cannot defer sigmoid")
    layers, blobs = map(int, lines[1].split())
    lines[1] = f"{layers - 1} {blobs - 1}"
    concat[5] = sigmoid[4]
    lines[-2:] = [" ".join(concat)]
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def add_visual_examples(source, checkpoint, labels, examples_file):
    """Optional JSON: [{label, weight, examples:[{image, bbox_xyxy}]}]."""
    text = source.get_text_pe(labels).clone()
    example_model = YOLOE(str(checkpoint))
    for group in json.loads(examples_file.read_text(encoding="utf-8")):
        index = labels.index(group["label"])
        weight = float(group["weight"])
        if not 0 <= weight <= 1 or not group["examples"]:
            raise ValueError("A visual prompt needs examples and a weight in [0, 1]")
        embeddings = []
        for item in group["examples"]:
            path = examples_file.parent / item["image"]
            frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Cannot decode visual example: {path}")
            example_model.predict(
                frame, refer_image=frame, imgsz=(384, 512), conf=.15, verbose=False,
                visual_prompts={"bboxes": np.array([item["bbox_xyxy"]], dtype=np.float32),
                                "cls": np.array([0], dtype=np.int64)},
            )
            embeddings.append(example_model.model.pe[:, 0].detach().to(text).clone())
        visual = F.normalize(torch.stack(embeddings).sum(dim=0), dim=-1)
        text[:, index] = F.normalize((1 - weight) * text[:, index] + weight * visual, dim=-1)
    source.set_classes(labels, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--visual-examples", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output directory")
    labels = args.labels.read_text(encoding="utf-8").splitlines()
    if not labels or len(labels) != len(set(labels)) or any(not label.strip() for label in labels):
        parser.error("each non-empty class name must appear exactly once")
    checkpoint = args.checkpoint.resolve(strict=True)
    source = YOLOE(str(checkpoint))
    if args.visual_examples:
        add_visual_examples(source, checkpoint, labels, args.visual_examples)
    else:
        source.set_classes(labels)
    with TemporaryDirectory() as tmp:
        profile = Path(tmp) / "prompts.npz"
        source.save_prompt_embeddings(profile)
        model = YOLOE("yoloe-26s.yaml").load(str(checkpoint))
        model.load_prompt_embeddings(profile)
        model.model.pt_path = str(Path(tmp) / checkpoint.name)
        exported = Path(model.export(format="ncnn", imgsz=(384, 512), quantize=16, device="cpu"))
        defer_sigmoid(exported / "model.ncnn.param")
        args.out.mkdir(parents=True)
        for name in ("model.ncnn.param", "model.ncnn.bin", "metadata.yaml"):
            shutil.copyfile(exported / name, args.out / name)
        (args.out / "input.txt").write_text("384x512 logits\n", encoding="ascii", newline="\n")
        (args.out / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
