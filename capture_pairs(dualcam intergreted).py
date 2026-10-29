"""双目预览采集：S 拍摄定格；定格后 S 保存、W 放弃；Q 退出。需在树莓派运行。"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


def capture_one(camera, warmup):
    camera.start()
    try:
        time.sleep(warmup)
        request = camera.capture_request()
        try:
            image = request.make_array("main").copy()
            metadata = request.get_metadata()
            return image, metadata
        finally:
            request.release()
    finally:
        camera.stop()


def save_pair(output, pair_id, left, right, record):
    paths = [output / "left" / f"{pair_id}_L.png",
             output / "right" / f"{pair_id}_R.png",
             output / "metadata" / f"{pair_id}.json"]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"拒绝覆盖已有配对：{pair_id}")
    try:
        for path, image in zip(paths[:2], (left, right)):
            if not cv2.imwrite(str(path), image):
                raise OSError(f"图片保存失败：{path}")
        with paths[2].open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, default=str)
    except Exception:
        # 不留下只有一侧图片、却看起来像成功配对的文件。
        for path in paths:
            path.unlink(missing_ok=True)
        raise


def show_pair(window, left, right, frozen):
    # 仅在拼接副本上显示操作提示，保存的原图不添加文字。
    canvas = cv2.hconcat([left, right])
    message = ("FROZEN | S: save + live | W: discard + live | Q: quit"
               if frozen else "LIVE | S: capture + freeze | Q: quit")
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(canvas, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255) if frozen else (0, 255, 0), 2)
    cv2.imshow(window, canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="新建的输出目录；不能已存在")
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="正式拍照时每路启动后等待秒数，默认 2 秒；不代表同步")
    parser.add_argument("--preview-warmup", type=float, default=0.01,
                        help="实时预览时每路启动后等待秒数，默认 0.01 秒")
    args = parser.parse_args()
    if args.left_camera == args.right_camera:
        parser.error("左右相机编号必须不同")
    if not np.isfinite(args.warmup) or args.warmup < 0:
        parser.error("warmup 必须是非负有限数")
    if not np.isfinite(args.preview_warmup) or args.preview_warmup < 0:
        parser.error("preview-warmup 必须是非负有限数")

    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("left", "right", "metadata"):
        (output / folder).mkdir()
    setup = {
        "left_camera": args.left_camera,
        "right_camera": args.right_camera,
        "size_per_camera": [640, 480],
        "pixel_format": "RGB888",
        "capture_mode": "sequential_static_only",
        "warmup_seconds_per_camera": args.warmup,
        "preview_warmup_seconds_per_camera": args.preview_warmup,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "capture_setup.json").write_text(
        json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "measurements.csv").write_text(
        "pair_id,target,actual_depth_m,reference_surface,pose_or_position,notes\n",
        encoding="utf-8")

    cameras = []
    count = 0
    window = "Stereo capture"
    pending = None
    try:
        for number in (args.left_camera, args.right_camera):
            camera = Picamera2(camera_num=number)
            cameras.append(camera)
            camera.configure(camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}))

        canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
        cv2.putText(canvas, "Starting cameras...", (40, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, canvas)
        print(f"保存位置：{output}")
        print("点击图像窗口：实时画面按 S 拍摄并定格（不需要按回车）。")
        print("定格后：S 保存并返回实时画面，W 放弃并返回实时画面。")
        print("Q 退出；尚未确认的定格照片不会保存。")
        print("左右依次采集；拍照约需 4 秒或更久，请保持相机和目标不动。")
        while True:
            if pending is None:
                left, _ = capture_one(cameras[0], args.preview_warmup)
                right, _ = capture_one(cameras[1], args.preview_warmup)
                show_pair(window, left, right, frozen=False)

            key = cv2.waitKey(50 if pending is not None else 1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

            if pending is not None:
                if key in (ord("w"), ord("W")):
                    pending = None
                    print("已放弃本次照片，返回实时预览。", flush=True)
                elif key in (ord("s"), ord("S")):
                    left, right, record = pending
                    pair_id = record["pair_id"]
                    save_pair(output, pair_id, left, right, record)
                    with (output / "measurements.csv").open("a", encoding="utf-8") as stream:
                        stream.write(f"{pair_id},,,,,\n")
                    count += 1
                    pending = None
                    print(f"已保存 {pair_id}：左、右各一张 640x480 PNG，返回实时预览。",
                          flush=True)
                continue

            if key not in (ord("s"), ord("S")):
                continue

            pair_id = f"pair_{count + 1:04d}"
            print(f"正在拍摄 {pair_id}，请保持不动……", flush=True)
            started = datetime.now(timezone.utc).isoformat()
            left, left_meta = capture_one(cameras[0], args.warmup)
            right, right_meta = capture_one(cameras[1], args.warmup)
            record = {
                "pair_id": pair_id,
                "capture_mode": "sequential_static_only",
                "host_started_utc": started,
                "host_finished_utc": datetime.now(timezone.utc).isoformat(),
                "left_camera": args.left_camera,
                "right_camera": args.right_camera,
                "left_metadata": left_meta,
                "right_metadata": right_meta,
            }
            pending = (left, right, record)
            show_pair(window, left, right, frozen=True)
            print("已定格，尚未保存：S 保存，W 放弃，均返回实时预览。", flush=True)
    finally:
        for camera in reversed(cameras):
            try:
                camera.close()
            except Exception as error:
                print(f"关闭相机时出现提示：{error}")
        cv2.destroyAllWindows()
        print(f"本次保存 {count} 对；目录：{output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("采集已中断。")
