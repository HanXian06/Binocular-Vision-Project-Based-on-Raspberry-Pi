"""发布入口：使用协作者YOLO目录独立子文件夹中的结构化接口。"""
from pathlib import Path
import importlib.util
repo=Path(__file__).resolve().parents[4]
path=repo/'Algorithm/Dual-Camera-YOLO/stereo_v0_3_interface/yolox_backend.py'
spec=importlib.util.spec_from_file_location('_stereo_yolox_backend',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Detector=module.Detector
decode=module.decode
preprocess=module.preprocess
CLASSES=module.CLASSES
