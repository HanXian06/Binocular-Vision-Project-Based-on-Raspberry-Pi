# v0.3结构化识别接口（独立另存）

更新者：陈珂乐，2026-11-23。原双摄识别由kabuda-pixel在PR #3提供，本目录只添加测距接口适配，不覆盖父目录detect.py、yolox_backend.py或模型。

infer返回类别、置信度、校正左图xyxy框；annotate保留显示功能。模型、416预处理、按类NMS算法未更换。Windows中文路径改为模型字节加载。

从仓库根目录先运行 `python "Algorithm/双目测距/prepare_paths.py"`，再运行：

```bash
python Algorithm/Dual-Camera-YOLO/stereo_v0_3_interface/run_stereo.py --pair PartC/pair_0007 --out stereo_demo
```

完整管线、配置和模型副本在[双目测距v0.3](../../双目测距/v0.3/README.md)，接口约定见[共享文件](../../../Shared/StereoRanging/README.md)。本入口是离线双图，未修改原相机循环，不宣称已上板实时运行。
