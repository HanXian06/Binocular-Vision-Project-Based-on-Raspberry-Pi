# 双目测距版本与复现入口

更新者：陈珂乐，2026-11-23。原识别模块及协作者贡献保留。

- [v0.1](v0.1/README.md)：稠密SGBM基线，包含历史反查和NCC对照。
- [v0.2](v0.2/README.md)：主动选点＋NCC，保留为实验路线。
- [v0.3](v0.3/README.md)：最新离线YOLO自动选区＋原SGBM闭环。
- [固定ROI逻辑](roi_protocol/README.md)：无YOLO静态对照，不能替代自动检测验收。
- [协作接口与接手路线](../../Shared/StereoRanging/README.md)。

从仓库根目录先运行 `python "Algorithm/双目测距/prepare_paths.py"`。v0.3再进入对应目录运行 `python run_offline.py --out results/my_run`。默认运行43对；目标未检出不补人工距离。

最新模型未重新训练，书本漏检和部分前景椅子漏检是当前主要检测缺口；密集上下搜索尚未加入。没有本次上板测速，不把本机延迟当树莓派帧率。相机参数副本为background1105，不是把Calibration1111候选默认启用。
