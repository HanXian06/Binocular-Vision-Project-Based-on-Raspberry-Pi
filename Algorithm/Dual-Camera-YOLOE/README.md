# 树莓派双摄 YOLOE 实时识别

本目录保存 261 类 YOLOE 识别、双摄补漏和流畅预览的部署版本。现有树莓派安装路径为 `/home/pi/pi-dual-yolo/`，VNC 桌面入口为 **Dual Camera YOLO**。新装或并行测试时使用独立目录 `~/pi-yoloe-preview/`。

## 当前进展（2026-11-23）

| 项目 | 已部署状态 | 验证与限制 |
| --- | --- | --- |
| 识别模型 | YOLOE-26s 检测版，固定 261 类，输入 512×384，NCNN FP16 离线推理 | 保留 80 个 COCO 类别，增加插线板、插座、智能手表等。原始提示已写入模型，改动类别需重新导出。 |
| 检测头 | 导出计算图输出 logits，类别筛选后再计算 Sigmoid | 避免对所有类别和候选框计算 Sigmoid。后续试验的内存池、卷积开关和输出拆分已回退。 |
| 双摄关联 | 从时间差不超过 45 毫秒的图对提取共同特征，配对后复用编号，必要时局部复检 | 18 对 PartC 离线图建立 48 次框关联，补回 1 个右路人物框。匹配证据不足时保留单路结果。 |
| 实时预览 | 相机取帧、界面刷新与推理解耦。识别只取最新图对，旧框在外观匹配时短暂跟随 | 树莓派 5 和两颗 OV5647 上，VNC 画面每路约 9 FPS，识别每路约 3.8 次/秒。95% 的画面刷新间隔约在 116 毫秒内。 |

离线比较使用 v0.3 相同的 43 张校正左图和参考框关联规则。两版沿用各自部署阈值。此处统计**参考目标检出**，单独保留仓库测距 v0.3 的原始记录。

| 参考目标 | YOLOX-Nano（测距 v0.3） | YOLOE（本分支） |
| --- | ---: | ---: |
| 书本 | 0/25 | 4/25 |
| 人物 | 9/9 | 6/9 |
| 椅子 | 5/9 | 9/9 |
| 合计 | 14/43 | 19/43 |

YOLOE 漏掉的 3 个人物样本均标成了夹克。书本命中的 4 张都在约半米处。实物插线板和手表还需要更多角度和遮挡条件下的验证。实时版本当前提供识别和双摄关联，测距 v0.3 单独运行。

## 安装与运行

依赖以实测的树莓派 5、64 位 Raspberry Pi OS（Debian 13）和 Python 3.13 为基准。在本目录运行：

```sh
sudo apt install python3-venv python3-tk python3-numpy python3-picamera2 util-linux
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps -r requirements.txt
sh start.sh
```

`--system-site-packages` 用于复用系统 Picamera2 和 NumPy。`--no-deps` 避免 NCNN 的 pip 依赖再安装另一套 OpenCV。无互联网时，可在电脑上取得匹配 ARM64 和 Python 3.13 的 wheel，再复制到树莓派用 pip 安装。运行时使用本目录模型，可以离线执行。

默认读取 CSI 0/1，按 Q 或 Esc 关闭预览。更换相机前可用 `rpicam-hello --list-cameras` 确认编号。同一对摄像头应由一个进程使用。启动脚本会检查已有安装的相机锁，测试时请先退出原预览。

```sh
sh start.sh --camera-check
sh start.sh --headless --frames 60
sh start.sh --independent
```

同编号的框表示左右关联。黄色表示经过局部复检，橙色问号表示两路类别冲突。编号只对应当前检测图对。快速移动时，预览可能在下一次识别前暂时隐藏旧框。

## 文件与版本隔离

| 路径 | 用途 |
| --- | --- |
| `detect.py`、`live_preview.py` | 双路相机、独立预览和识别进程 |
| `ncnn_backend.py`、`stereo_fusion.py` | 模型推理、左右匹配和补漏 |
| `model/` | NCNN 计算图、FP16 权重、输入配置和类别表 |
| `tools/export_model.py` | 在电脑端导出独立模型的脚本 |

本模块从自身目录加载模型和依赖。原 YOLOX 检测模块、v0.3 的模型与标定参数以及其历史结果仍归各自目录。试验导出请使用新的输出目录。

## 模型来源与导出

权重来自 [Ultralytics YOLOE-26s](https://docs.ultralytics.com/models/yoloe/)，发布文件为 `yoloe-26s-seg.pt`。使用 Ultralytics 8.4.215、PNNX 20260526、NCNN 1.0.20261020 转为检测版。附带权重的 SHA-256 为 `88acb54a03eeb8d200f87b2c9ac1de75aadfeec02c4adb63134134ab114cf27c`。

附带模型包含为插线板和智能手表加入的视觉提示。原始参考图片未随分支发布。已有部署可直接使用附带模型；要以自己的样例重导出，执行 `python tools/export_model.py --help`。在可联网的电脑上准备原始权重、文本编码器和导出依赖，使用新输出目录。模型相关源码、许可和使用条款见 [AGPL-3.0](ULTRALYTICS-LICENSE.txt) 与 [Ultralytics 项目](https://github.com/ultralytics/ultralytics)。
