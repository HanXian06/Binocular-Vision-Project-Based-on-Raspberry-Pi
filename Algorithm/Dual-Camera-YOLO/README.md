# 树莓派双摄物体识别

在树莓派本地运行 YOLOX-Nano，对两路摄像头分别做物体检测，在同一个窗口并排显示画面、检测框、类别和置信度。模型使用 COCO 预训练权重，可识别人、车、杯子等 80 类常见物体。安装完成后可离线运行，默认不保存画面。两路画面独立识别，不用于双目测距。

已在 **树莓派 5、64 位 Raspberry Pi OS（Debian 13）、双 OV5647 排线摄像头**上验证。测试时两路同时运行，每路约 5 帧/秒；实际速度取决于设备和画面。

## 树莓派上的实际安装路径

本项目已部署在树莓派 `pi` 账户的 **`/home/pi/pi-dual-yolo/`**。GitHub 上的模块目录是 `Algorithm/Dual-Camera-YOLO/`。

| 树莓派上的路径 | 用途 |
| --- | --- |
| `/home/pi/pi-dual-yolo/detect.py` | 双摄识别主程序 |
| `/home/pi/pi-dual-yolo/yolox_backend.py` | YOLOX 推理与绘制检测结果 |
| `/home/pi/pi-dual-yolo/yolox_nano.onnx` | 本地模型 |
| `/home/pi/pi-dual-yolo/start.sh` | 启动双摄预览 |
| `/home/pi/pi-dual-yolo/.venv/` | 树莓派上的 Python 虚拟环境 |
| `/home/pi/Desktop/Dual-Camera-YOLO.desktop` | VNC 桌面上的启动快捷方式 |

在这台树莓派的桌面终端运行 `/home/pi/pi-dual-yolo/start.sh`，或双击 **Dual Camera YOLO**，即可启动识别。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `detect.py` | 打开两路摄像头，显示并排预览，也支持无窗口测试 |
| `yolox_backend.py` | YOLOX 推理、检测框和标签绘制 |
| `yolox_nano.onnx` | 官方 YOLOX-Nano 模型，可直接离线加载 |
| `requirements.txt` | Python 依赖 |
| `start.sh` | 启动双排线摄像头预览 |
| `Dual-Camera-YOLO.desktop` | 可选的桌面快捷方式 |
| `YOLOX-LICENSE.txt` | 上游 YOLOX 项目的 Apache 2.0 许可文本 |

## 安装

将仓库文件放到树莓派的 `~/pi-dual-yolo`。如果只用 USB 摄像头，可省略 `python3-picamera2`。

```bash
cd ~/pi-dual-yolo
sudo apt update
sudo apt install -y python3-venv python3-tk python3-picamera2 v4l-utils
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

`--system-site-packages` 让虚拟环境可以使用系统安装的 Picamera2。安装依赖需要联网；如果树莓派只通过网线直连电脑且无法上网，可以先在另一台电脑下载与树莓派架构及 Python 版本匹配的依赖包，再复制过去离线安装。模型已随仓库提供。

## 确认摄像头

| 摄像头类型 | 查看设备 | 传给程序的参数示例 |
| --- | --- | --- |
| 两个排线摄像头 | `rpicam-hello --list-cameras` | `csi:0 csi:1` |
| 两个 USB 摄像头 | `v4l2-ctl --list-devices` | `usb:/dev/video0 usb:/dev/video2` |
| 排线和 USB 混用 | 分别运行上面两条命令 | `csi:0 usb:/dev/video0` |

USB 设备编号因机器而异。一个 USB 摄像头可能对应多个节点，请选择它的实际视频采集节点；也可以使用 `usb:/dev/v4l/by-id/...` 指定设备。先验证两路都能取到画面：

```bash
.venv/bin/python detect.py csi:0 csi:1 --camera-check
```

## 运行

在树莓派桌面终端（例如通过 VNC 打开）执行：

```bash
sh start.sh
```

默认使用 `csi:0 csi:1`，按 **Q** 或 **Esc** 退出。其他摄像头组合请直接运行 `.venv/bin/python detect.py` 并指定实际设备。可加 `--conf 0.25` 降低置信度门槛，检出更多低置信度目标的同时也可能增加误报。

通过 SSH 验证时可以不打开窗口，只运行三对画面：

```bash
.venv/bin/python detect.py csi:0 csi:1 --headless --frames 3
```

若项目放在 `~/pi-dual-yolo`，可选装桌面图标：

```bash
install -m 755 Dual-Camera-YOLO.desktop "$(xdg-user-dir DESKTOP)/Dual-Camera-YOLO.desktop"
```

如果摄像头提示忙，请先退出正在使用摄像头的预览程序。如果模型没有框出某件物体，可用光线充足的场景和水瓶等常见类别测试。识别特定零件或缺陷需要对应的训练数据和模型。

## 上游项目

- [YOLOX 官方 ONNX 推理说明与模型下载](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime)
- [树莓派摄像头软件文档](https://www.raspberrypi.com/documentation/accessories/camera-software.html)

随仓库附带的模型来自 YOLOX 官方发布。上游许可证见 [YOLOX-LICENSE.txt](YOLOX-LICENSE.txt)，该文件用于说明上游项目的许可。
