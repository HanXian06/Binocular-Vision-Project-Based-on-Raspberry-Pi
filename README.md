# Binocular-Vision-Project-Based-on-Raspberry-Pi

基于树莓派的双目视觉实践项目，目标是实现双目测距与目标识别。

## 当前阶段与共享资料

当前仅开展**相机标定与结果核查**，标定尚未验收；**识别和测距算法均未正式开始**。两套共享参数均为试验结果，没有指定正式默认参数，不能直接当作可靠测距配置。

- [标定共享包](Calibration/Calibration1105/README.md)：原图、试标定参数、误差记录、校正图与复现脚本。
- [标定方法、参数格式与限制](Calibration/Calibration1105/docs/标定说明.md)。
- [后续开发进程与指令约定](Calibration/Calibration1105/docs/开发进程与指令约定.md)。

建议顺序：标定核查与验收 → 视差 → 人工区域测距 → 可靠性判断 → 独立识别 → 融合 → 树莓派部署与评估。每一步先解释方案，由项目负责人选择并明确授权后执行；更新后说明变化、方法、输入输出及验证结果，不自动推进下一步。当前优先利用已有数据，不新增非必要采集任务。

### 标定结果摘要

Data1105 PartA 共 20 对，每路 640×480，内角点 8×6，格长 15 mm，实测镜头中心间距 58.5 mm。

| 方法 | 当前结果 | 状态 |
| --- | --- | --- |
| 棋盘格直接试标定 | 单目 RMS 约 0.349/0.343 px；双目 RMS 约 3.975 px；试算基线约 27.99 mm，与实测不符 | 未验收 |
| 背景约束实验 | 采用棋盘格内参，以背景 SIFT/RANSAC 估计相对姿态；58.5 mm 为输入尺度；拟合内点上下偏差中位数约 0.51 px | 未验收；拟合内点指标不等于整体精度 |

## 仓库结构

```text
Calibration/Calibration1105/            # 标定共享包，参数尚未验收
Algorithm/Algo_Capture/
  capture_pairs(dualcam intergreted).py   # 双目预览与采集
Dataset/
  Data1029/                             # 20 对图片及早期采集说明
    left/
    right/
  Data1105/
    PartA(1-8)/                         # 静态棋盘格检查，8 对
    PartA(9-20)/                        # 补充标定图，12 对
    PartB/                              # 已知距离纸板，25 对
    PartC/                              # 人、椅子静态场景，18 对
    下一轮任务单.md
```

各 Part 内均按 `left/`、`right/` 分别存放图片。

## 使用方式

在具备双相机、图形桌面及 Picamera2、OpenCV（cv2）、NumPy 的树莓派 Python 环境中，从仓库根目录运行：

```bash
python "Algorithm/Algo_Capture/capture_pairs(dualcam intergreted).py" --out "Dataset/new_capture_01"
```

`--out` 必须指定尚不存在的目录。默认左相机编号为 0、右相机为 1；确认左右接反时可加 `--left-camera 1 --right-camera 0`。

点击图像窗口后操作：

- **S**：拍摄并定格；再次按 **S** 确认保存。
- **W**：放弃定格照片，返回预览。
- **Q**：退出，未确认的照片不保存。

脚本每路采集 640×480 PNG，并生成 `metadata/`、`capture_setup.json` 和 `measurements.csv`。

## 注意事项

- 左右相机依次采集，并非同步拍摄；默认正式拍摄约需 4 秒或更久，期间保持相机和目标静止。
- Data1105 任务单记录：棋盘格为 8×6 内角点、格长 15 mm、镜头中心间距 58.5 mm；使用前核实与实际采集配置一致。
