# Binocular-Vision-Project-Based-on-Raspberry-Pi

基于树莓派的双目视觉实践项目，目标是实现双目测距与目标识别。

## 仓库结构

```text
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
