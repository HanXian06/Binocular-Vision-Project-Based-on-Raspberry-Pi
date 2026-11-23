# v0.1发布说明

更新者：陈珂乐。发布日期：2026-11-23。

原StereoSGBM全图视差＋Q深度＋人工表面中位数；后续含可选独立反查及局部NCC实验。初始43对全有输出，但椅子错误严重；独立反查未改善整体结果。固定ROI最终评估：书本25/25、误差13.14cm；人物9/9、27.92cm；椅子9/9、40.84cm。

运行前从仓库根目录执行 `python "Algorithm/双目测距/prepare_paths.py"`，解析数据集路径。模型和相机副本已包含；数据原图使用仓库Dataset，不重复打包。

发布内容含源码、配置、必要样例、统计和开发报告；大批中间数组/逐点日志/重复预览保留本地，必要时重新生成，见[发布范围](../OMITTED_ARTIFACTS.md)。需要旧SGBM数组的深入分析先运行 `python "Algorithm/双目测距/restore_baselines.py"`。历史统计和原运行指纹保留，当前发布文件校验见仓库共享清单。

# v0.1：离线双目区域测距

2026-11-21 已完成步骤1的局部二维匹配对照，详见 [步骤1结果与下一步建议](docs/local_matching_step1_review.md)。新方法作为独立实验入口保存，原SGBM主流程不变。书本整体未改善，部分人物改善，椅背未获得足够点；步骤2等待用户批准。

输入一对左右原图，固定相机参数，经校正、StereoSGBM、Q 重建，输出人工选定区域的前方深度。首个样例已运行完成，详见 [开发结果](docs/result_review.md)。2026-11-20 已完成43对已采集测距图片的固定参数评估，详见 [多组评估报告](docs/dataset_evaluation.md) 与 [批量运行说明](evaluation/README.md)。纸盒整体偏近，部分人物和椅子发生错配；改进建议已整理，等待审阅后执行。

同日已按指令加入独立的左右对应检查，完成43对前后对比及误差原因分析，详见 [检查与原因分析报告](docs/consistency_check_review.md)。本次检查拦下2张，但整体误差未改善；上下错位、坐标方向差异、椅背深度混合和人物拍摄变化已分别记录。

## 直接运行

在 `Algorithm/双目测距/v0.1` 文件夹打开终端：

```bash
python run_ranging.py --out results/my_run_01
```

路径有空格或中文可正常读取，输出目录必须是新目录。程序默认读取 `configs/sample_pair0006.json`，ROI 已配置，可不打开窗口直接复现。

鼠标重新框选目标：

```bash
python run_ranging.py --select-roi --out results/my_roi_01
```

在左侧校正图上拖出矩形，按 Enter 或空格确认。选择盒子正面实体表面，避开手、椅子及边缘混合区域；Esc/取消不产生距离结果。实际选择保存在该次输出的 `config_resolved.json`，原始配置不被覆盖。

再次复现自己选择的 ROI：

```bash
python run_ranging.py --config results/my_roi_01/config_resolved.json --out results/my_roi_repeat
```

只生成校正图查看构图：

```bash
python run_ranging.py --prepare --out results/rectified_only
```

启用独立的右图反查，往返误差门槛默认1像素：

```bash
python run_ranging.py --check-consistency --out results/my_consistency_01
```

开关也会写进实际配置，后续直接用 `config_resolved.json` 即可复现。新增模块为 `stereo_ranging/consistency.py`；原有命令不加开关时继续使用原方法。

## 配置中可以改什么

| 字段 | 当前值 | 意义 |
| --- | --- | --- |
| `roi_xywh` | `[270,245,100,55]` | 左校正图上 x、y、宽、高；左闭右开 |
| `sgbm.num_disparities` | 128 | 视差搜索范围宽度，为16的正倍数 |
| `sgbm.block_size` | 5 | 奇数窗口大小 |
| `sgbm.p1 / p2` | 200 / 800 | 匹配平滑惩罚，要求0<P1<P2；窗口变更后可相应调整 |
| `sgbm.uniqueness_ratio` | 10 | 候选唯一性要求 |
| `sgbm.disp12_max_diff` | 1 | SGBM内置一致性阈值；本版没有额外计算独立右视差图 |
| `sgbm.speckle_window_size / range` | 100 / 2 | 散斑过滤参数 |
| `region_thresholds.min_pixels` | 30 | 输出区域距离所需的最少有效像素数 |
| `region_thresholds.min_fraction` | 0.1 | 输出区域距离所需的最低有效比例 |
| `visualization_depth_max_m` | 4 | 只影响深度预览色标，超过4米仅显示饱和，不从统计中排除 |
| `reference_distance_m` | 1.0 | 仅在计算结束后用于报告误差，不参与匹配、过滤或修正 |

建议复制配置为新文件后修改，每次主要改变一个因素。发布配置中的repo://先用prepare_paths.py解析；自行配置的相对路径按配置文件位置解析。

## 算法文件与数据

```text
run_ranging.py                  # 命令行、交互选框、结果与可视化
stereo_ranging/
  geometry.py                   # 固定参数校正、Q深度重建
  matching.py                   # SGBM、16倍视差缩放、有效支撑
  region.py                     # ROI深度中位数、有效率、四分位距
configs/sample_pair0006.json    # 完整起步配置
parameters/background1105.npz  # 冻结旧背景相机参数
data/sample/                   # PartB/pair_0006的两张原图
results/sample_pair0006/        # 本次审阅结果
tests/test_numerics.py          # 必要数值与功能验证
docs/                          # 实现计划、验证记录、结果解释
```

原图来源：仓库提交 `03445cf5d2eadb918949b145bb2907336ecc12db` 的 `Dataset/Data1105/PartB/`。参数复制自已整理的 `background1105`，基线0.0585米，校正焦距约726.99像素。参数SHA写在配置和结果中，程序核对SHA后使用，不自动切换参数。代码不改标定文件。

## 中间结果与输出

- `left_rectified.png / right_rectified.png`：实际参与匹配的640×480校正图。
- `arrays.npz`：原始16倍视差、像素视差、米单位深度、有效掩码及左右图像支持掩码。
- `disparity_preview.png / depth_preview.png`：观察用色彩图；黑色为无效，不表示零距离。
- `valid_mask.png`：最终有效区域。
- `result_preview.png`：左图ROI及估计深度，右边为深度预览。
- `result.json / result.csv`：区域中位数、有效数/比例、Q25/Q75/IQR、参考差、分段耗时及输入哈希。
- `config_resolved.json`：绝对输入路径、实际ROI、全部参数，可直接再次运行。

深度是校正后左相机坐标的 Z，不是三维点到相机的直线距离。本版把ROI内有效深度的中位数作为表面深度，不做语义分割。有效数量/比例达到门槛时 `available=true`，这是可计算状态，不是精度概率。

过滤包括 SGBM 无效标记、非正视差、非有限/非正深度、窗口边界，以及右图对应位置 `x-d` 的真实图像支持。匹配窗口的支持区域经侵蚀，避免黑边混入。没有根据参考距离限制输出范围或拟合补偿系数。

## 环境与必要验证

本次运行：Python3.13，OpenCV5.0.0，NumPy2.5.4。依赖范围写在 requirements.txt；需要鼠标选框时使用具备窗口功能的 OpenCV，在图形桌面运行。

```bash
python -m unittest discover -s tests -v
```

5项验证覆盖米单位重建、无效值及数量门槛、右侧对应位置掩码、已知8像素位移的合成SGBM结果和ROI边界。真实样例重复运行，视差、深度、掩码及区域统计一致。

当前仍处于 v0.1 初版开发。单对实现、PartB 多组评估和后续改进都在本版中推进，整体版本变更时才新增版号。

ROI 按当前方式选择待测物体表面。43对多组评估已完成；输出信息及参数调整建议见评估报告，先供用户审阅，获批准后执行修改。

代码注释使用中文，解释变量、输入输出和处理原因；尽量按清楚的操作步骤组织，避免使用空泛提示。
