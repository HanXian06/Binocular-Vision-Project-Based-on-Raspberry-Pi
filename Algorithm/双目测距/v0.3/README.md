# v0.3发布说明

更新者：陈珂乐。发布日期：2026-11-23。

复用原YOLOX-Nano识别，D2自动表面适配＋原v0.1 SGBM＋G1汇总，不依赖主动角点。真实43对目标结果：人物9/9、自动误差29.39cm（人工同9对27.92cm）；椅子5/9、23.42cm（人工同5对23.13cm）；书本0/25。上下行稠密搜索尚未实现。

运行前从仓库根目录执行 `python "Algorithm/双目测距/prepare_paths.py"`，解析数据集路径。模型和相机副本已包含；数据原图使用仓库Dataset，不重复打包。

发布内容含源码、配置、必要样例、统计和开发报告；大批中间数组/逐点日志/重复预览保留本地，必要时重新生成，见[发布范围](../OMITTED_ARTIFACTS.md)。需要旧SGBM数组的深入分析先运行 `python "Algorithm/双目测距/restore_baselines.py"`。历史统计和原运行指纹保留，当前发布文件校验见仓库共享清单。

# v0.3：YOLOX自动选区＋原SGBM测距

当前完成D2＋M1＋G1的本地离线闭环；已经使用仓库原模型跑完43对并与冻结人工框比较。[完整开发报告](docs/2026-11-23_v03_report.md)

## 运行

在本目录打开终端，依赖Python、NumPy和OpenCV：

```bash
python run_offline.py --out results/my_local43
python run_offline.py --pair PartC/pair_0007 --out results/my_person
python -m unittest discover -s tests -v
```

输出目录使用新名字。默认数据索引为 `configs/local_targets.json`，指向本机已有原图；换电脑时修改该索引路径。本轮汇总和逐对统计在 `results/local43_completed/`。

## 本地文件管理

| 位置 | 内容 |
| --- | --- |
| `detection/yolox_backend.py` | 另存的结构化接口，infer返回类别/坐标/分数；annotate复用结果 |
| `pipeline.py` | 原图到逐目标距离的完整接口，不读取人工框 |
| `stereo/` | 原v0.1几何和SGBM副本 |
| `roi_policy.py` | 冻结多区域协议副本 |
| `models/` | 原仓库ONNX模型，未改变权重 |
| `parameters/` | 原相机参数副本 |
| `configs/` | 检测、选区、匹配参数及本地数据索引 |
| `references/` | 原识别代码只读文本快照、来源、模型校验值和许可 |
| `results/local43_completed/` | 主评估原始结果和人工对照 |
| `results/detection_input_control/` | 原图检测对照，仅诊断漏检来源 |
| `results/_model_path_load_failure/` | 最初中文路径加载失败留下的配置记录，无推理结果 |

原仓库及v0.1、v0.2都不覆盖。本版仍不含上下行搜索、独立反查、Shi–Tomasi测距、实例分割或训练。当前入口是本地图片；`StereoPipeline.process`可接收左右BGR帧和采集时间，尚未接入树莓派相机循环或实测上板。

## 结构化输出与坐标

`Detector.infer(frame)`返回 `detection_id,class_id,class_name,confidence,bbox_xyxy`。检测框属于输入校正左图；模型的416输入按原后端左上补边，恢复框时除缩放比。

`StereoPipeline.process(left,right,pair_id,left_timestamp_ns=None,right_timestamp_ns=None)`返回：逐对象结果、左右校正图和稠密数值数组。时间缺失写null，不用文件编号伪造采集同步。

每个检测对象包含表面框、13区域统计、最终距离及状态：`distance_available`、`insufficient_depth`或`unsupported_class`。当前只对person/chair/book测距，其他类别仍在结果里保留。检测置信度与深度有效比例分别记录，不混写准确率。

## 查看结果

- [主统计](results/local43_completed/summary.json)
- [逐对自动/人工对比CSV](results/local43_completed/comparison.csv)
- [总览图](results/local43_completed/overview.jpg)
- [验证记录](results/local43_completed/verification.json)

原本地完整运行的每对目录保存 `automatic.json`（所有检测对象，不含人工纠正）、`manual_comparison.json`、`stereo_arrays.npz`、左右校正图、自动预览和对照预览。黄色是自动表面范围，紫色是仅用于对照的人工表面框。

离线目标关联会读取人工标注，以确认参考距离属于哪个检测对象；该关联只评估，真正自动选区和测距不读取人工框。目标未检出时，自动距离为空，人工结果不会补入。

配置参数解释、候选和授权范围见 [已批准方案](docs/2026-11-23_v03_plan.md)。后续按 开发约定（历史本地材料未随发布副本携带） 继续，先说明算法及备选，再按指令执行。

发布版的detection/yolox_backend.py是兼容入口，实际接口保存在原YOLO目录的新子文件夹中。不要单独移动v0.3而丢失该子文件夹。原图、数组及重复预览未全部上传，代码重新运行会生成完整输出。
