# v0.2发布说明

更新者：陈珂乐。发布日期：2026-11-23。

Shi–Tomasi主动角点＋局部NCC上下±8搜索＋三角化。选点效率提高，共同样本精度基本持平；初始远书本因30点门槛失去输出。固定ROI候选合并后，书本25/25、13.83cm，人物9/9、24.68cm，椅子0/9。稀疏路线不作为当前主线。

运行前从仓库根目录执行 `python "Algorithm/双目测距/prepare_paths.py"`，解析数据集路径。模型和相机副本已包含；数据原图使用仓库Dataset，不重复打包。

发布内容含源码、配置、必要样例、统计和开发报告；大批中间数组/逐点日志/重复预览保留本地，必要时重新生成，见[发布范围](../OMITTED_ARTIFACTS.md)。需要旧SGBM数组的深入分析先运行 `python "Algorithm/双目测距/restore_baselines.py"`。历史统计和原运行指纹保留，当前发布文件校验见仓库共享清单。

# v0.2：方案A，主动选点后匹配


2026-11-23 已继续完成9对椅子的7类ROI对照，详见 [椅子ROI排查](docs/chair_roi_review.md)。扩大网状内部会增加错误深度；边框网格在6对中出现可输出候选，尚未形成自动稳定测距规则。

用户已批准的新版本。使用 **Shi–Tomasi选点＋局部NCC＋上下±8像素搜索**。已完成现有43对图片的固定对照，详见 [开发反馈](docs/result_review.md)。

当前收益是减少匹配点数、提高相同点预算下的有效点数量；测距精度没有显著提高，椅子仍不能可靠输出。v0.1保留在原位置。

## 怎么运行

在本目录打开终端。完整43对：

```bash
python evaluate_scheme_a.py --out results/my_scheme_a
```

只处理一对，例如2米人物：

```bash
python evaluate_scheme_a.py --pair PartC/pair_0007 --out results/my_person
```

输出目录应使用新名字。程序读取 `configs/dataset_pairs.json` 中本机原图路径；旧SGBM对照读取相邻 `v0.1/results/dataset_evaluation/runs/`。相机参数及NCC、几何模块在本版本内保存独立副本。

## 程序分工

| 文件 | 工作 |
| --- | --- |
| `stereo_ranging/selection.py` | Shi–Tomasi主动选点、点间距及3×3分区名额 |
| `stereo_ranging/local_matching.py` | 复用v0.1的NCC匹配和二维坐标三角化 |
| `stereo_ranging/geometry.py` | 复用固定相机参数校正 |
| `evaluate_scheme_a.py` | 读取输入，执行三组对照，汇总距离、误差和图示 |
| `configs/scheme_a.json` | 本轮选点、匹配和输出门槛 |
| `configs/dataset_pairs.json` | 与v0.1相同的43对原图、ROI和参考值 |

## 先理解哪些参数

- `selection.quality_level=0.01`：角点响应相对于ROI内强候选的最低比例，不是匹配准确率。
- `selection.min_distance=3`：角点之间至少间隔3像素。
- `selection.block_size=3`：计算角点响应的局部窗口。
- `selection.cells=3`：把ROI划分为3×3区域，每格限制选点名额；总预算不超过原网格点数。
- `matching`：与v0.1的二维NCC相同，9×9图块、上下±8像素、横向正视差1～127，相关度≥0.75、最佳与次佳差≥0.03、纹理标准差≥5。
- `region`：至少30个有效点、有效点占选点数至少10%才正式输出距离；未达门槛的中位数单独保存为诊断值。

本轮固定参数完成比较。后续更改方案先说明原理及范围，用户批准后实施。

## 如何看结果

`results/scheme_a/` 是本轮正式结果，包含逐对CSV、统计JSON、运行配置与源文件指纹。每对图片目录中：

- `points_grid_full.json`：原4像素网格的候选、得分和匹配结果。
- `points_grid_equal_count.json`：与主动角点数相同的均匀网格对照。
- `points_shi_tomasi.json`：主动选点及对应点、拒绝原因、XYZ和重投影残差。
- `selection_*.png`：绿色表示通过NCC的左点，红色表示未通过，黄色框是原ROI。绿色不代表深度准确率。
- `result.json`：三种方式的正式距离与诊断中位数、数量、分区分布、耗时；深度仍为矫正左相机Z。

选点只看校正左图，不知道参考距离。NCC需要左右图寻找对应；所有匹配完成后才使用采集参考记录计算误差。没有加入反查、背景排除、深度分组、YOLO或实例分割。

## 必要验证

```bash
python -m unittest discover -s tests -v
```

8项验证包括选点范围、间隔、数量上限、确定性、纯色区域、图像支持、NCC已知位移、重复候选及三角化。批量运行另外核对了旧网格结果与v0.1完全一致。
