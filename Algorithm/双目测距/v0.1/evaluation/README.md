# v0.1 数据集评估操作

步骤1局部对应实验入口：`python evaluation/evaluate_local_matching.py --out results/my_local_comparison`。包含原SGBM、同网格SGBM、同一行NCC和上下±8像素NCC，结果解释见 [步骤1反馈](../docs/local_matching_step1_review.md)。它不更改原主程序。

本目录用于离线测试和汇总，测距算法仍在 `stereo_ranging/`，版本仍为 v0.1。

1. `roi_annotations.json` 保存43对图片的原图路径、目标表面框和参考距离。坐标是在**校正左图**上的 `[x,y,宽,高]`，不是原图坐标。该文件在本轮计算距离前固定。
2. 在 `Algorithm/双目测距/v0.1` 打开终端，运行：

   ```bash
   python evaluation/evaluate_dataset.py --out results/dataset_evaluation_repeat
   ```

3. 等待逐对完成，查看新目录中的 `per_pair.csv` 和 `summary.json`。新目录下每对图片都有完整原始输出，`configs/` 保存本次逐对配置，`annotations_used.json` 保存执行时的标注副本。

输出目录应取一个尚未使用的名字。路径当前指向本机数据集；换电脑时修改标注中的 `left`、`right` 路径。参考距离只用于计算结束后的误差报告，不参与匹配和深度估计。

本轮原始结果位于 `results/dataset_evaluation/runs/`，解释与建议见 [评估报告](../docs/dataset_evaluation.md)。`roi_annotation_draft.json` 是选框过程中的草稿，不是执行输入；实际执行输入为 `roi_annotations.json`。

## 独立左右对应检查

在批量命令中加入 `--check-consistency`，例如：

```bash
python evaluation/evaluate_dataset.py --check-consistency --out results/my_consistency_batch
```

本轮结果已保存于 `results/consistency_evaluation/runs/`。`before_consistency` 保留原方法的统计，`consistency_check` 记录通过情况；实际配置以每对目录中的 `config_resolved.json` 为准。

原因分析脚本：

```bash
python evaluation/analyze_errors.py --out results/my_error_analysis
```

它读取本轮固定的原结果及检查后结果，另存坐标方向诊断、SIFT图案对应、背景稳定性等证据，均不改写测距值。完整结论见 [检查与原因分析报告](../docs/consistency_check_review.md)。
