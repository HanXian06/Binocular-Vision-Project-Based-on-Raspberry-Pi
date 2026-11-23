"""用冻结的 v0.1 配置逐对测试已有数据集，保存原始输出和分组统计。

先在 roi_annotations.json 写好物体表面矩形和拍摄记录中的参考值，再运行本文件。
本脚本只负责批量调用、汇总；测距仍由原来的 run_ranging.py 完成。
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    """读取 UTF-8 JSON，兼容文本开头带编码标记的文件。"""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path, value):
    """保留中文文字，便于直接阅读配置和统计结果。"""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def summarize(rows):
    """有输出的图片用于计算误差；没有输出的图片仍计入总数和输出率。"""
    usable = [r for r in rows if r['available']]
    return {
        'total': len(rows), 'available': len(usable),
        'output_rate': len(usable)/len(rows),
        'mean_depth_m': statistics.mean(r['depth_m'] for r in usable) if usable else None,
        'mae_m': statistics.mean(r['absolute_error_m'] for r in usable) if usable else None,
        'mean_signed_error_m': statistics.mean(r['signed_error_m'] for r in usable) if usable else None,
        'mape_percent': statistics.mean(r['relative_error_percent'] for r in usable) if usable else None,
        'max_absolute_error_m': max((r['absolute_error_m'] for r in usable), default=None),
        'mean_valid_fraction': statistics.mean(r['valid_fraction'] for r in rows),
        'within_10_percent': sum(r['relative_error_percent'] <= 10 for r in usable),
        'within_20_percent': sum(r['relative_error_percent'] <= 20 for r in usable),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/dataset_evaluation/runs')
    parser.add_argument('--check-consistency', action='store_true', help='Run the independent correspondence check')
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    annotation_path = ROOT/'evaluation/roi_annotations.json'
    annotations = load_json(annotation_path)
    template = load_json(ROOT/'configs/sample_pair0006.json')
    config_dir = output/'configs'
    config_dir.mkdir()
    # 留下执行时的标注副本与指纹，以后可核对这次究竟测试了哪些图和区域。
    save_json(output/'annotations_used.json', annotations)
    provenance = {
        'annotation_sha256': hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        'sgbm': template['sgbm'], 'region_thresholds': template['region_thresholds'],
        'camera_sha256': template['camera_sha256'],
        'consistency': {'enabled':args.check_consistency, 'max_diff_px':1.0},
        'algorithm_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'run_ranging.py', *sorted((ROOT/'stereo_ranging').glob('*.py'))]},
    }
    save_json(output/'provenance.json', provenance)
    rows = []
    for index, a in enumerate(annotations, 1):
        name = f"{a['part']}_{a['pair_id']}"
        # 每张图只改变输入路径、预先选好的 ROI 和参考记录；匹配参数全部沿用样例。
        cfg = dict(template)
        cfg.update(run_id=name, pair_id=f"{a['part']}/{a['pair_id']}",
            left_image=a['left'], right_image=a['right'],
            camera_parameters=str(ROOT/'parameters/background1105.npz'),
            roi_xywh=a['roi_xywh'], reference_distance_m=a['reference_distance_m'],
            reference_source=a['reference_source'], roi_description=a['roi_note'])
        config_path = config_dir/f'{name}.json'
        if args.check_consistency:
            cfg['consistency'] = {'enabled':True, 'max_diff_px':1.0}
        save_json(config_path, cfg)
        # 使用现有命令行入口，得到与单对测试相同的图片、数组和数值文件。
        command = [sys.executable, str(ROOT/'run_ranging.py'),
            '--config', str(config_path), '--out', str(output/name)]
        if args.check_consistency:
            command.append('--check-consistency')
        process = subprocess.run(command,
            capture_output=True, text=True, encoding='utf-8',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        (output/f'{name}.log').write_text(process.stdout+process.stderr, encoding='utf-8')
        if process.returncode:
            raise RuntimeError(f'{name} failed; see its log')
        result = load_json(output/name/'result.json')
        # 相对误差和分组字段仅加入评估表，不改变测距程序的原有输出格式。
        row = {k: a[k] for k in ('part', 'pair_id', 'object', 'nominal_distance_m', 'reference_distance_m')}
        row.update({k: result[k] for k in ('available', 'depth_m', 'absolute_error_m',
            'valid_pixels', 'roi_pixels', 'valid_fraction', 'q25_m', 'q75_m', 'iqr_m', 'compute_total_ms')})
        row['roi_xywh'] = json.dumps(a['roi_xywh'])
        row['signed_error_m'] = row['depth_m']-row['reference_distance_m'] if row['available'] else None
        row['relative_error_percent'] = row['absolute_error_m']/row['reference_distance_m']*100 if row['available'] else None
        row['reasons'] = ';'.join(result['reasons'])
        if args.check_consistency:
            row.update({f'consistency_{k}':v for k,v in result['consistency_check'].items()})
            row['before_consistency_depth_m'] = result['before_consistency']['depth_m']
        rows.append(row)
        print(f"{index}/{len(annotations)} {name}: Z={row['depth_m']} ref={row['reference_distance_m']}", flush=True)
    save_json(output/'per_pair.json', rows)
    with (output/'per_pair.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # 同类物体、同一计划距离作为一组；避免混在一起掩盖距离变化的影响。
    groups = {}
    for row in rows:
        key = f"{row['part']}/{row['object']}/{row['nominal_distance_m']:g}m"
        groups.setdefault(key, []).append(row)
    summary = {'all': summarize(rows),
        'by_object': {obj: summarize([r for r in rows if r['object'] == obj]) for obj in ('box', 'person', 'chair')},
        'groups': {key: summarize(value) for key, value in groups.items()},
        'mean_compute_ms': statistics.mean(r['compute_total_ms'] for r in rows)}
    save_json(output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
