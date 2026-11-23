"""用上一轮保存的SIFT候选核对局部匹配位置；候选只作参考，不作为人工真值。"""
import argparse
from pathlib import Path
import sys
import statistics
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.evaluate_local_matching import read,save
from run_ranging import read_image
from stereo_ranging.local_matching import match_points

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/local_matching_step1/comparison/feature_comparison.json')
    args=parser.parse_args()
    sources=read(ROOT/'results/consistency_evaluation/analysis_final/feature_correspondences.json')
    records=[]
    for source in sources:
        features=source['matches']
        if not features:continue
        # 固定按保存顺序均匀取最多20个参考候选，不按候选深度或本轮匹配结果挑选。
        indices=np.linspace(0,len(features)-1,min(20,len(features)),dtype=int)
        features=[features[i] for i in indices]
        path=ROOT/'results/dataset_evaluation/runs'/source['pair']
        images=[cv2.cvtColor(read_image(path/f'{s}_rectified.png'),cv2.COLOR_BGR2GRAY) for s in ('left','right')]
        points=[tuple(round(v) for v in f['left_xy']) for f in features]
        # 将SIFT位移平移到本轮整数左点，避免直接比较不同的左侧坐标。
        reference=[np.array(p)+np.array(f['right_xy'])-np.array(f['left_xy']) for p,f in zip(points,features)]
        row=dict(pair=source['pair'],sample_count=len(points))
        for radius in (0,8):
            matches=match_points(*images,points,vertical_radius=radius)
            row[f'y{radius}']=[dict(left_xy=m['left_xy'],reference_right_xy=ref.tolist(),
                right_xy=m['right_xy'],accepted=m['accepted'],reason=m['reason'],
                coordinate_gap_px=float(np.linalg.norm(np.array(m['right_xy'])-ref)) if m['right_xy'] else None)
                for m,ref in zip(matches,reference)]
        records.append(row)
    summary={}
    for part in ('PartB','PartC'):
        common=[(a,b) for r in records if r['pair'].startswith(part) for a,b in zip(r['y0'],r['y8']) if a['accepted'] and b['accepted']]
        summary[part]=dict(common_accepted_points=len(common),
            y0_median_gap_px=statistics.median(a['coordinate_gap_px'] for a,b in common) if common else None,
            y8_median_gap_px=statistics.median(b['coordinate_gap_px'] for a,b in common) if common else None)
    save(args.out,dict(note='与旧SIFT候选的二维位置差；不是人工对应真值误差。人物候选区域含外套，与测距ROI分开。',summary=summary,records=records))
    print(summary)

if __name__=='__main__':main()
