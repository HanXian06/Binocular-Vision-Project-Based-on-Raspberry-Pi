"""在两种NCC共同保留的同一批左点上比较深度，单独观察对应位置变化的影响。"""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results/local_matching_step1/comparison')
    args=parser.parse_args();p=args.results
    read=lambda f:json.loads(f.read_text(encoding='utf-8'))
    paired=[]
    for row in read(p/'per_pair.json'):
        name=f"{row['part']}_{row['pair_id']}"
        a,b=(read(p/name/f'matches_y{radius}.json') for radius in (0,8))
        common=[(u,v) for u,v in zip(a,b) if u['accepted'] and v['accepted']
            and u['xyz_rectified_m'][2]>0 and v['xyz_rectified_m'][2]>0]
        if len(common)>=30 and len(common)/len(a)>=.1:
            z0=float(np.median([u['xyz_rectified_m'][2] for u,v in common]))
            z8=float(np.median([v['xyz_rectified_m'][2] for u,v in common]))
            paired.append(dict(pair=name,object=row['object'],points=len(common),depth_y0=z0,depth_y8=z8,
                error_y0=abs(z0-row['reference_distance_m']),error_y8=abs(z8-row['reference_distance_m'])))
    summary={}
    for obj in ('box','person','chair'):
        group=[r for r in paired if r['object']==obj]
        summary[obj]=dict(pairs=len(group),
            mae_y0=statistics.mean(r['error_y0'] for r in group) if group else None,
            mae_y8=statistics.mean(r['error_y8'] for r in group) if group else None)
    (p/'same_points_control.json').write_text(json.dumps(dict(summary=summary,per_pair=paired),ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary)

if __name__=='__main__':main()
