"""固定选点与匹配算法，比较椅子原窄ROI、上部带、内部和较完整椅背。

候选区在匹配前依据左图外观固定。参考距离量到上端，扩大到椅背内部后只作为近似参考。
"""
import argparse
from collections import Counter
from pathlib import Path
import csv
import hashlib
import numpy as np
import cv2
from evaluate_scheme_a import ROOT,read_json,save_json,read_image,save_image,summarize_depth
from stereo_ranging.geometry import rectify
from stereo_ranging.selection import select_points
from stereo_ranging.local_matching import match_points,triangulate_matches

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/chair_roi_review/runs')
    parser.add_argument('--roi-config',type=Path,default=ROOT/'configs/chair_roi_candidates.json')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(ROOT/'configs/scheme_a.json');annotations=read_json(args.roi_config)
    save_json(out/'annotations_used.json',annotations);save_json(out/'config_used.json',cfg)
    save_json(out/'provenance.json',dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        roi_config_sha256=hashlib.sha256(args.roi_config.read_bytes()).hexdigest()))
    with np.load(ROOT/'parameters/background1105.npz') as f:camera=dict(f)
    records=[]
    for a in annotations:
        name='PartC_'+a['pair_id'];folder=out/name;folder.mkdir()
        rect,support=rectify(*[read_image(Path(a[k])) for k in ('left','right')],camera)
        gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in rect]
        safe=cv2.erode(support[0].astype(np.uint8),np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        original=a['roi_xywh'];fixed_budget=len(range(original[0]+2,original[0]+original[2],4))*len(range(original[1]+2,original[1]+original[3],4))
        for label,roi in a['candidate_rois'].items():
            x,y,w,h=roi
            grid=np.array([(u,v) for v in range(y+2,y+h,4) for u in range(x+2,x+w,4)],int).reshape(-1,2)
            sets={'active_area':select_points(gray[0],roi,len(grid),cfg['selection'],safe),
                'active_fixed':select_points(gray[0],roi,fixed_budget,cfg['selection'],safe),'grid':grid}
            for method,points in sets.items():
                matches=match_points(*gray,points,supports=support,**cfg['matching'])
                good=[m for m in matches if m['accepted']]
                xyz,error=triangulate_matches([m['left_xy'] for m in good],[m['right_xy'] for m in good],camera['P1'],camera['P2'])
                for m,p,e in zip(good,xyz,error):m.update(xyz_rectified_m=p.tolist(),reprojection_rms_px=float(e))
                stats=summarize_depth(xyz[:,2],len(points),cfg['region'])
                row=dict(pair=a['pair_id'],nominal_m=a['nominal_distance_m'],reference_m=a['reference_distance_m'],
                    roi=label,roi_xywh=roi,method=method,**stats,
                    reasons=dict(Counter(m['reason'] for m in matches if not m['accepted'])))
                # 参考值不参与匹配；椅背内部与上沿不是完全相同表面，此差值仅是参考比较。
                row['difference_from_top_reference_m']=abs(stats['depth_m']-a['reference_distance_m']) if stats['available'] else None
                records.append(row);save_json(folder/f'{label}_{method}_matches.json',matches)
                if method=='active_area':
                    im=rect[0].copy();cv2.rectangle(im,(x,y),(x+w-1,y+h-1),(0,255,255),1)
                    for m in matches:cv2.circle(im,tuple(m['left_xy']),2,(0,255,0) if m['accepted'] else (0,0,255),-1)
                    cv2.putText(im,f'{label}: {stats["valid_points"]}/{len(points)} Z={stats["depth_m"]}',(8,25),0,.5,(0,255,255),1)
                    save_image(folder/f'{label}_points.jpg',im)
            print(a['pair_id'],label,[(r['method'],r['valid_points'],r['depth_m']) for r in records[-3:]],flush=True)
    save_json(out/'per_roi.json',records)
    flat=[{k:(str(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in records]
    with (out/'per_roi.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

if __name__=='__main__':main()
