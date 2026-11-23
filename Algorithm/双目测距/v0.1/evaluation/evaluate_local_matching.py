"""步骤1的固定数据对照入口。运行时新建结果目录，不覆盖原SGBM或反查结果。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_ranging import read_image,save_image
from stereo_ranging.geometry import rectify
from stereo_ranging.local_matching import match_points,triangulate_matches


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def clean(value):
    """JSON中以null表示没有结果，避免把NaN写成难以跨程序读取的数值。"""
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(path,value):path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2),encoding='utf-8')


def region(values,total):
    """保留原中位数汇总和30个点/10%门槛；分母为本次实际采样点数量。"""
    values=np.asarray(values);values=values[np.isfinite(values)&(values>0)]
    available=len(values)>=30 and len(values)/total>=.1
    return dict(available=bool(available),depth_m=float(np.median(values)) if available else None,
        accepted_points=int(len(values)),sample_points=total,accepted_fraction=len(values)/total)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/local_matching_step1')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    annotations=read(ROOT/'evaluation/roi_annotations.json')
    camera_path=ROOT/'parameters/background1105.npz'
    with np.load(camera_path) as f:camera=dict(f)
    settings=dict(grid_step=4,patch_size=9,max_disparity=127,min_std=5.,min_score=.75,min_margin=.03,
        vertical_radii=[0,8],min_points=30,min_fraction=.1)
    save(out/'settings.json',settings);save(out/'annotations_used.json',annotations)
    save(out/'provenance.json',dict(camera_sha256=hashlib.sha256(camera_path.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).resolve(),ROOT/'stereo_ranging/local_matching.py']},
        opencv_version=cv2.__version__,numpy_version=np.__version__))
    records=[]
    for index,a in enumerate(annotations,1):
        name=f"{a['part']}_{a['pair_id']}";folder=out/name;folder.mkdir()
        im=[read_image(Path(a[k])) for k in ('left','right')]
        rect,support=rectify(*im,camera);gray=[cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) for i in rect]
        x,y,w,h=a['roi_xywh']
        points=np.array([(u,v) for v in range(y+2,y+h,4) for u in range(x+2,x+w,4)],dtype=int)
        old=ROOT/'results/dataset_evaluation/runs'/name
        with np.load(old/'arrays.npz') as f:
            # 以相同网格读取SGBM，分开观察“采样变化”和“匹配方法变化”。
            ds=f['depth_m'][points[:,1],points[:,0]]
        base=read(old/'result.json')
        row={k:a[k] for k in ('part','pair_id','object','nominal_distance_m','reference_distance_m')}
        row.update(sgbm_full=dict(available=base['available'],depth_m=base['depth_m']),sgbm_grid=region(ds,len(points)))
        for radius in (0,8):
            start=perf_counter()
            matches=match_points(*gray,points,vertical_radius=radius,supports=support)
            elapsed=(perf_counter()-start)*1000
            good=[m for m in matches if m['accepted']]
            xyz,errors=triangulate_matches([m['left_xy'] for m in good],[m['right_xy'] for m in good],camera['P1'],camera['P2'])
            for m,p,e in zip(good,xyz,errors):
                m.update(disparity_px=m['left_xy'][0]-m['right_xy'][0],
                    vertical_offset_px=m['right_xy'][1]-m['left_xy'][1],xyz_rectified_m=p.tolist(),
                    original_left_axis_z_m=float((p@camera['R1'])[2]),reprojection_rms_px=float(e))
            stats=region(xyz[:,2],len(points))
            stats.update(matching_ms=elapsed,rejections=dict(Counter(m['reason'] for m in matches if not m['accepted'])),
                vertical_median_px=float(np.median([m['vertical_offset_px'] for m in good])) if good else None,
                reprojection_median_px=float(np.nanmedian(errors)) if len(errors) else None,
                vertical_limit_fraction=float(np.mean([m['vertical_at_limit'] for m in good])) if good else None,
                sgbm_on_same_accepted_points=region([ds[i] for i,m in enumerate(matches) if m['accepted']],len(points)),
                original_left_axis_z_diagnostic_m=float(np.median([m['original_left_axis_z_m'] for m in good])) if good else None)
            row[f'ncc_y{radius}']=stats
            save(folder/f'matches_y{radius}.json',matches)
            # 显示均匀选出的对应点，图示不按参考距离挑选。
            canvas=cv2.hconcat(rect)
            for m in good[::max(1,len(good)//18)][:18]:
                u,v=m['left_xy'];ur,vr=m['right_xy'];color=(0,255,0)
                cv2.circle(canvas,(u,v),3,color,1);cv2.circle(canvas,(round(ur)+640,round(vr)),3,color,1)
                cv2.line(canvas,(u,v),(round(ur)+640,round(vr)),(0,190,255),1)
            cv2.putText(canvas,f'{name} NCC y+/-{radius} accepted={len(good)}/{len(points)}',(8,25),0,.6,(0,255,255),2)
            save_image(folder/f'correspondences_y{radius}.jpg',canvas)
        # 到所有匹配结束后才使用参考值计算误差，参考值不会进入候选选择。
        for method in ('sgbm_full','sgbm_grid','ncc_y0','ncc_y8'):
            m=row[method];m['absolute_error_m']=abs(m['depth_m']-a['reference_distance_m']) if m['available'] else None
            m['relative_error_percent']=m['absolute_error_m']/a['reference_distance_m']*100 if m['available'] else None
        save(folder/'result.json',row);records.append(row)
        print(f"{index}/43 {name}: SGBM={row['sgbm_full']['depth_m']} NCC0={row['ncc_y0']['depth_m']} NCC8={row['ncc_y8']['depth_m']}",flush=True)
    save(out/'per_pair.json',records)
    flat=[]
    for r in records:
        f={k:v for k,v in r.items() if not isinstance(v,dict)}
        for method in ('sgbm_full','sgbm_grid','ncc_y0','ncc_y8'):
            f.update({method+'_'+k:v for k,v in r[method].items() if not isinstance(v,dict)})
        flat.append(f)
    with (out/'per_pair.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    summary={}
    for obj in ('all','box','person','chair'):
        group=[r for r in records if obj=='all' or r['object']==obj]
        summary[obj]={}
        for method in ('sgbm_full','sgbm_grid','ncc_y0','ncc_y8'):
            common=[r for r in group if r[method]['available']]
            summary[obj][method]=dict(total=len(group),available=len(common),
                mae_m=statistics.mean(r[method]['absolute_error_m'] for r in common) if common else None,
                sgbm_full_mae_on_same_pairs_m=statistics.mean(r['sgbm_full']['absolute_error_m'] for r in common) if common else None,
                severe_over20_count=sum(r[method]['relative_error_percent']>20 for r in common))
    save(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
