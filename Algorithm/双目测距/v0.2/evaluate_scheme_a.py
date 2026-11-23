"""v0.2方案A：固定ROI内，比较主动角点与网格选点，复用同一套NCC匹配。

默认评估43对；--pair PartC/pair_0007 可只运行一对。输出目录每次取新名字。
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
from time import perf_counter

import cv2
import numpy as np
from stereo_ranging.geometry import rectify
from stereo_ranging.local_matching import match_points,triangulate_matches
from stereo_ranging.selection import select_points

ROOT=Path(__file__).resolve().parent
METHODS=('grid_full','grid_equal_count','shi_tomasi')


def read_json(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def clean(value):
    """将NumPy标量转成普通数值，非有限数值转成JSON的null。"""
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save_json(path,value):path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2),encoding='utf-8')


def read_image(path):
    """先读字节再解码，兼容中文路径。"""
    im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if im is None:raise ValueError(f'Cannot read {path}')
    return im


def save_image(path,im):
    ok,buf=cv2.imencode(path.suffix,im)
    if not ok:raise ValueError(f'Cannot encode {path}')
    path.write_bytes(buf.tobytes())


def summarize_depth(z,total,cfg):
    """正式距离沿用旧门槛；不足门槛的中位数单列为诊断，不能当作正式输出。"""
    values=z[np.isfinite(z)&(z>0)]
    fraction=len(values)/total if total else 0.
    available=len(values)>=cfg['min_points'] and fraction>=cfg['min_fraction']
    median=float(np.median(values)) if len(values) else None
    return dict(available=bool(available),depth_m=median if available else None,
        diagnostic_median_m=median,valid_points=len(values),selected_points=total,valid_fraction=fraction)


def coverage(points,roi):
    """统计3×3分区中有点的格数；这是分布描述，不是目标像素覆盖率。"""
    x,y,w,h=roi
    return len({(min(2,(int(u)-x)*3//w),min(2,(int(v)-y)*3//h)) for u,v in points})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/scheme_a')
    parser.add_argument('--pair',help='Example: PartC/pair_0007')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(ROOT/'configs/scheme_a.json');annotations=read_json(ROOT/'configs/dataset_pairs.json')
    if args.pair:annotations=[a for a in annotations if a['part']+'/'+a['pair_id']==args.pair]
    if not annotations:raise ValueError('No pair matched')
    camera_path=ROOT/'parameters/background1105.npz'
    with np.load(camera_path) as archive:camera=dict(archive)
    save_json(out/'config_used.json',cfg);save_json(out/'annotations_used.json',annotations)
    sources=[Path(__file__).resolve(),*sorted((ROOT/'stereo_ranging').glob('*.py'))]
    save_json(out/'provenance.json',dict(version='v0.2',camera_sha256=hashlib.sha256(camera_path.read_bytes()).hexdigest(),
        source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        opencv_version=cv2.__version__,numpy_version=np.__version__))
    records=[]
    for index,a in enumerate(annotations,1):
        name=a['part']+'_'+a['pair_id'];folder=out/name;folder.mkdir()
        images=[read_image(Path(a[side])) for side in ('left','right')]
        rect,support=rectify(*images,camera);gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in rect]
        x,y,w,h=a['roi_xywh'];step=cfg['grid_step']
        grid=np.array([(u,v) for v in range(y+step//2,y+h,step) for u in range(x+step//2,x+w,step)],int).reshape(-1,2)
        # 主动点限制在与NCC窗口相容的有效图像内，图像外黑边不能产生候选。
        kernel=np.ones((cfg['matching']['patch_size'],)*2,np.uint8)
        safe=cv2.erode(support[0].astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        start=perf_counter();active=select_points(gray[0],a['roi_xywh'],len(grid),cfg['selection'],safe)
        selection_ms=(perf_counter()-start)*1000
        # 同数量网格按固定顺序均匀抽取，避免用更多点的网格与更少角点混为一谈。
        equal=grid[np.linspace(0,len(grid)-1,len(active),dtype=int)] if len(active) else np.empty((0,2),int)
        point_sets=dict(grid_full=grid,grid_equal_count=equal,shi_tomasi=active)
        row={k:a[k] for k in ('part','pair_id','object','nominal_distance_m','reference_distance_m')}
        row.update(version='v0.2',roi_xywh=a['roi_xywh'],selection_ms=selection_ms)
        for method,points in point_sets.items():
            start=perf_counter();matches=match_points(*gray,points,supports=support,**cfg['matching'])
            matching_ms=(perf_counter()-start)*1000
            good=[m for m in matches if m['accepted']]
            xyz,error=triangulate_matches([m['left_xy'] for m in good],[m['right_xy'] for m in good],camera['P1'],camera['P2'])
            for m,p,e in zip(good,xyz,error):
                m.update(xyz_rectified_m=p.tolist(),reprojection_rms_px=float(e),
                    vertical_offset_px=m['right_xy'][1]-m['left_xy'][1])
            valid=np.isfinite(xyz).all(axis=1)&(xyz[:,2]>0)
            stats=summarize_depth(xyz[:,2],len(points),cfg['region'])
            stats.update(matching_ms=matching_ms,selected_cells=coverage(points,a['roi_xywh']),
                valid_cells=coverage([m['left_xy'] for m,v in zip(good,valid) if v],a['roi_xywh']),
                rejected=Counter(m['reason'] for m in matches if not m['accepted']),
                reprojection_median_px=float(np.median(error[valid])) if valid.any() else None)
            row[method]=stats;save_json(folder/f'points_{method}.json',matches)
            canvas=rect[0].copy()
            for m in matches:
                cv2.circle(canvas,tuple(m['left_xy']),2,(0,255,0) if m['accepted'] else (0,0,255),-1)
            cv2.rectangle(canvas,(x,y),(x+w-1,y+h-1),(0,255,255),1)
            cv2.putText(canvas,f'{method}: {stats["valid_points"]}/{len(points)}',(8,22),0,.55,(0,255,255),1)
            save_image(folder/f'selection_{method}.png',canvas)
        # 所有选点和匹配完成后，才引入参考距离作误差报告。
        old=read_json(ROOT.parent/'v0.1/results/dataset_evaluation/runs'/name/'result.json')
        row['sgbm_reference']=dict(depth_m=old['depth_m'],absolute_error_m=old['absolute_error_m'])
        for method in METHODS:
            s=row[method];s['absolute_error_m']=abs(s['depth_m']-a['reference_distance_m']) if s['available'] else None
        save_json(folder/'result.json',row);records.append(row)
        print(f'{index}/{len(annotations)} {name}: points={len(active)} distance={row["shi_tomasi"]["depth_m"]}',flush=True)
    save_json(out/'per_pair.json',records)
    flat=[]
    for row in records:
        r={k:row[k] for k in ('version','part','pair_id','object','reference_distance_m','selection_ms')}
        for method in METHODS:
            r.update({method+'_'+k:v for k,v in row[method].items() if not isinstance(v,dict)})
        flat.append(r)
    with (out/'per_pair.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    summary={}
    for obj in ('all','box','person','chair'):
        rows=[r for r in records if obj=='all' or r['object']==obj];summary[obj]={}
        for method in METHODS:
            usable=[r for r in rows if r[method]['available']]
            summary[obj][method]=dict(total=len(rows),available=len(usable),
                mae_m=statistics.mean(r[method]['absolute_error_m'] for r in usable) if usable else None,
                sgbm_mae_on_same_pairs_m=statistics.mean(r['sgbm_reference']['absolute_error_m'] for r in usable) if usable else None,
                mean_selected_points=statistics.mean(r[method]['selected_points'] for r in rows),
                mean_valid_points=statistics.mean(r[method]['valid_points'] for r in rows))
        common=[r for r in rows if r['shi_tomasi']['available'] and r['grid_equal_count']['available']]
        summary[obj]['common_equal_count']=dict(pairs=len(common),
            active_mae_m=statistics.mean(r['shi_tomasi']['absolute_error_m'] for r in common) if common else None,
            grid_mae_m=statistics.mean(r['grid_equal_count']['absolute_error_m'] for r in common) if common else None)
    save_json(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
