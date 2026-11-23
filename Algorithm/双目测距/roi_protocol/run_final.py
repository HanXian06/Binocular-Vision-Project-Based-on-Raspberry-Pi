"""冻结ROI协议的独立最终评估；每个版本仅允许一次，旧版本算法本身不修改。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
import cv2
import numpy as np
from policy import regions,contains,stats,POLICY

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def clean(v):
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def save(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2),encoding='utf-8')

def image(p):return cv2.imdecode(np.frombuffer(Path(p).read_bytes(),np.uint8),1)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--version',choices=['v0.1','v0.2'],required=True)
    args=parser.parse_args();version=args.version;vp=ROOT.parent/version
    sys.path.insert(0,str(vp))
    from stereo_ranging.geometry import rectify,depth_from_disparity
    if version=='v0.1':
        from stereo_ranging.matching import compute_disparity,valid_geometry
        cfg=read(vp/'configs/sample_pair0006.json')
    else:
        from stereo_ranging.selection import select_points
        from stereo_ranging.local_matching import match_points,triangulate_matches
        cfg=read(vp/'configs/scheme_a.json')
    targets=read(ROOT/'frozen_targets.json');freeze=read(ROOT/'freeze.json')
    assert hashlib.sha256((ROOT/'frozen_targets.json').read_bytes()).hexdigest()==freeze['target_sha256']
    assert hashlib.sha256((ROOT/'policy.py').read_bytes()).hexdigest()==freeze['policy_sha256']
    ledger=read(ROOT/'call_budget.json')
    if ledger['independent_final_used'][version]>=1:raise RuntimeError('Final evaluation budget exhausted')
    out=ROOT/'results'/version;out.mkdir(exist_ok=False)
    # 在推理前记账；失败也保留调用记录，禁止默默重试。
    ledger['independent_final_used'][version]+=1
    event=dict(version=version,kind='independent_final',status='running',pairs_completed=0,matching_calls=0)
    ledger['events'].append(event);save(ROOT/'call_budget.json',ledger)
    with np.load(vp/'parameters/background1105.npz') as f:camera=dict(f)
    save(out/'provenance.json',dict(version=version,roi_policy=POLICY,freeze=freeze,config=cfg,
        camera_sha256=hashlib.sha256((vp/'parameters/background1105.npz').read_bytes()).hexdigest()))
    rows=[]
    for a in targets:
        start=perf_counter();ims,support=rectify(image(a['left']),image(a['right']),camera)
        gray=[cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) for i in ims];rois=regions(a['surface_box'])
        surface=np.zeros(gray[0].shape,np.uint8);x,y,w,h=rois['full'];surface[y:y+h,x:x+w]=1
        block=cfg['sgbm']['block_size'] if version=='v0.1' else cfg['matching']['patch_size']
        # 所有子区共享完整表面支持，不让九宫格的人工分界截断同一物体的窗口。
        safe=cv2.erode(surface,np.ones((block,block),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        if version=='v0.1':
            raw,d=compute_disparity(*gray,cfg['sgbm']);z=depth_from_disparity(d,camera['Q'])
            valid=valid_geometry(d,raw,z,support,cfg['sgbm']);z[~valid]=np.nan
            yy,xx=np.indices(z.shape);points=np.column_stack([xx[safe],yy[safe]]);values=z[safe]
        else:
            safe&=cv2.erode(support[0].astype(np.uint8),np.ones((block,block),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            all_points=[]
            for roi in rois.values():
                x,y,w,h=roi;budget=len(range(y+2,y+h,4))*len(range(x+2,x+w,4))
                all_points.extend(select_points(gray[0],roi,budget,cfg['selection'],safe).tolist())
            points=np.array(sorted(set(map(tuple,all_points))),int).reshape(-1,2)
            matches=match_points(*gray,points,supports=support,**cfg['matching']);accepted=[i for i,m in enumerate(matches) if m['accepted']]
            xyz,error=triangulate_matches([matches[i]['left_xy'] for i in accepted],[matches[i]['right_xy'] for i in accepted],camera['P1'],camera['P2'])
            values=np.full(len(points),np.nan)
            for i,p,e in zip(accepted,xyz,error):
                values[i]=p[2];matches[i].update(xyz_rectified_m=p.tolist(),reprojection_rms_px=float(e))
        event['matching_calls']+=1
        final=stats(values,np.ones(len(values),bool));regional={k:stats(values,contains(points,r)) for k,r in rois.items()}
        row={k:a[k] for k in ['part','pair_id','object','reference_distance_m','surface_box','box_source']}
        row.update(version=version,final=final,regions=regional,compute_ms=(perf_counter()-start)*1000)
        final['absolute_error_m']=abs(final['depth_m']-a['reference_distance_m']) if final['available'] else None
        rows.append(row);folder=out/(a['part']+'_'+a['pair_id']);folder.mkdir();save(folder/'result.json',row)
        np.savez_compressed(folder/'samples.npz',left_xy=points,depth_m=values)
        if version=='v0.2':save(folder/'matches.json',matches)
        view=ims[0].copy()
        for name,r in rois.items():
            x,y,w,h=r;cv2.rectangle(view,(x,y),(x+w-1,y+h-1),(0,220,0) if name=='full' else (0,180,255),1)
        cv2.putText(view,f'{version} Z={final["depth_m"]}',(8,25),0,.6,(0,255,255),2)
        ok,b=cv2.imencode('.jpg',view);assert ok;(folder/'regions.jpg').write_bytes(b.tobytes())
        event['pairs_completed']+=1;save(ROOT/'call_budget.json',ledger)
        print(version,event['pairs_completed'],a['part'],a['pair_id'],final['depth_m'],flush=True)
    save(out/'per_pair.json',rows)
    summary={}
    for obj in ['all','box','person','chair']:
        group=[r for r in rows if obj=='all' or r['object']==obj];good=[r for r in group if r['final']['available']]
        summary[obj]=dict(total=len(group),available=len(good),mae_m=statistics.mean(r['final']['absolute_error_m'] for r in good) if good else None)
    save(out/'summary.json',summary);event['status']='complete';save(ROOT/'call_budget.json',ledger);print(summary)

if __name__=='__main__':main()
