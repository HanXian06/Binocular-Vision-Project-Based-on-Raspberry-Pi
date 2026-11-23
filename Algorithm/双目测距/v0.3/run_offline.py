"""v0.3本地YOLO测距评估。默认43对，可用--pair只运行一对。

自动路径不读取人工表面框；人工框在自动推理完成后用于独立对照和目标关联评估。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import cv2
import numpy as np
from pipeline import StereoPipeline,measure_surface

ROOT=Path(__file__).resolve().parent

def read_json(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def clean(v):
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,float) and not np.isfinite(v):return None
    return v

def save_json(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2),encoding='utf-8')

def read_image(p):
    im=cv2.imdecode(np.frombuffer(Path(p).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if im is None:raise ValueError(f'无法读取图像: {p}')
    return im

def save_image(p,im):
    ok,b=cv2.imencode(p.suffix,im)
    if not ok:raise ValueError(f'无法编码图像: {p}')
    p.write_bytes(b.tobytes())

def associate_for_evaluation(objects,annotation):
    """仅用于离线关联参考目标，不改变检测或测距。

    按同类别、检测框覆盖人工表面至少50%关联，优先覆盖率再置信度。不会用距离挑框。
    这不是全图检测mAP评估，因为图片里其他物体没有完整人工标注。
    """
    name={'box':'book','person':'person','chair':'chair'}[annotation['object']]
    x,y,w,h=annotation['surface_box'];candidates=[]
    for obj in objects:
        if obj['class_name']!=name:continue
        a,b,c,d=obj['bbox_xyxy'];area=max(0,min(c,x+w)-max(a,x))*max(0,min(d,y+h)-max(b,y))
        coverage=area/(w*h)
        if coverage>=.5:candidates.append((coverage,obj['confidence'],obj))
    return max(candidates,key=lambda x:(x[0],x[1])) if candidates else None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/local43')
    parser.add_argument('--pair',help='例如PartC/pair_0007')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(ROOT/'configs/pipeline.json');annotations=read_json(ROOT/'configs/local_targets.json')
    if args.pair:annotations=[a for a in annotations if a['part']+'/'+a['pair_id']==args.pair]
    if not annotations:raise ValueError('没有找到指定图片')
    save_json(out/'config_used.json',cfg);save_json(out/'targets_used.json',annotations)
    save_json(out/'provenance.json',dict(source=read_json(ROOT/'references/source.json'),
        camera_sha256=hashlib.sha256((ROOT/cfg['camera']).read_bytes()).hexdigest(),
        code_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'pipeline.py',ROOT/'run_offline.py',ROOT/'roi_policy.py',*sorted((ROOT/'detection').glob('*.py')),*sorted((ROOT/'stereo').glob('*.py'))]},
        opencv_version=cv2.__version__,numpy_version=np.__version__))
    # 此设置只限制CPU线程，不更改网络和匹配算法。
    cv2.setNumThreads(3)
    pipe=StereoPipeline(cfg,ROOT);comparisons=[];run_log=dict(version='v0.3',authorized='用户批准本版本YOLO本地测试与人工ROI对照',yolo_calls=0,sgbm_calls=0,status='running')
    for index,a in enumerate(annotations,1):
        pair_id=a['part']+'/'+a['pair_id'];folder=out/(a['part']+'_'+a['pair_id']);folder.mkdir()
        result,rect,arrays=pipe.process(read_image(a['left']),read_image(a['right']),pair_id)
        run_log['yolo_calls']+=1;run_log['sgbm_calls']+=1
        # 自动结果已经固定，现在才读取人工框及参考值进行比较。
        manual=measure_surface(arrays['depth_m'],a['surface_box'],cfg['sgbm']['block_size'])
        associated=associate_for_evaluation(result['objects'],a)
        chosen=associated[2] if associated else None
        auto=chosen['final'] if chosen else None
        z=auto['depth_m'] if auto and auto['available'] else None
        reference=a['reference_distance_m'];mz=manual['final']['depth_m']
        row=dict(pair_id=pair_id,object=a['object'],reference_distance_m=reference,
            target_detected=chosen is not None,association_coverage=associated[0] if associated else None,
            detection_id=chosen['detection_id'] if chosen else None,
            confidence=chosen['confidence'] if chosen else None,
            auto_available=z is not None,auto_depth_m=z,manual_depth_m=mz,
            auto_error_m=abs(z-reference) if z is not None else None,
            manual_error_m=abs(mz-reference) if mz is not None else None,
            status='target_not_detected' if chosen is None else chosen['status'],
            detect_ms=result['timing_ms']['detect_ms'],stereo_ms=result['timing_ms']['stereo_ms'])
        comparisons.append(row);save_json(folder/'automatic.json',result);save_json(folder/'manual_comparison.json',dict(manual=manual,evaluation=row))
        np.savez_compressed(folder/'stereo_arrays.npz',**arrays)
        save_image(folder/'left_rectified.png',rect[0]);save_image(folder/'right_rectified.png',rect[1])
        view=pipe.detector.annotate(rect[0],result['objects'])[0]
        for obj in result['objects']:
            if obj.get('surface_box'):
                x,y,w,h=obj['surface_box'];cv2.rectangle(view,(x,y),(x+w-1,y+h-1),(0,200,255),1)
                label=f"Z={obj['final']['depth_m']:.2f}m" if obj['final']['available'] else 'no reliable depth'
                cv2.putText(view,label,(x,min(view.shape[0]-5,y+18)),0,.45,(0,200,255),1)
        save_image(folder/'automatic_preview.jpg',view)
        x,y,w,h=a['surface_box'];cv2.rectangle(view,(x,y),(x+w-1,y+h-1),(255,0,255),1)
        save_image(folder/'comparison_preview.jpg',view)
        save_json(out/'run_log.json',run_log)
        print(f"{index}/{len(annotations)} {pair_id}: detected={chosen is not None} auto={z} manual={mz}",flush=True)
    save_json(out/'comparison.json',comparisons)
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    summary={}
    for obj in ('all','box','person','chair'):
        rows=[r for r in comparisons if obj=='all' or r['object']==obj];good=[r for r in rows if r['auto_available']]
        summary[obj]=dict(total=len(rows),detected=sum(r['target_detected'] for r in rows),auto_available=len(good),
            auto_mae_m=statistics.mean(r['auto_error_m'] for r in good) if good else None,
            manual_mae_same_pairs_m=statistics.mean(r['manual_error_m'] for r in good) if good else None,
            manual_mae_all_m=statistics.mean(r['manual_error_m'] for r in rows if r['manual_error_m'] is not None))
    save_json(out/'summary.json',summary);run_log['status']='complete';save_json(out/'run_log.json',run_log)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
