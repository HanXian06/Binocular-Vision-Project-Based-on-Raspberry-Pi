"""原图与校正图的检测对照，用于定位漏检；不修改正式自动测距结果。"""
import argparse
import cv2
import numpy as np
from run_offline import ROOT,read_json,save_json,read_image
from detection.yolox_backend import Detector

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=__import__('pathlib').Path,default=ROOT/'results/detection_input_control')
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
    cfg=read_json(ROOT/'configs/pipeline.json');targets=read_json(ROOT/'configs/local_targets.json')
    with np.load(ROOT/cfg['camera']) as f:c=dict(f)
    # 查找表将“校正图坐标”映射回“原图采样坐标”，人工表面只用于评估目标关联。
    mx,my=cv2.initUndistortRectifyMap(c['K1'],c['D1'],c['R1'],c['P1'],(640,480),cv2.CV_32FC1)
    detector=Detector(ROOT/cfg['model'],cfg['detector']['confidence'],cfg['detector']['nms_iou'])
    cv2.setNumThreads(3);rows=[]
    for a in targets:
        detections=detector.infer(read_image(a['left']))
        x,y,w,h=a['surface_box'];xx,yy=np.meshgrid(np.linspace(x,x+w-1,9).astype(int),np.linspace(y,y+h-1,9).astype(int))
        px,py=mx[yy,xx],my[yy,xx]
        valid=(px>=0)&(px<640)&(py>=0)&(py<480)
        name={'box':'book','person':'person','chair':'chair'}[a['object']]
        candidates=[]
        for d in detections:
            if d['class_name']!=name:continue
            l,t,r,b=d['bbox_xyxy']
            coverage=float(np.mean(((px>=l)&(px<=r)&(py>=t)&(py<=b))[valid])) if valid.any() else 0
            if coverage>=.5:candidates.append(d)
        rows.append(dict(pair_id=a['part']+'/'+a['pair_id'],object=a['object'],raw_target_detected=bool(candidates),detections=detections))
    save_json(out/'raw_detections.json',rows)
    summary={obj:dict(total=sum(a['object']==obj for a in rows),detected=sum(a['object']==obj and a['raw_target_detected'] for a in rows)) for obj in ('box','person','chair')}
    save_json(out/'summary.json',summary);print(summary)

if __name__=='__main__':main()
