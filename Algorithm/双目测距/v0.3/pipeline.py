"""D2+M1+G1：检测框自动表面适配、固定13区域、原SGBM及逐目标深度。

运行不读取人工框。人工框只由离线评估入口另行传入measure_surface作对照。
"""
from time import perf_counter
import cv2
import numpy as np
from detection.yolox_backend import Detector
from stereo.geometry import rectify,depth_from_disparity
from stereo.matching import compute_disparity,valid_geometry
from roi_policy import regions,stats

def surface_box(detection,fractions,size):
    """按类别将物体框转换成表面框；浮点框先裁到图像，表面外边界向外取整。"""
    name=detection['class_name']
    if name not in fractions:return None
    x1,y1,x2,y2=detection['bbox_xyxy'];a,b,c,d=fractions[name]
    w,h=x2-x1,y2-y1
    left=max(0,int(np.floor(x1+a*w)));top=max(0,int(np.floor(y1+b*h)))
    right=min(size[0],int(np.ceil(x1+c*w)));bottom=min(size[1],int(np.ceil(y1+d*h)))
    return [left,top,right-left,bottom-top] if right>left and bottom>top else None

def measure_surface(depth,box,block_size):
    """全部候选去重后的主结果等于完整区；子区只显示空间敏感性，不挑最佳。"""
    if box is None:return dict(status='unsupported_class',final=None,regions={})
    rois=regions(box,size=(depth.shape[1],depth.shape[0]));mask=np.zeros(depth.shape,np.uint8)
    x,y,w,h=rois['full'];mask[y:y+h,x:x+w]=1
    safe=cv2.erode(mask,np.ones((block_size,block_size),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    values=depth[safe];final=stats(values,np.ones(len(values),bool))
    regional={}
    for label,(x,y,w,h) in rois.items():
        candidate=np.zeros(depth.shape,bool);candidate[y:y+h,x:x+w]=True
        regional[label]=stats(depth,safe&candidate)
    return dict(surface_box=box,status='distance_available' if final['available'] else 'insufficient_depth',
        final=final,regions=regional)

class StereoPipeline:
    def __init__(self,cfg,base):
        self.cfg=cfg
        with np.load(base/cfg['camera']) as a:self.camera=dict(a)
        self.detector=Detector(base/cfg['model'],cfg['detector']['confidence'],cfg['detector']['nms_iou'])
    def process(self,left,right,pair_id,left_timestamp_ns=None,right_timestamp_ns=None):
        """输入同一帧组的原图；先检测未绘制的校正左图，独立计算全图深度。"""
        timing={};t=perf_counter();rect,support=rectify(left,right,self.camera)
        timing['rectify_ms']=(perf_counter()-t)*1000
        t=perf_counter();detections=self.detector.infer(rect[0]);timing['detect_ms']=(perf_counter()-t)*1000
        t=perf_counter();gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in rect]
        raw,disp=compute_disparity(*gray,self.cfg['sgbm']);depth=depth_from_disparity(disp,self.camera['Q'])
        valid=valid_geometry(disp,raw,depth,support,self.cfg['sgbm']);depth[~valid]=np.nan
        timing['stereo_ms']=(perf_counter()-t)*1000
        t=perf_counter();objects=[]
        for detection in detections:
            box=surface_box(detection,self.cfg['surface_fractions'],(left.shape[1],left.shape[0]))
            objects.append({**detection,**measure_surface(depth,box,self.cfg['sgbm']['block_size'])})
        timing['regions_ms']=(perf_counter()-t)*1000
        delta=abs(left_timestamp_ns-right_timestamp_ns) if left_timestamp_ns is not None and right_timestamp_ns is not None else None
        result=dict(version='v0.3',pair_id=pair_id,left_timestamp_ns=left_timestamp_ns,right_timestamp_ns=right_timestamp_ns,
            pair_time_difference_ns=delta,distance_definition='rectified_left_optical_axis_Z',objects=objects,timing_ms=timing)
        return result,rect,dict(disparity_px=disp,depth_m=depth,valid_mask=valid)
