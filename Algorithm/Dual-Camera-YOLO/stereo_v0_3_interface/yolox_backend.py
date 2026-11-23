"""原仓库YOLOX-Nano的接口适配副本，不改变模型、预处理和输出解码算法。

infer返回原图坐标的结构化结果，annotate仅在副本上绘制；测距始终使用未绘制图。
"""
import json
from pathlib import Path
import cv2
import numpy as np

CLASSES=json.loads(Path(__file__).with_name('classes.json').read_text(encoding='utf-8'))

def preprocess(frame,size=416):
    """等比缩放到416方形左上角，右侧/下侧补114，保留原BGR顺序。"""
    h,w=frame.shape[:2];scale=size/max(h,w)
    nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
    canvas=np.full((size,size,3),114,np.uint8)
    canvas[:nh,:nw]=cv2.resize(frame,(nw,nh))
    return canvas,scale

def decode(pred,shape,threshold=.35,nms_iou=.45,size=416):
    """按原仓库YOLOX公式解码，再按类别NMS；返回同一输入图坐标的xyxy。"""
    h,w=shape[:2];scale=size/max(h,w);grids=[];strides=[]
    for stride in (8,16,32):
        side=size//stride;x,y=np.meshgrid(np.arange(side),np.arange(side))
        grids.append(np.stack((x,y),axis=-1).reshape(-1,2))
        strides.append(np.full((side*side,1),stride))
    grid=np.concatenate(grids);stride=np.concatenate(strides)
    pred=np.asarray(pred).reshape(-1,85)
    if len(pred)!=len(grid):raise ValueError(f'YOLOX输出数量不符: {pred.shape}')
    # 目标存在分数乘类别分数，保持原仓库的置信度定义。
    scores=pred[:,4:5]*pred[:,5:];labels=scores.argmax(axis=1);confidence=scores.max(axis=1)
    selected=confidence>=threshold
    centers=(pred[selected,:2]+grid[selected])*stride[selected]
    sizes=np.exp(pred[selected,2:4])*stride[selected]
    low=np.maximum((centers-sizes/2)/scale,(0,0))
    high=np.minimum((centers+sizes/2)/scale,(w-1,h-1))
    boxes=np.concatenate((low,high-low),axis=1);valid=np.all(boxes[:,2:]>1,axis=1)
    boxes=boxes[valid];labels=labels[selected][valid];probs=confidence[selected][valid]
    detections=[]
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label)
        keep=cv2.dnn.NMSBoxes(boxes[ids].tolist(),probs[ids].tolist(),threshold,nms_iou)
        for j in np.asarray(keep,dtype=int).ravel():
            i=ids[j];x,y,bw,bh=boxes[i]
            detections.append(dict(detection_id=len(detections),class_id=int(label),class_name=CLASSES[label],
                confidence=float(probs[i]),bbox_xyxy=[float(x),float(y),float(x+bw),float(y+bh)]))
    return detections

class Detector:
    def __init__(self,path,threshold=.35,nms_iou=.45):
        # Windows下OpenCV直接读取中文模型路径可能失败；内存字节接口读取相同权重。
        self.net=cv2.dnn.readNetFromONNX(np.frombuffer(Path(path).read_bytes(),np.uint8))
        self.threshold=threshold;self.nms_iou=nms_iou
    def infer(self,frame):
        canvas,_=preprocess(frame)
        self.net.setInput(cv2.dnn.blobFromImage(canvas))
        return decode(self.net.forward(),frame.shape,self.threshold,self.nms_iou)
    def annotate(self,frame,detections=None):
        """兼容原显示用途；已给detections时不重复推理。"""
        detections=self.infer(frame) if detections is None else detections
        out=frame.copy()
        for d in detections:
            x1,y1,x2,y2=map(int,d['bbox_xyxy'])
            cv2.rectangle(out,(x1,y1),(x2,y2),(50,235,100),2)
            cv2.putText(out,f"{d['class_name']} {d['confidence']:.0%}",(x1,max(16,y1-5)),0,.55,(50,235,100),1)
        return out,[d['class_name'] for d in detections]
