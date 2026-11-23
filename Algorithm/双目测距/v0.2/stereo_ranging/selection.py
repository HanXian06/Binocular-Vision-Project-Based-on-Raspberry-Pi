"""主动选点：在已指定的物体表面ROI内寻找可定位的Shi–Tomasi角点。

角点不是物体识别结果，只说明附近图案在两个方向上都有变化。
分区名额防止所有点挤在一小块文字上；本模块不使用深度或参考距离。
"""
import math
import cv2
import numpy as np


def select_points(gray,roi,budget,cfg,support=None):
    """返回N×2整数数组，每一行是校正左图的(x,y)，N不超过budget。"""
    x,y,w,h=map(int,roi)
    if gray.ndim!=2 or min(x,y)<0 or min(w,h)<=0 or x+w>gray.shape[1] or y+h>gray.shape[0]:
        raise ValueError('ROI must be inside a grayscale image')
    if budget<0 or not 0<cfg['quality_level']<=1 or cfg['min_distance']<0 or cfg['cells']<1:
        raise ValueError('Invalid point selection settings')
    if budget==0:return np.empty((0,2),int)
    mask=np.zeros(gray.shape,np.uint8)
    mask[y:y+h,x:x+w]=255
    if support is not None:
        if support.shape!=gray.shape:raise ValueError('Support size differs from image')
        mask[~support.astype(bool)]=0
    # maxCorners=0先取得满足质量和间隔条件的所有候选，再按分区决定本轮保留哪些。
    candidates=cv2.goodFeaturesToTrack(gray,maxCorners=0,qualityLevel=cfg['quality_level'],
        minDistance=cfg['min_distance'],mask=mask,blockSize=cfg['block_size'],useHarrisDetector=False)
    if candidates is None:return np.empty((0,2),int)
    cells=int(cfg['cells']);quota=math.ceil(budget/(cells*cells))
    counts={};chosen=[]
    # OpenCV按角点响应由强到弱排序；每格最多quota个，总数仍不超过原网格预算。
    for u,v in np.rint(candidates[:,0,:]).astype(int):
        key=(min(cells-1,(u-x)*cells//w),min(cells-1,(v-y)*cells//h))
        if counts.get(key,0)>=quota:continue
        chosen.append((int(u),int(v)));counts[key]=counts.get(key,0)+1
        if len(chosen)>=budget:break
    return np.array(sorted(chosen,key=lambda p:(p[1],p[0])),dtype=int).reshape(-1,2)
