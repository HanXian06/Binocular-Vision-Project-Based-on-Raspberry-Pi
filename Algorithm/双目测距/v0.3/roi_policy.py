"""固定的本地/检测框选区协议。函数不读取距离、视差或图片编号。

输入是校正左图上的目标表面框；本地使用冻结标注，未来由YOLO类别适配器产生。
一次产生完整区、中心区、左右边带和九宫格子区。所有区域都保存，不按深度挑框。
"""
import numpy as np

POLICY=dict(id='surface_regions_1',inset=.05,center=[.2,.2,.8,.8],side_width=.2,
    tiles=3,final='所有候选位置去重后取有效深度中位数',min_count=30,min_fraction=.1)

def regions(box,size=(640,480)):
    """坐标为左闭右开的[x,y,w,h]；裁剪后区域均在画面内。"""
    x,y,w,h=box
    x0,y0=max(0,int(x)),max(0,int(y))
    x1,y1=min(size[0],int(x+w)),min(size[1],int(y+h))
    if x1<=x0 or y1<=y0:raise ValueError('Empty surface box')
    def part(a,b,c,d):
        left=x0+int((x1-x0)*a);top=y0+int((y1-y0)*b)
        right=x0+int((x1-x0)*c);bottom=y0+int((y1-y0)*d)
        return [left,top,max(1,right-left),max(1,bottom-top)]
    r={'full':part(.05,.05,.95,.95),'center':part(.2,.2,.8,.8),
       'left':part(.05,.05,.25,.95),'right':part(.75,.05,.95,.95)}
    # 九宫格覆盖相同目标表面，可显示某个局部的异常，不借助参考值选择。
    for j in range(3):
        for i in range(3):r[f'tile_{j}_{i}']=part(.05+.9*i/3,.05+.9*j/3,.05+.9*(i+1)/3,.05+.9*(j+1)/3)
    return r

def contains(points,roi):
    """判断点中心属于哪个候选区；匹配窗口支持仍交给原匹配器检查。"""
    x,y,w,h=roi;p=np.asarray(points).reshape(-1,2)
    return (p[:,0]>=x)&(p[:,0]<x+w)&(p[:,1]>=y)&(p[:,1]<y+h)

def stats(z,mask):
    """合并位置后仍用原30点/10%规则。无输出不写成0米。"""
    z=np.asarray(z);mask=np.asarray(mask,bool);valid=mask&np.isfinite(z)&(z>0)
    total=int(mask.sum());count=int(valid.sum());fraction=count/total if total else 0
    median=float(np.median(z[valid])) if count else None
    return dict(available=count>=30 and fraction>=.1,depth_m=median if count>=30 and fraction>=.1 else None,
        diagnostic_median_m=median,valid_count=count,candidate_count=total,valid_fraction=fraction)
