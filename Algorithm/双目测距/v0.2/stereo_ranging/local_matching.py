"""离线局部NCC匹配实验：保存二维对应，不改变主程序的SGBM。

NCC先减去图块平均亮度再归一化，比较的是图案的相对明暗变化。
相似度接近1表示外观接近；它并不直接给出物体身份或深度准确率。
"""
import cv2
import numpy as np


def match_points(left,right,points,vertical_radius=8,patch_size=9,max_disparity=127,
                 min_std=5.0,min_score=.75,min_margin=.03,supports=None):
    """在每个左点附近取图块，在右图有限横向/纵向范围搜索，返回逐点记录。

    points中坐标按(x,y)排列。right_xy保留最佳候选，即使未通过门槛也可用于诊断。
    正视差要求右点在左点左方；纵向偏移定义为y右−y左。
    """
    if left.shape!=right.shape or left.ndim!=2:
        raise ValueError('Expected equally sized grayscale images')
    if patch_size<3 or patch_size%2==0 or vertical_radius<0 or max_disparity<1:
        raise ValueError('Invalid patch or search range')
    h,w=left.shape;r=patch_size//2
    if supports is None:supports=(np.ones(left.shape,bool),np.ones(right.shape,bool))
    kernel=np.ones((patch_size,patch_size),np.uint8)
    safe=[cv2.erode(s.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0 for s in supports]
    rows=[]
    for x,y in points:
        x,y=int(x),int(y)
        row=dict(left_xy=[x,y],right_xy=None,score=None,second_score=None,margin=None,
            texture_std=None,accepted=False,reason='',vertical_at_limit=False,horizontal_at_limit=False)
        rows.append(row)
        if not (r<=x<w-r and r<=y<h-r) or not safe[0][y,x]:
            row['reason']='unsupported_left';continue
        template=left[y-r:y+r+1,x-r:x+r+1]
        row['texture_std']=float(template.std())
        if row['texture_std']<min_std:
            row['reason']='weak_texture';continue
        # 以下边界描述“右图候选中心”，外围再各加r个像素取得完整图块。
        x0,x1=max(r,x-max_disparity),min(w-r-1,x-1)
        y0,y1=max(r,y-vertical_radius),min(h-r-1,y+vertical_radius)
        if x0>x1 or y0>y1:row['reason']='empty_search';continue
        search=right[y0-r:y1+r+1,x0-r:x1+r+1]
        scores=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
        supported=safe[1][y0:y1+1,x0:x1+1]
        scores=np.where(supported & np.isfinite(scores),scores,-2.0)
        py,px=np.unravel_index(np.argmax(scores),scores.shape)
        peak=float(scores[py,px])
        if peak<=-2:row['reason']='unsupported_right';continue
        # 同一个峰附近的小位移属于同一候选；次佳值从峰周围±2像素以外寻找。
        alternatives=scores.copy()
        alternatives[max(0,py-2):py+3,max(0,px-2):px+3]=-2
        second=float(alternatives.max())
        offset=0.0
        # 对横向峰值拟合抛物线，估计不足一个像素的小数位移。
        if 0<px<scores.shape[1]-1 and np.all(scores[py,px-1:px+2]>-2):
            a,b,c=map(float,scores[py,px-1:px+2]);den=a-2*b+c
            if den < -1e-6:offset=float(np.clip(.5*(a-c)/den,-.5,.5))
        row.update(right_xy=[float(x0+px+offset),float(y0+py)],score=peak,
            second_score=second if second>-2 else None,margin=peak-second if second>-2 else None,
            vertical_at_limit=vertical_radius>0 and abs(y0+py-y)==vertical_radius,
            horizontal_at_limit=px in (0,scores.shape[1]-1))
        if peak<min_score:row['reason']='low_similarity'
        elif second>-2 and peak-second<min_margin:row['reason']='ambiguous'
        elif row['horizontal_at_limit']:row['reason']='horizontal_search_limit'
        else:row.update(accepted=True,reason='accepted')
    return rows


def triangulate_matches(left_xy,right_xy,p1,p2):
    """用完整二维坐标三角化，返回矫正坐标XYZ和两侧重投影的均方根像素残差。

    非零行差意味着两条观测射线通常不相交；三角化给出折中解，残差把这种不相容记录下来。
    """
    left_xy=np.asarray(left_xy,np.float64).reshape(-1,2)
    right_xy=np.asarray(right_xy,np.float64).reshape(-1,2)
    if left_xy.shape!=right_xy.shape:raise ValueError('Point arrays must have the same shape')
    if not len(left_xy):return np.empty((0,3)),np.empty(0)
    homogeneous=cv2.triangulatePoints(p1,p2,left_xy.T,right_xy.T).T
    xyz=np.full((len(left_xy),3),np.nan)
    good=np.abs(homogeneous[:,3])>1e-12
    xyz[good]=homogeneous[good,:3]/homogeneous[good,3,None]
    points=np.column_stack([xyz,np.ones(len(xyz))])
    errors=[]
    for projection,observed in ((p1,left_xy),(p2,right_xy)):
        q=points@projection.T
        uv=np.full((len(xyz),2),np.nan)
        visible=np.isfinite(q).all(axis=1)&(np.abs(q[:,2])>1e-12)
        uv[visible]=q[visible,:2]/q[visible,2,None]
        errors.append(np.sum((uv-observed)**2,axis=1))
    return xyz,np.sqrt((errors[0]+errors[1])/2)
