"""分析固定43对的检查前后差异；诊断结果单独保存，不改变测距输出。

分析分三部分：筛点前后误差、现有矩阵下的坐标方向差异、图案特征点的二维对应。
SIFT只用于核对图像证据，不接入测距流程，不用参考距离决定保留哪些匹配点。
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ranging import read_image, save_image


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def mutual_features(gray_left, gray_right, sift):
    """外观最近邻须明显优于次近邻，并要求左右互为最近邻，减少重复图案候选。"""
    kl, dl = sift.detectAndCompute(gray_left, None)
    kr, dr = sift.detectAndCompute(gray_right, None)
    if dl is None or dr is None or min(len(dl), len(dr)) < 2:
        return kl, kr, []
    matcher = cv2.BFMatcher()
    def good(a, b):
        return {pair[0].queryIdx:pair[0] for pair in matcher.knnMatch(a, b, k=2)
            if len(pair) == 2 and pair[0].distance < .7*pair[1].distance}
    forward, backward = good(dl, dr), good(dr, dl)
    matches = [m for i,m in forward.items()
        if m.trainIdx in backward and backward[m.trainIdx].trainIdx == i]
    # 同一特征可能以多个方向重复出现，按半像素坐标去重，避免重复计数。
    unique = {}
    for m in sorted(matches, key=lambda m:m.distance):
        key = tuple(round(v*2) for v in (*kl[m.queryIdx].pt, *kr[m.trainIdx].pt))
        unique.setdefault(key, m)
    return kl, kr, list(unique.values())


def compare_scene_changes(old_dir, sift):
    """比较0013～0015：背景是否整体移动，以及人物图案的尺度和位置是否变化。"""
    images = {n:[cv2.cvtColor(read_image(old_dir/f'PartC_pair_{n:04d}'/f'{side}_rectified.png'),
        cv2.COLOR_BGR2GRAY) for side in ('left','right')] for n in (13,14,15)}
    background, body = [], []
    for side in range(2):
        base = images[13][side]
        mask = np.full(base.shape,255,np.uint8)
        # 避开三张图中人物可能出现的范围，仅追踪静态桌椅和墙面图案。
        mask[:,190:440]=0
        mask[:20]=mask[-20:]=0
        pts = cv2.goodFeaturesToTrack(base,200,.02,8,mask=mask)
        for n in (14,15):
            target = images[n][side]
            q,st,error = cv2.calcOpticalFlowPyrLK(base,target,pts,None,winSize=(21,21),maxLevel=3)
            back,st_back,_ = cv2.calcOpticalFlowPyrLK(target,base,q,None,winSize=(21,21),maxLevel=3)
            good = (st[:,0]>0)&(st_back[:,0]>0)&(error[:,0]<15)
            good &= np.linalg.norm(back[:,0]-pts[:,0],axis=1)<.5
            shift = q[good,0]-pts[good,0]
            background.append(dict(side=('left','right')[side],from_pair=13,to_pair=n,
                points=int(good.sum()),median_xy=np.median(shift,axis=0).tolist(),
                median_abs_xy=np.median(np.abs(shift),axis=0).tolist()))
            # 同一侧相机跨组比较外套图案；估计表观尺度，仅用于判断是否真的靠近了一大段。
            kl,kr,matches = mutual_features(base,target,sift)
            pairs = [m for m in matches if 220<kl[m.queryIdx].pt[0]<410
                and 90<kl[m.queryIdx].pt[1]<295]
            if len(pairs)>=6:
                src=np.float32([kl[m.queryIdx].pt for m in pairs])
                dst=np.float32([kr[m.trainIdx].pt for m in pairs])
                matrix,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2)
                if matrix is not None:
                    center=np.array([310,190,1.0])
                    body.append(dict(side=('left','right')[side],from_pair=13,to_pair=n,
                        inliers=int(inliers.sum()),approximate_image_scale=float(np.hypot(matrix[0,0],matrix[0,1])),
                        shift_at_center_xy=(matrix@center-center[:2]).tolist()))
    return dict(background=background,body_appearance=body)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/consistency_evaluation/analysis')
    args=parser.parse_args()
    old_dir = ROOT/'results/dataset_evaluation/runs'
    new_dir = ROOT/'results/consistency_evaluation/runs'
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = read_json(old_dir/'per_pair.json')
    after = {(r['part'],r['pair_id']):r for r in read_json(new_dir/'per_pair.json')}
    with np.load(ROOT/'parameters/background1105.npz') as f:
        camera = dict(f)
    fb = float(-camera['P2'][0,3])
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.02)
    rows, all_features = [], []
    for old in before:
        name = f"{old['part']}_{old['pair_id']}"
        new = after[(old['part'],old['pair_id'])]
        result = read_json(new_dir/name/'result.json')
        cfg = read_json(new_dir/name/'config_resolved.json')
        left = read_image(old_dir/name/'left_rectified.png')
        right = read_image(old_dir/name/'right_rectified.png')
        gl, gr = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (left,right))
        x,y,w,h = cfg['roi_xywh']
        area = np.s_[y:y+h, x:x+w]
        with np.load(old_dir/name/'arrays.npz') as f:
            disp, mask = f['disparity_px'], f['valid_mask']
            # Q给出矫正坐标；R1的逆旋转将三维点换回原左相机坐标。
            # 数组中每个点写成行向量，因此这里右乘R1（等价于列向量左乘R1转置）。
            xyz = cv2.reprojectImageTo3D(disp, camera['Q'])
            z_original = np.full(disp.shape,np.nan,np.float64)
            # 仅旋转已有有限深度的点，不让零视差产生的无穷大参与矩阵乘法。
            z_original[mask] = (xyz[mask] @ camera['R1'])[:,2]
            base_depth = f['depth_m'][area][mask[area]]
            original_z = float(np.median(z_original[area][mask[area]]))
            base_disp = disp[area][mask[area]]
        with np.load(new_dir/name/'arrays.npz') as f:
            new_z = f['depth_m'][area][f['valid_mask'][area]]
        # 纹理强弱只作描述：Sobel/8将3×3梯度换到大致的灰度每像素变化尺度。
        gx = cv2.Sobel(gl,cv2.CV_32F,1,0,ksize=3)/8
        gy = cv2.Sobel(gl,cv2.CV_32F,0,1,ksize=3)/8
        row = {k:old[k] for k in ('part','pair_id','object','nominal_distance_m','reference_distance_m')}
        row.update(before_depth_m=old['depth_m'], after_depth_m=new['depth_m'],
            after_available=new['available'], before_error_m=old['absolute_error_m'],
            after_error_m=new['absolute_error_m'], before_valid_fraction=old['valid_fraction'],
            after_valid_fraction=new['valid_fraction'], **result['consistency_check'],
            original_left_axis_z_m=original_z,
            original_axis_absolute_error_m=abs(original_z-old['reference_distance_m']),
            coordinate_change_m=original_z-old['depth_m'],
            disparity_median_px=float(np.median(base_disp)),
            # 假设现场参考值就是原左相机Z时，计算所需视差与观测视差的差；不用于补偿。
            disparity_excess_after_axis_change_px=float(np.median(base_disp*(1-z_original[area][mask[area]]/old['reference_distance_m']))),
            texture_gradient_mean=float(np.mean(np.hypot(gx[area],gy[area]))),
            roi_gray_std=float(gl[area].std()),
            before_depth_over10_fraction=float(np.mean(base_depth>10)),
            after_depth_over10_fraction=float(np.mean(new_z>10)) if len(new_z) else None,
            right_matching_ms=result['timing_ms']['right_matching_ms'],
            check_ms=result['timing_ms']['consistency_ms'])
        # 衣服中央可能缺少图案，诊断时还查看周围外套，测距ROI始终保持原样。
        feature_roi = [x,y,w,h]
        if old['object']=='person':
            feature_roi = {1:[200,15,380,210],2:[230,80,230,240],3:[240,90,165,205]}[int(old['nominal_distance_m'])]
        fx,fy,fw,fh = feature_roi
        kl,kr,matches = mutual_features(gl,gr,sift)
        chosen=[]
        for m in matches:
            pl,pr = kl[m.queryIdx].pt,kr[m.trainIdx].pt
            dx,dy = pl[0]-pr[0],pl[1]-pr[1]
            if fx<=pl[0]<fx+fw and fy<=pl[1]<fy+fh and 0<dx<128 and abs(dy)<=12:
                chosen.append(m)
        features=[dict(left_xy=list(kl[m.queryIdx].pt),right_xy=list(kr[m.trainIdx].pt),
            disparity_px=kl[m.queryIdx].pt[0]-kr[m.trainIdx].pt[0],
            vertical_residual_px=kl[m.queryIdx].pt[1]-kr[m.trainIdx].pt[1]) for m in chosen]
        dx=np.array([f['disparity_px'] for f in features]);dy=np.array([f['vertical_residual_px'] for f in features])
        row.update(feature_roi_xywh=json.dumps(feature_roi),feature_count=len(features),
            feature_disparity_median_px=float(np.median(dx)) if len(dx) else None,
            feature_vertical_median_px=float(np.median(dy)) if len(dy) else None,
            feature_abs_vertical_median_px=float(np.median(np.abs(dy))) if len(dy) else None)
        differences,feature_z = [],[]
        for point in features:
            u,v = point['left_xy']
            ix,iy = round(u),round(v)
            d = point['disparity_px']
            # 用图案候选自身的横坐标差计算诊断深度，再换回原左相机方向。
            z = fb/d
            normalized = np.array([(u-camera['P1'][0,2])/camera['P1'][0,0],
                (v-camera['P1'][1,2])/camera['P1'][1,1],1.0])
            feature_z.append(float(z*(normalized@camera['R1'][:,2])))
            if mask[iy,ix]:
                differences.append(float(disp[iy,ix])-d)
        row.update(feature_comparable_count=len(differences),
            sgbm_minus_feature_disparity_px=float(np.median(differences)) if differences else None,
            feature_original_axis_z_m=float(np.median(feature_z)) if feature_z else None)
        all_features.append(dict(pair=name,feature_roi_xywh=feature_roi,matches=features))
        rows.append(row)
        if old['part']=='PartC' and old['pair_id'] in ('pair_0013','pair_0014','pair_0015'):
            # 每个小格最多画一个候选，避免大量连线遮住衣服图案。
            spaced={}
            for m in sorted(chosen,key=lambda m:m.distance):
                u,v=kl[m.queryIdx].pt
                spaced.setdefault((int(u)//20,int(v)//25),m)
            view=cv2.drawMatches(left,kl,right,kr,list(spaced.values())[:20],None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            save_image(output/f'{name}_feature_matches.jpg',view)
        print(f'{name}: features={len(features)}',flush=True)
    save_json(output/'per_pair_diagnostics.json',rows)
    save_json(output/'feature_correspondences.json',all_features)
    with (output/'per_pair_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # 在同样能输出的41对上比较，另列拒绝数量，避免因为删掉难样本而显得精度提高。
    common=[r for r in rows if r['after_available']]
    aggregate=dict(total=len(rows),after_available=len(common),
        before_mae_on_common_m=statistics.mean(r['before_error_m'] for r in common),
        after_mae_on_common_m=statistics.mean(r['after_error_m'] for r in common),
        mean_extra_right_matching_ms=statistics.mean(r['right_matching_ms'] for r in rows),
        mean_consistency_check_ms=statistics.mean(r['check_ms'] for r in rows),
        coordinate_diagnostic={obj:dict(
            before_mae_m=statistics.mean(r['before_error_m'] for r in rows if r['object']==obj),
            original_axis_mae_m=statistics.mean(r['original_axis_absolute_error_m'] for r in rows if r['object']==obj))
            for obj in ('box','person','chair')})
    save_json(output/'summary.json',aggregate)
    save_json(output/'scene_changes_0013_to_0015.json',compare_scene_changes(old_dir,sift))
    print(json.dumps(aggregate,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
