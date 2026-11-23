"""独立的左右对应检查：左图找到右点后，再看右图能否返回原左点。

左视差 dL=x左−x右，右视差 dR=x右−x左，因此正确往返应满足 dL+dR≈0。
这里仅排除无法确认或往返偏差较大的点，不重新估计它们的深度。
"""
import cv2
import numpy as np

from .matching import compute_disparity, support_mask


def compute_right_disparity(left_gray, right_gray, supports, cfg):
    """独立运行右→左SGBM，返回16倍整数视差、像素视差和右图有效掩码。"""
    reverse = dict(cfg)
    # 左范围为[m,m+n−1]；右范围应为[−m−n+1,−m]，当前即[−127,0]。
    # 复制配置后修改，不改变左图使用的参数。其他匹配参数与左图相同。
    reverse['min_disparity'] = -cfg['min_disparity']-cfg['num_disparities']+1
    raw, disparity = compute_disparity(right_gray, left_gray, reverse)
    valid = raw > (reverse['min_disparity']-1)*16
    # 正深度的当前双目布局对应负的右视差。
    valid &= np.isfinite(disparity) & (disparity < 0)
    # 此时“起点图”为右图；x−dR是返回左图的位置。
    valid &= support_mask(disparity, supports[1], supports[0], cfg['block_size'])
    h, w = disparity.shape
    x, y, rw, rh = cv2.getValidDisparityROI((0, 0, w, h), (0, 0, w, h),
        reverse['min_disparity'], reverse['num_disparities'], reverse['block_size'])
    bounds = np.zeros((h, w), bool)
    bounds[y:y+rh, x:x+rw] = True
    return raw, disparity, valid & bounds


def left_right_consistency(left_disparity, right_disparity, left_valid, right_valid, max_diff_px=1.0):
    """返回可检查位置、通过位置、往返误差及取到的右视差，单位均为像素。

    checked=False 表示没有可用于往返比较的有效右点；residual_px在这些位置为NaN。
    consistent=True 才表示该左点通过检查，能交给后续区域统计。
    """
    dl, dr = np.asarray(left_disparity), np.asarray(right_disparity)
    lv, rv = np.asarray(left_valid, bool), np.asarray(right_valid, bool)
    if dl.ndim != 2 or any(a.shape != dl.shape for a in (dr, lv, rv)):
        raise ValueError('Disparity maps and masks must be equally sized 2D arrays')
    if not np.isfinite(max_diff_px) or max_diff_px < 0:
        raise ValueError('max_diff_px must be a finite nonnegative number')
    h, w = dl.shape
    yy, xx = np.indices((h, w))
    # 小数右坐标可能落在两列之间，需要线性插值；只查询图像范围内的有限坐标。
    xr = xx-dl
    in_bounds = lv & np.isfinite(dl) & (dl > 0) & np.isfinite(xr) & (xr >= 0) & (xr <= w-1)
    safe_x = np.where(in_bounds, xr, 0)
    lower = np.floor(safe_x).astype(np.intp)
    upper = np.ceil(safe_x).astype(np.intp)
    a, b = dr[yy, lower], dr[yy, upper]
    # 恰好在整数列时lower=upper，不会错误要求一个没有参与插值的相邻像素也有效。
    checked = in_bounds & rv[yy, lower] & rv[yy, upper]
    checked &= np.isfinite(a) & np.isfinite(b) & (a < 0) & (b < 0)
    sampled = np.full(dl.shape, np.nan, np.float32)
    t = (safe_x-lower)[checked]
    sampled[checked] = (1-t)*a[checked]+t*b[checked]
    residual = np.full(dl.shape, np.nan, np.float32)
    # x右−dR 与原x左的差，等价于abs(dL+dR)。
    residual[checked] = np.abs(dl[checked]+sampled[checked])
    consistent = checked & (residual <= max_diff_px)
    return dict(checked=checked, consistent=consistent, residual_px=residual,
                sampled_right_px=sampled)
