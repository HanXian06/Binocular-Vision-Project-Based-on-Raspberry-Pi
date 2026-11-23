"""物体表面的区域测距：汇总人工 ROI 内的有效深度。

ROI 写成 [x, y, width, height]，对应左侧校正图中的一个矩形。
测量值是这个表面区域的 Z 深度中位数，单位米。"""
import numpy as np


def measure_region(depth, valid, roi, min_pixels, min_fraction):
    """从选定物体表面的有效深度中计算中位数、有效比例和四分位距。"""
    if len(roi) != 4 or any(int(v) != v for v in roi):
        raise ValueError('ROI must contain integer x, y, width, height')
    # 将 ROI 的左上角和宽高分别取出。
    x, y, w, h = map(int, roi)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > depth.shape[1] or y + h > depth.shape[0]:
        raise ValueError('ROI exceeds the rectified image bounds')
    if min_pixels < 1 or not 0 <= min_fraction <= 1:
        raise ValueError('Invalid region validity thresholds')
    # NumPy 切片先行后列，即先 y 后 x；右端不包含，所以正好取 h×w 个像素。
    z = depth[y:y+h, x:x+w]
    # 在目标表面 ROI 内，合并有效标记、有限数和正深度条件。
    mask = valid[y:y+h, x:x+w] & np.isfinite(z) & (z > 0)
    # 布尔索引把 ROI 中通过条件的深度取出，形成一维数组。
    values = z[mask]
    # 有效比例以整个 ROI 的像素数为分母，体现该区域有多少像素产生了结果。
    fraction = len(values) / (w * h)
    # 按配置判断像素数量和比例是否足够；不够时记录相应原因。
    reasons = []
    if len(values) < min_pixels:
        reasons.append('too_few_valid_pixels')
    if fraction < min_fraction:
        reasons.append('valid_fraction_below_threshold')
    # 原因列表为空，表示达到本版的输出门槛。
    available = not reasons
    # Q25/Q75 是深度从小到大排列后25%/75%的位置；两者之差为IQR。
    q25, q75 = np.percentile(values, (25, 75)) if len(values) else (None, None)
    # 区域深度取中位数；IQR描述中间一半结果的分散程度。未达到门槛时距离填None。
    return {'available': available, 'depth_m': float(np.median(values)) if available else None,
        'roi_xywh': [x, y, w, h], 'roi_pixels': w*h, 'valid_pixels': int(len(values)),
        'valid_fraction': fraction, 'q25_m': float(q25) if q25 is not None else None,
        'q75_m': float(q75) if q75 is not None else None,
        'iqr_m': float(q75-q25) if len(values) else None, 'reasons': reasons}
