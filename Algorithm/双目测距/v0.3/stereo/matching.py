"""视差匹配：用 StereoSGBM 寻找左右对应位置，并判断哪些像素可以参与测距。

视差定义为左图横坐标减右图横坐标，即 d = x_left - x_right。
本模块的布尔掩码与图像同样大小：True 表示保留该像素，False 表示排除。"""
import cv2
import numpy as np


def compute_disparity(left_gray, right_gray, cfg):
    """输入左右灰度图及配置，同时返回原始整数视差和像素单位视差。"""
    if left_gray.shape != right_gray.shape or left_gray.ndim != 2:
        raise ValueError('Matching requires equally sized grayscale images')
    # n 是搜索范围宽度，b 是用于比较局部外观的方形窗口边长。
    n, b = cfg['num_disparities'], cfg['block_size']
    if n <= 0 or n % 16 or b < 1 or b % 2 == 0:
        raise ValueError('num_disparities must be a positive multiple of 16; block_size odd')
    if not 0 < cfg['p1'] < cfg['p2']:
        raise ValueError('Require 0 < p1 < p2')
    # 创建匹配器。P1/P2 惩罚视差的不连续，uniquenessRatio 要求最佳候选有足够优势。
    matcher = cv2.StereoSGBM_create(minDisparity=cfg['min_disparity'],
        numDisparities=n, blockSize=b, P1=cfg['p1'], P2=cfg['p2'],
        uniquenessRatio=cfg['uniqueness_ratio'], disp12MaxDiff=cfg['disp12_max_diff'],
        speckleWindowSize=cfg['speckle_window_size'], speckleRange=cfg['speckle_range'],
        mode=cv2.STEREO_SGBM_MODE_SGBM)
    # 每个左图像素获得一个视差；OpenCV 用16倍整数保存亚像素结果。
    raw = matcher.compute(left_gray, right_gray)
    # 先转浮点再除16，保留小数。例如 raw=128 对应8像素视差。
    return raw, raw.astype(np.float32) / 16.0


def support_mask(disparity, left_support, right_support, block_size):
    """检查左右匹配窗口是否都来自真实图像，返回可保留的像素位置。"""
    # 匹配要使用整个局部窗口，所以只看中心像素是否有效还不够。
    kernel = np.ones((block_size, block_size), np.uint8)
    # 侵蚀会收缩有效区域：窗口内全是有效像素时，中心才被保留。
    left = cv2.erode(left_support.astype(np.uint8), kernel,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    # 右图也做相同处理；转成浮点后可以检查小数坐标的插值支持。
    right = cv2.erode(right_support.astype(np.uint8), kernel,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    # 生成与视差图同尺寸的坐标表：xx 记录列号 x，yy 记录行号 y。
    yy, xx = np.indices(disparity.shape, dtype=np.float32)
    # 右侧对应位置为（x-d,y）；查的是这个位置的支持，而不是右图相同 x 的支持。
    right_at_match = cv2.remap(right, xx - disparity, yy, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
    # & 是逐像素“同时满足”；返回左右都支持、且视差是有限数的位置。
    return left & (right_at_match >= .999) & np.isfinite(disparity)


def valid_geometry(disparity, raw, depth, supports, cfg):
    """合并视差、深度、图像支持和匹配边界的有效条件。"""
    # SGBM 用（最小视差-1）×16标记无效匹配，先排除这些值。
    valid = raw > (cfg['min_disparity'] - 1) * 16
    # 本版左右顺序和Q对应正视差、正深度；isfinite 排除 NaN 和无穷大。
    valid &= (disparity > 0) & np.isfinite(depth) & (depth > 0)
    # 继续合并真实图像支持条件；&= 表示在已有条件上追加要求。
    valid &= support_mask(disparity, *supports, cfg['block_size'])
    h, w = disparity.shape
    # 搜索范围和窗口会占用图像边缘，取得能完整进行匹配的矩形范围。
    x, y, rw, rh = cv2.getValidDisparityROI((0, 0, w, h), (0, 0, w, h),
        cfg['min_disparity'], cfg['num_disparities'], cfg['block_size'])
    # 先将所有位置标记为False，再把完整匹配范围填为True。
    bounds = np.zeros((h, w), bool)
    bounds[y:y + rh, x:x + rw] = True
    return valid & bounds
