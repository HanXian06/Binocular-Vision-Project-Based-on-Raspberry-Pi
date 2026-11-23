"""相机几何处理：把左右原图变成校正图，再把像素视差换算为米单位深度。

图像数组的形状是（高、宽、颜色通道），像素坐标写作（x、y）。
K 是相机内参，D 是畸变系数，R1/R2 和 P1/P2 用于左右校正，Q 用于深度重建。"""
import cv2
import numpy as np


def rectify(left, right, camera):
    """输入两张 BGR 原图和相机参数，返回两张校正图及左右有效画面掩码。"""
    # OpenCV 的尺寸参数顺序是（宽、高），与数组形状的（高、宽）相反。
    size = (left.shape[1], left.shape[0])
    # 从相机文件读取适用尺寸；旧参数未存尺寸字段时使用本项目的640×480。
    expected = tuple(int(x) for x in camera.get('image_size', (640, 480)))
    if left.shape != right.shape or size != expected:
        raise ValueError(f'Expected two {expected} images; received {left.shape}, {right.shape}')
    # 分别收集左右校正图，以及哪些位置来自原图的有效标记。
    images, masks = [], []
    # i=1 为左相机，i=2 为右相机，借此读取 K1/D1 和 K2/D2 等字段。
    for i, image in enumerate((left, right), 1):
        # 建立查找表：校正图每个像素，需要去原图的哪个 x/y 位置取颜色。
        mx, my = cv2.initUndistortRectifyMap(camera[f'K{i}'], camera[f'D{i}'],
            camera[f'R{i}'], camera[f'P{i}'], size, cv2.CV_32FC1)
        # 按查找表重采样；双线性插值用原图相邻像素计算新像素颜色。
        images.append(cv2.remap(image, mx, my, cv2.INTER_LINEAR))
        # 将原图全置为1，再进行同样的映射；原图外默认是0。
        support = cv2.remap(np.ones(image.shape[:2], np.float32), mx, my,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # 接近1表示插值需要的来源都在原图内；边界混入0的位置会被排除。
        masks.append(support >= .999)
    return images, masks


def depth_from_disparity(disparity, q):
    """输入像素单位视差和 Q 矩阵，返回每个左图像素的 Z 深度。"""
    # 重建结果形状为（高、宽、3），最后一维依次为 X、Y、Z；这里只取 Z。
    return cv2.reprojectImageTo3D(np.asarray(disparity, np.float32), q,
                                handleMissingValues=False)[:, :, 2]
