"""v0.1 离线区域测距入口。

从配置读取左右图片和相机参数，依次完成校正、视差计算、深度换算和 ROI 统计，
最后保存数值结果与预览。ROI 是你选定的待测物体表面，坐标属于左侧校正图。
可先读 main() 中的分段注释，再查看 stereo_ranging 下的三个算法模块。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from stereo_ranging.geometry import rectify, depth_from_disparity
from stereo_ranging.matching import compute_disparity, valid_geometry
from stereo_ranging.region import measure_region
from stereo_ranging.consistency import compute_right_disparity, left_right_consistency


def read_image(path):
    """读取图片为 OpenCV 使用的 BGR 数组；先读文件字节可兼容中文路径。"""
    # read_bytes 读入文件；frombuffer 将字节视为数组；imdecode 解码出图片。
    im = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f'Cannot read image: {path}')
    return im


def save_image(path, im):
    """按文件扩展名编码图片，再写入文件，兼容中文路径。"""
    ok, buf = cv2.imencode(path.suffix, im)
    if not ok:
        raise OSError(f'Cannot encode: {path}')
    path.write_bytes(buf.tobytes())


def sha(path):
    """计算文件的 SHA-256 校验值，相当于给本次使用的输入文件留一份指纹。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """串起一次离线测距：解析配置 → 读图 → 校正 → 选框 → 匹配 → 区域统计 → 保存。"""
    # 一、命令行选项：--config 选配置，--out 选输出目录，--select-roi 开启鼠标选框。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/sample_pair0006.json')
    parser.add_argument('--out', type=Path, help='New result directory, relative to current working directory')
    parser.add_argument('--select-roi', action='store_true', help='Select ROI on rectified left image')
    parser.add_argument('--prepare', action='store_true', help='Export rectified images for ROI selection, without matching')
    parser.add_argument('--check-consistency', action='store_true', help='Independently check right-to-left correspondence')
    args = parser.parse_args()
    # 二、读取配置。resolve() 将路径变成绝对路径，后续定位文件更明确。
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    # 可由命令行开启，也可在配置中写consistency.enabled=true；结果配置会记录实际选择。
    if args.check_consistency:
        cfg['consistency'] = {**cfg.get('consistency', {}), 'enabled': True}
    check_cfg = cfg.get('consistency', {})
    check_enabled = check_cfg.get('enabled', False)
    if check_enabled:
        cfg['consistency'] = check_cfg = {'max_diff_px': 1.0, **check_cfg}
    # 配置中的相对路径以配置文件所在文件夹为起点。
    base = config_path.parent
    # 这里的字典分别保存左图、右图、相机参数三个实际路径。
    paths = {k: (base/cfg[k]).resolve() for k in ('left_image', 'right_image', 'camera_parameters')}
    # 将相机参数的文件指纹与配置核对，保证这次运行使用约定的参数副本。
    param_hash = sha(paths['camera_parameters'])
    if param_hash != cfg['camera_sha256']:
        raise ValueError('Camera parameter hash differs from the configuration')
    # NPZ 是保存多个数组的文件；camera 字典中包含 K、D、R、P、Q 等矩阵。
    with np.load(paths['camera_parameters'], allow_pickle=False) as archive:
        camera = dict(archive)
    output = (args.out or (config_path.parent.parent/'results'/cfg['run_id'])).resolve()
    # 每次使用新目录，方便把不同 ROI 或不同参数的结果分别保存。
    output.mkdir(parents=True, exist_ok=False)
    # perf_counter 用于计时；秒乘 1000 后得到毫秒。每段耗时存进 timing 字典。
    timing = {}
    start = perf_counter()
    # 三、左右原图按这个顺序放入列表：images[0] 是左图，images[1] 是右图。
    images = [read_image(paths[k]) for k in ('left_image', 'right_image')]
    timing['read_ms'] = (perf_counter()-start)*1000
    start = perf_counter()
    # 四、应用相机几何。*images 将列表展开为左图和右图两个函数参数。
    rectified, supports = rectify(*images, camera)
    timing['rectify_ms'] = (perf_counter()-start)*1000
    # 校正图会保存，便于你查看真正参与匹配的图像。
    for side, im in zip(('left', 'right'), rectified):
        save_image(output/f'{side}_rectified.png', im)
    # --prepare 到这里结束，只输出校正图，供查看构图和选择目标表面。
    if args.prepare:
        print(f'Rectified images: {output}')
        return
    # 五、确定 ROI：默认使用配置中的矩形；鼠标选框时使用本次选择替代。
    roi = cfg.get('roi_xywh')
    # 选框窗口显示左侧校正图；拖动框住待测物体的实体表面，按 Enter 确认。
    if args.select_roi:
        roi = list(map(int, cv2.selectROI('Select target surface, ENTER confirms', rectified[0], False, False)))
        cv2.destroyAllWindows()
    if roi is None or len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
        raise ValueError('Set roi_xywh in config or use --select-roi')
    start = perf_counter()
    # 六、BGR 转为单通道灰度，SGBM 在左右灰度图之间寻找对应位置。
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in rectified]
    # raw 是带16倍缩放的整数输出；disparity 已换成像素单位。
    raw, disparity = compute_disparity(*gray, cfg['sgbm'])
    timing['matching_ms'] = (perf_counter()-start)*1000
    start = perf_counter()
    # 七、用 Q 将视差换算成 Z 深度。相机参数以米为单位，输出也以米为单位。
    depth = depth_from_disparity(disparity, camera['Q'])
    # valid 综合判断哪些像素具有可用于统计的深度。
    valid = valid_geometry(disparity, raw, depth, supports, cfg['sgbm'])
    # ~valid 表示所有未保留的位置；NaN 用于记录这些位置没有有效深度。
    depth[~valid] = np.nan
    timing['reproject_filter_ms'] = (perf_counter()-start)*1000
    extra_arrays = {}
    before_consistency = None
    check_info = None
    if check_enabled:
        # 先保留原方法的区域统计，能直接比较“检查前后”，无需根据参考距离筛选像素。
        before_consistency = measure_region(depth, valid, roi, **cfg['region_thresholds'])
        valid_before = valid.copy()
        depth_before = depth.copy()
        start = perf_counter()
        raw_right, disparity_right, right_valid = compute_right_disparity(*gray, supports, cfg['sgbm'])
        timing['right_matching_ms'] = (perf_counter()-start)*1000
        start = perf_counter()
        checked = left_right_consistency(disparity, disparity_right, valid, right_valid,
                                         check_cfg['max_diff_px'])
        valid = checked['consistent']
        depth[~valid] = np.nan
        x, y, w, h = before_consistency['roi_xywh']
        area = np.s_[y:y+h, x:x+w]
        base_count = before_consistency['valid_pixels']
        check_count = int(checked['checked'][area].sum())
        pass_count = int(valid[area].sum())
        values = checked['residual_px'][area][checked['checked'][area]]
        check_info = dict(enabled=True, max_diff_px=check_cfg['max_diff_px'],
            roi_base_valid_pixels=base_count, roi_checked_pixels=check_count,
            roi_passed_pixels=pass_count,
            roi_pass_fraction_of_base=pass_count/base_count if base_count else 0.0,
            roi_pass_fraction_of_checked=pass_count/check_count if check_count else 0.0,
            roi_residual_median_px=float(np.median(values)) if len(values) else None)
        extra_arrays = dict(disparity_right_raw_x16=raw_right, disparity_right_px=disparity_right,
            right_disparity_valid=right_valid, valid_before_consistency=valid_before,
            depth_before_consistency_m=depth_before, consistency_checked=checked['checked'],
            consistency_residual_px=checked['residual_px'])
        timing['consistency_ms'] = (perf_counter()-start)*1000
    start = perf_counter()
    # 八、统计 ROI。** 将阈值字典展开为 min_pixels、min_fraction 两个命名参数。
    region = measure_region(depth, valid, roi, **cfg['region_thresholds'])
    timing['region_ms'] = (perf_counter()-start)*1000
    # 保存实际使用的配置，包含最终 ROI 和绝对路径，便于重复运行。
    resolved = dict(cfg)
    resolved.update(roi_xywh=region['roi_xywh'], **{k:str(v) for k,v in paths.items()})
    # 参考距离来自拍摄记录，此处用于计算估计值与参考值的差。
    reference = cfg.get('reference_distance_m')
    # 九、组织结构化结果。**region 将区域统计的各个字段放进结果字典。
    result = {'version':'v0.1', 'pair_id':cfg['pair_id'], 'distance_definition':'rectified_left_optical_axis_Z',
        'unit':'m', **region, 'reference_distance_m':reference,
        'absolute_error_m': abs(region['depth_m']-reference) if region['available'] and reference is not None else None,
        'timing_ms':timing, 'compute_total_ms':sum(timing.values()),
        'parameter_sha256':param_hash, 'left_sha256':sha(paths['left_image']),
        'right_sha256':sha(paths['right_image']), 'opencv_version':cv2.__version__, 'numpy_version':np.__version__,
        'notes':'Availability is pixel support only; not a calibrated probability of accuracy.'}
    if check_enabled:
        result['consistency_check'] = check_info
        result['before_consistency'] = {**before_consistency,
            'absolute_error_m': abs(before_consistency['depth_m']-reference)
                if before_consistency['available'] and reference is not None else None}
    start = perf_counter()
    # 十、保存数值数组。后续分析应读取这些数值，预览图片只用于观察。
    np.savez_compressed(output/'arrays.npz', disparity_raw_x16=raw, disparity_px=disparity,
        depth_m=depth, valid_mask=valid, left_support=supports[0], right_support=supports[1], **extra_arrays)
    # 制作视差色彩图：按搜索范围缩放到0～255，转换为8位图像。
    d8 = np.uint8(np.clip((disparity-cfg['sgbm']['min_disparity'])/cfg['sgbm']['num_disparities'],0,1)*255)
    preview = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
    # 将没有有效结果的位置涂黑，方便在预览中识别。
    preview[~valid] = 0
    save_image(output/'disparity_preview.png', preview)
    # 深度色标上限由配置指定；这个数只影响颜色显示。
    zmax = cfg['visualization_depth_max_m']
    # 显示时将 NaN 临时转为0，再按固定色标着色；原始 depth 数组仍保留 NaN。
    z8 = np.uint8(np.clip(np.nan_to_num(depth,nan=0)/zmax,0,1)*255)
    zview = cv2.applyColorMap(z8,cv2.COLORMAP_TURBO)
    zview[~valid] = 0
    save_image(output/'depth_preview.png',zview)
    save_image(output/'valid_mask.png',valid.astype(np.uint8)*255)
    if check_enabled:
        # 绿色通过；红色往返偏差超限；蓝色没有有效右点可比较；原本无效的位置保持黑色。
        check_view = np.zeros((*valid.shape, 3), np.uint8)
        check_view[valid_before & ~checked['checked']] = (255, 0, 0)
        check_view[checked['checked'] & ~valid] = (0, 0, 255)
        check_view[valid] = (0, 255, 0)
        save_image(output/'consistency_mask.png', check_view)
    # 复制左校正图，在副本上画出 ROI 和数值，原校正图保持原样。
    left = rectified[0].copy()
    # ROI 中 x/y 是左上角，w/h 是宽高；矩形最后一个像素为 x+w-1、y+h-1。
    x,y,w,h = region['roi_xywh']
    cv2.rectangle(left,(x,y),(x+w-1,y+h-1),(0,255,0),2)
    title = f"Z={region['depth_m']:.3f}m" if region['available'] else 'Distance unavailable'
    cv2.putText(left,title,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
    cv2.putText(left,f"valid={region['valid_fraction']:.1%} N={region['valid_pixels']}",(8,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
    save_image(output/'result_preview.png',cv2.hconcat([left,zview]))
    # 文件编码、写盘时间单独记录，与前面的核心计算耗时分开。
    result['artifact_write_ms'] = (perf_counter()-start)*1000
    (output/'config_resolved.json').write_text(json.dumps(resolved,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    # CSV 每格存一个值：先取普通字段，再将 ROI、原因及耗时展开成列。
    flat={k:v for k,v in result.items() if not isinstance(v,(dict,list))}
    flat['roi_xywh']=json.dumps(region['roi_xywh'])
    flat['reasons']=';'.join(region['reasons'])
    flat.update(timing)
    if check_enabled:
        flat.update({f'consistency_{k}':v for k,v in check_info.items()})
    with (output/'result.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print(f'Results: {output}')


# 直接运行本文件时执行 main()；被其他文件导入时只提供函数。
if __name__=='__main__':
    main()
