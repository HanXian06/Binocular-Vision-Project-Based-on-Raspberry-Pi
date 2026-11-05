"""复现直接试标定；方格边长 0.015 米，参数尚未验收。"""
from pathlib import Path
import csv
import hashlib
import json
import cv2
import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
OUT = PACKAGE / 'calibration' / 'direct'
ROOT = PACKAGE / 'data'
PAIRS = sorted(ROOT.glob('PartA*/left/*.png'))
def write_image(path, image):
    ok, data = cv2.imencode(path.suffix, image)
    assert ok
    path.write_bytes(data.tobytes())

PATTERN = (8, 6)
obj = np.zeros((48, 3), np.float32)
obj[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 0.015
objects, lefts, rights, ids, rows, manifest, thumbs = [], [], [], [], [], [], []
for lp in PAIRS:
    i = len(rows) + 1
    original_pair = lp.stem[:-2]
    pair = f'pair_{i:04d}'
    session = lp.parent.parent
    images, corners, found = [], [], []
    row = {'pair_id': pair, 'source_session': session.name, 'original_pair': original_pair}
    for side, label in [('left', 'L'), ('right', 'R')]:
        path = session / side / f'{original_pair}_{label}.png'
        data = path.read_bytes()
        manifest.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest()})
        im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert im.shape[:2] == (480, 640), (path, im.shape)
        ok, c = cv2.findChessboardCornersSB(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), PATTERN,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        row[f'{label}_corners_found'] = bool(ok)
        found.append(ok)
        corners.append(c)
        images.append(im)
    if all(found):
        # 同一视场的小角度横向双目：统一棋盘格遍历方向。
        if np.dot(corners[0][-1, 0] - corners[0][0, 0], corners[1][-1, 0] - corners[1][0, 0]) < 0:
            corners[1] = corners[1][::-1].copy()
        objects.append(obj.copy())
        lefts.append(corners[0]); rights.append(corners[1]); ids.append(pair)
        for j in range(2):
            cv2.drawChessboardCorners(images[j], PATTERN, corners[j], True)
    thumb = cv2.resize(cv2.hconcat(images), (640, 240))
    cv2.putText(thumb, pair, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    thumbs.append(thumb)
    rows.append(row)
    print(pair, found, flush=True)

write_image(OUT / 'corners_contact.jpg', cv2.vconcat([cv2.hconcat(thumbs[k:k+2]) for k in range(0,20,2)]))
(OUT / 'sha256_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
size = (640, 480)
rmsL, K1, D1, rv1, tv1 = cv2.calibrateCamera(objects, lefts, size, None, None)
rmsR, K2, D2, rv2, tv2 = cv2.calibrateCamera(objects, rights, size, None, None)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objects, lefts, rights,
        K1, D1, K2, D2, size, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(K1,D1,K2,D2,size,R,T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
for idx, pair in enumerate(ids):
    row = next(x for x in rows if x['pair_id'] == pair)
    for label, K, D, rv, tv, observed in [('L',K1,D1,rv1,tv1,lefts),('R',K2,D2,rv2,tv2,rights)]:
        pred, _ = cv2.projectPoints(obj,rv[idx],tv[idx],K,D)
        row[f'{label}_reprojection_rms_px'] = float(np.sqrt(np.mean(np.sum((pred-observed[idx])**2,axis=2))))
    a = cv2.undistortPoints(lefts[idx],K1,D1,R=R1,P=P1).reshape(-1,2)
    b = cv2.undistortPoints(rights[idx],K2,D2,R=R2,P=P2).reshape(-1,2)
    dy = np.abs(a[:,1]-b[:,1])
    row['rectified_vertical_median_px'] = float(np.median(dy))
    row['rectified_vertical_p95_px'] = float(np.percentile(dy,95))
    row['rectified_vertical_max_px'] = float(dy.max())
    if idx in (0, 9, 17, 18, 19):
        ims=[]
        for side,label,K,D,rr,pp in [('left','L',K1,D1,R1,P1),('right','R',K2,D2,R2,P2)]:
            im=cv2.imdecode(np.frombuffer((ROOT / row['source_session'] / side / f"{row['original_pair']}_{label}.png").read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            mx,my=cv2.initUndistortRectifyMap(K,D,rr,pp,size,cv2.CV_32FC1)
            ims.append(cv2.remap(im,mx,my,cv2.INTER_LINEAR))
        canvas=cv2.hconcat(ims)
        for y in range(0,480,30): cv2.line(canvas,(0,y),(1279,y),(0,255,0),1)
        write_image(OUT/f'{pair}_rectified.jpg',canvas)
fields = sorted(set().union(*(r.keys() for r in rows)))
with (OUT/'per_pair.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
np.savez(OUT/'trial_calibration_meters.npz',K1=K1,D1=D1,K2=K2,D2=D2,R=R,T=T,R1=R1,R2=R2,P1=P1,P2=P2,Q=Q,image_size=size,square_size=0.015,pattern=PATTERN)
summary={'opencv':cv2.__version__,'pattern_inner_corners':PATTERN,'square_size':'0.015 meters',
 'valid_pairs':ids,'mono_rms_L_px':rmsL,'mono_rms_R_px':rmsR,'stereo_rms_px':rms,
 'K1':K1.tolist(),'K2':K2.tolist(),'D1':D1.tolist(),'D2':D2.tolist(),'T_m':T.tolist(),
 'baseline_m':float(np.linalg.norm(T)), 'R':R.tolist(),
 'vertical_median_per_pair_px':[r.get('rectified_vertical_median_px') for r in rows],
 'warning':'Trial fit on calibration data only. Metric calibration; not independent accuracy validation.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
