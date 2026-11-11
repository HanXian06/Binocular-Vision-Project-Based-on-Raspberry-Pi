"""Fixed-baseline joint reprojection fit. Does not run depth/detection or upload."""
from pathlib import Path
import sys,json,csv,hashlib
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
BASE=Path(__file__).resolve().parent.parent
OUT=BASE/'results';OUT.mkdir(exist_ok=True)
a=np.load(BASE/'reference/direct_previous.npz')
B=.0585;K=[a['K1'],a['K2']];D=[a['D1'],a['D2']]
obj=np.zeros((48,3),np.float64);obj[:,:2]=np.mgrid[:8,:6].T.reshape(-1,2)*.015
pairs=[];corn=[];initial=[];manifest=[]
for lp in sorted((BASE/'data').glob('PartA*/left/*.png')):
 rp=lp.parent.parent/'right'/lp.name.replace('_L','_R')
 cc=[]
 for p in [lp,rp]:
  raw=p.read_bytes();manifest.append({'file':p.relative_to(BASE).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()})
  im=cv2.imdecode(np.frombuffer(raw,np.uint8),0);assert im.shape==(480,640)
  ok,c=cv2.findChessboardCornersSB(im,(8,6),flags=cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY);assert ok,p
  cc.append(c.reshape(-1,2).astype(np.float64))
 if np.dot(cc[0][-1]-cc[0][0],cc[1][-1]-cc[1][0])<0:cc[1]=cc[1][::-1].copy()
 ok,r,t=cv2.solvePnP(obj,cc[0],K[0],D[0]);assert ok
 initial.append(np.r_[r.ravel(),t.ravel()]);corn.append(cc);pairs.append(lp)
N=len(pairs);assert N==20
np.savez(BASE/'corners.npz',object_points_m=obj,left=np.array(corn)[:,0],right=np.array(corn)[:,1])
def extrinsic(g):
 R=cv2.Rodrigues(g[:3])[0];v=np.array([-1.,g[3],g[4]]);return R,B*v/np.linalg.norm(v)
def pack(R,T,ids):
 T=np.asarray(T).ravel()
 return np.r_[cv2.Rodrigues(R)[0].ravel(),T[1:3]/(-T[0]),np.array(initial)[ids].ravel()]
def residual(x,ids):
 R,T=extrinsic(x[:5]);poses=x[5:].reshape(-1,6);res=[]
 for p,i in zip(poses,ids):
  rl=cv2.Rodrigues(p[:3])[0];rr=cv2.Rodrigues(R@rl)[0];tr=R@p[3:]+T
  ql=cv2.projectPoints(obj,p[:3],p[3:],K[0],D[0])[0].reshape(-1,2)
  qr=cv2.projectPoints(obj,rr,tr,K[1],D[1])[0].reshape(-1,2)
  res.extend([(ql-corn[i][0]).ravel(),(qr-corn[i][1]).ravel()])
 return np.concatenate(res)
def fit(ids,R,T):
 x=pack(R,T,ids);s=lil_matrix((192*len(ids),len(x)),dtype=int)
 for j in range(len(ids)):
  s[j*192:(j+1)*192,5+j*6:5+(j+1)*6]=1;s[j*192+96:(j+1)*192,:5]=1
 result=least_squares(residual,x,args=(ids,),jac_sparsity=s.tocsr(),x_scale='jac',ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=400)
 print('FIT',len(ids),result.success,result.nfev,float(np.sum(result.fun**2)),flush=True)
 return result
ids=list(range(N));fits=[]
for file in ['direct_previous.npz','background_previous.npz']:
 ref=np.load(BASE/'reference'/file);fits.append(fit(ids,ref['R'],ref['T']))
best=min(fits,key=lambda r:np.sum(r.fun**2));R,T=extrinsic(best.x[:5])
assert abs(np.linalg.norm(T)-B)<1e-12
R1,R2,P1,P2,Q,roi1,roi2=cv2.stereoRectify(K[0],D[0],K[1],D[1],(640,480),R,T.reshape(3,1),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
np.savez(OUT/'fixed_baseline_candidate_meters.npz',K1=K[0],D1=D[0],K2=K[1],D2=D[1],R=R,T=T.reshape(3,1),R1=R1,R2=R2,P1=P1,P2=P2,Q=Q,image_size=[640,480],square_size_m=.015,baseline_m=B,pattern=[8,6],unit='m',status='candidate_not_accepted')
def metrics(R,T,idx):
 rect=cv2.stereoRectify(K[0],D[0],K[1],D[1],(640,480),R,T.reshape(3,1),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
 pts=[cv2.undistortPoints(corn[idx][j].reshape(-1,1,2),K[j],D[j],R=rect[j],P=rect[j+2]).reshape(-1,2) for j in [0,1]]
 err=np.abs(pts[0][:,1]-pts[1][:,1]);return float(np.median(err)),float(np.percentile(err,95))
rows=[]
for i,lp in enumerate(pairs):
 old=metrics(a['R'],a['T'],i);new=metrics(R,T,i)
 row={'index':i+1,'session':lp.parent.parent.name,'original_pair':lp.stem[:-2],'old_vertical_median_px':old[0],'new_vertical_median_px':new[0],'old_vertical_p95_px':old[1],'new_vertical_p95_px':new[1]};rows.append(row)
 if i in [0,2,7,12,17,19]:
  ims=[]
  for j,p in enumerate([lp,lp.parent.parent/'right'/lp.name.replace('_L','_R')]):
   im=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),1);maps=cv2.initUndistortRectifyMap(K[j],D[j],R1 if j==0 else R2,P1 if j==0 else P2,(640,480),cv2.CV_32FC1);ims.append(cv2.remap(im,*maps,cv2.INTER_LINEAR))
  preview=cv2.hconcat(ims)
  for y in range(0,480,30):cv2.line(preview,(0,y),(1279,y),(0,255,0),1)
  ok,data=cv2.imencode('.jpg',preview);assert ok;(OUT/f'pair_{i+1:04d}_rectified.jpg').write_bytes(data.tobytes())
with (OUT/'per_pair.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
# Extrinsic-only validation: intrinsics were originally fitted using all 20 images.
held=[2,7,12,17];train=[i for i in ids if i not in held];v=fit(train,R,T);vr,vt=extrinsic(v.x[:5])
validation=[{'index':i+1,'vertical_median_px':metrics(vr,vt,i)[0],'vertical_p95_px':metrics(vr,vt,i)[1]} for i in held]
np.savez(OUT/'validation_16pair_extrinsic.npz',R=vr,T=vt.reshape(3,1),train_indices_1based=np.array(train)+1,heldout_indices_1based=np.array(held)+1)
summary={'baseline_m':float(np.linalg.norm(T)),'old_baseline_m':float(np.linalg.norm(a['T'])),'intrinsics_policy':'frozen previous K/D fitted on all 20 pairs','loss':'linear least squares; no pairs discarded','full_fit_success':bool(best.success),'full_fit_nfev':best.nfev,'stereo_reprojection_rms_px':float(np.sqrt(np.sum(best.fun**2)/(N*48*2))),'old_stereo_rms_px':3.975068802722845,'R':R.tolist(),'T_m':T.tolist(),'median_of_pair_vertical_medians_px':float(np.median([r['new_vertical_median_px'] for r in rows])),'old_median_of_pair_vertical_medians_px':float(np.median([r['old_vertical_median_px'] for r in rows])),'starts':[{'success':bool(f.success),'squared_error':float(np.sum(f.fun**2))} for f in fits],'extrinsic_validation':validation,'validation_success':bool(v.success),'warning':'Fixed baseline is an input constraint, not accuracy evidence. Heldout images were used for previous intrinsics. Candidate not accepted.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(BASE/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
