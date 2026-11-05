from pathlib import Path
import json, cv2, numpy as np
package=Path(__file__).resolve().parent.parent
out=package/'calibration'/'background'
root=package/'data'
a=np.load(out.parent/'direct'/'trial_calibration_meters.npz'); sift=cv2.SIFT_create(nfeatures=2500)
ps=[[],[]]
for lp in sorted(root.glob('PartA*/left/*.png')):
 rp=lp.parent.parent/'right'/lp.name.replace('_L','_R')
 ims=[cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),0) for p in (lp,rp)]
 kp=[];ds=[]
 for im in ims:
  mask=np.full(im.shape,255,np.uint8)
  ok,c=cv2.findChessboardCornersSB(im,(8,6))
  if ok:
   x,y,w,h=cv2.boundingRect(c);cv2.rectangle(mask,(max(0,x-45),max(0,y-45)),(x+w+45,y+h+45),0,-1)
  k,d=sift.detectAndCompute(im,mask);kp.append(k);ds.append(d)
 matches=cv2.BFMatcher().knnMatch(ds[0],ds[1],k=2)
 good=[m for m,n in matches if m.distance<.7*n.distance]
 for j in range(2): ps[j].extend([kp[j][m.queryIdx if j==0 else m.trainIdx].pt for m in good])
x=[cv2.undistortPoints(np.float32(p).reshape(-1,1,2),a['K1' if j==0 else 'K2'],a['D1' if j==0 else 'D2']).reshape(-1,2) for j,p in enumerate(ps)]
cv2.setRNGSeed(42)
E,mask=cv2.findEssentialMat(x[0],x[1],np.eye(3),method=cv2.RANSAC,prob=.999,threshold=.001)
n,R,t,mask2=cv2.recoverPose(E,x[0],x[1],np.eye(3),mask=mask)
T=t*.0585
R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(a['K1'],a['D1'],a['K2'],a['D2'],(640,480),R,T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
np.savez(out/'background_calibration_meters.npz',K1=a['K1'],D1=a['D1'],K2=a['K2'],D2=a['D2'],R=R,T=T,R1=R1,R2=R2,P1=P1,P2=P2,Q=Q)
y=[cv2.undistortPoints(np.float32(ps[j]).reshape(-1,1,2),a['K1' if j==0 else 'K2'],a['D1' if j==0 else 'D2'],R=R1 if j==0 else R2,P=P1 if j==0 else P2).reshape(-1,2) for j in range(2)]
dy=np.abs(y[0][:,1]-y[1][:,1])[mask2.ravel()>0]
r={'matches':len(x[0]),'pose_inliers':n,'inlier_vertical_median_px':float(np.median(dy)),'inlier_vertical_p95_px':float(np.percentile(dy,95)),'baseline_m':.0585,'baseline_source':'user measured, imposed, not independently estimated','R':R.tolist(),'T':T.tolist()}
(out/'background_summary.json').write_text(json.dumps(r,indent=2))
print(r)
