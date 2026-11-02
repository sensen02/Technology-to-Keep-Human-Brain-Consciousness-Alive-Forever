"""Arithmetic sign convention for conditional static Farkas evidence. No solver."""
from pathlib import Path
import json,math,hashlib
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s2-support-dual-convention/20261102T172633Z')
def check(E,b,H,h,y,z,tolerance=1e-8):
 E=np.asarray(E,float);b=np.asarray(b,float);H=np.asarray(H,float);h=np.asarray(h,float);y=np.asarray(y,float);z=np.asarray(z,float)
 if E.ndim!=2 or H.ndim!=2 or E.shape[1]!=H.shape[1] or b.shape!=(len(E),) or h.shape!=(len(H),) or y.shape!=b.shape or z.shape!=h.shape:raise ValueError('Finite dimensionally consistent equality and <= inequality certificate required')
 if not all(np.isfinite(a).all() for a in [E,b,H,h,y,z]):raise ValueError('Finite arrays only')
 stationarity=[math.fsum([float(E[i,j])*float(y[i]) for i in range(len(E))]+[float(H[k,j])*float(z[k]) for k in range(len(H))]) for j in range(E.shape[1])]
 margin=math.fsum([float(b[i])*float(y[i]) for i in range(len(b))]+[float(h[i])*float(z[i]) for i in range(len(h))]);error=max(map(abs,stationarity),default=0.);nonneg=bool(np.all(z>=-tolerance));passed=error<=tolerance and nonneg and margin<=-1+tolerance
 return dict(passed=bool(passed),stationarity_maxabs=float(error),inequality_dual_nonnegative=nonneg,margin=float(margin),tolerance=float(tolerance))
def main():
 O.mkdir(parents=True,exist_ok=False)
 # Ex=b,Hx<=h: E.T*y+H.T*z=0,z>=0,b.T*y+h.T*z<0 contradicts feasibility.
 cases=[dict(label='x1_and_xge2',E=[[1]],b=[1],H=[[-1]],h=[-2],y=[1],z=[1],expected=True),dict(label='x2_and_xle1',E=[[1]],b=[2],H=[[1]],h=[1],y=[-1],z=[1],expected=True),dict(label='wrong_margin_sign',E=[[1]],b=[2],H=[[1]],h=[1],y=[1],z=[-1],expected=False),dict(label='stationarity_wrong',E=[[1]],b=[2],H=[[1]],h=[1],y=[-1],z=[0],expected=False),dict(label='feasible_xhalf_upper1',E=[[1]],b=[.5],H=[[1]],h=[1],y=[-1],z=[1],expected=False)]
 out=[]
 for c in cases:
  record=check(*[c[k] for k in ['E','b','H','h','y','z']]);assert record['passed']==c['expected'];out.append(dict(case=c,result=record))
 result=dict(passed=True,source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},convention='E x=b,H x<=h; free y,z>=0; E.T y+H.T z=0 and b.T y+h.T z<=-1 numerically. Bounds must be explicit H/h rows. Certifies statedmatrix only; a floatresidual is not symbolic proof.',hand_constructed_cases=out,calls=dict(model_compile=0,body_step=0,neural_step=0,linprog=0),limits='No biological or fullphysicalinfeasibilityclaim. No runtimecontroller or matrixwitness generated here.')
 (O/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
