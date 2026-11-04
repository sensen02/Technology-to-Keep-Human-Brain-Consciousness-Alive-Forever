"""Fit the 115-dim learning curve and extrapolate the ceiling it is heading for.

The curve is a monotone staircase (running best), so an exponential approach is used and the fit is
checked on several windows; the spread between windows is reported as the uncertainty instead of a
single confident number. Reading the archived history only. modes: fit
"""
import json,numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
O=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-connectivity-training/20261104T090000Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-rebuilt-regression-analysis/20261104T160000Z')
def f(x,J,A,t,x0):return J+A*np.exp(-(x-x0)/t)
def fit():
 h=json.loads((O/'CMA_HISTORY_V2.json').read_text())
 g=np.array([r['generation'] for r in h],float);run=np.minimum.accumulate([r['best'] for r in h])
 ev=np.array([r['evals'] for r in h],float)
 fits={}
 for lo in (0,20):
  p,_=curve_fit(lambda x,J,A,t:f(x,J,A,t,g[0]),g[lo:],run[lo:],p0=[3.44,0.16,30.],maxfev=400000)
  fits[f'from_gen_{lo}']=dict(J_inf=float(p[0]),A=float(p[1]),tau_generations=float(p[2]),tau_evals=float(p[2]*16))
 p,_=curve_fit(lambda x,J,A,t:f(x,J,A,t,g[0]),g,run,p0=[3.44,0.16,30.],maxfev=400000)
 table=[dict(generations=n,evaluations=int(n*16),predicted_J=float(f(np.array([float(n)]),p[0],p[1],p[2],g[0])[0])) for n in (48,64,80,94,128,188,376,752)]
 res=dict(observed=dict(generations=int(g[-1]+1),evaluations=int(ev[-1]),
   best=float(run[-1]),running_best_every4=np.round(run[::4],4).tolist(),
   last12_improvement=float(run[-13]-run[-1])),
  fits=fits,
  extrapolation=table,
  cannot_reach_old_record=bool(p[0]>3.3902926855730198),
  old_record=3.3902926855730198,
  uncertainty=dict(spread_across_windows=float(abs(fits['from_gen_0']['J_inf']-fits['from_gen_20']['J_inf'])),
   reason='48 monotone points with five restarts; restarts are discrete jumps, so a smooth fit understates the chance of an occasional lucky jump'),
  caveat='this is conditional on the same 115-dim parameter space, the same sigma schedule and the same restart policy; it is not a statement about what a different input or parameter space could reach')
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'CEILING_EXTRAPOLATION.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(observed=res['observed'],fits=fits,table=table,J_inf=table[-1]['predicted_J']),ensure_ascii=False,indent=1))
if __name__=='__main__':fit()
