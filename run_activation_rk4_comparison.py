"""Classical RK4 isolated muscle activation, NOT MuJoCo whole-body RK4."""
from pathlib import Path
import time,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parent
PREV=Path('/home/sensen/Desktop/engine修复验收/S1-fixed-activation-20261014T225432')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rhs(a,u):
 scale=.5+1.5*np.clip(a,0,1);d=np.clip(u,0,1)-a;return d/np.where(d>0,.0001*scale,.0004/scale)
def main():
 out=Path('/home/sensen/Desktop/engine修复验收')/('S1-activation-rk4-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 paths=[Path(__file__),ROOT/'S1_ACTIVATION_RK4_PROTOCOL.zh-CN.md',PREV/'activation.npz',PREV/'report.json'];hashes={str(p):sha(p) for p in paths}
 old=json.loads((PREV/'report.json').read_text())
 with np.load(PREV/'activation.npz') as z:u=z['commands'].copy();oldref=z['reference_tight_common'].copy()
 a=np.zeros(90);sols=[]
 for k,ctrl in enumerate(u):
  sol=solve_ivp(lambda t,y:rhs(y,ctrl),(k*.005,(k+1)*.005),a,method='DOP853',rtol=1e-12,atol=1e-14,max_step=1e-6,dense_output=True);assert sol.success;sols.append(sol);a=sol.y[:,-1]
 def ref(factor):return np.concatenate([s.sol((k*50*factor+np.arange(1,50*factor+1))*(.0001/factor)).T for k,s in enumerate(sols)])
 refdiff=float(abs(ref(1)-oldref).max());assert refdiff<=1e-10
 arrays=dict(commands=u);cases=[]
 for factor in (1,2,4,8,16):
  dt=.0001/factor;a=np.zeros(90);trace=[];stage_min=0.;stage_max=0.;outside=0;start=time.perf_counter()
  for i in range(500*factor):
   ctrl=u[i//(50*factor)];k1=rhs(a,ctrl);s2=a+dt*k1/2;k2=rhs(s2,ctrl);s3=a+dt*k2/2;k3=rhs(s3,ctrl);s4=a+dt*k3;k4=rhs(s4,ctrl)
   for stage in (a,s2,s3,s4):stage_min=min(stage_min,float(stage.min()));stage_max=max(stage_max,float(stage.max()));outside+=int(((stage<-1e-12)|(stage>1+1e-12)).sum())
   a=a+dt*(k1+2*k2+2*k3+k4)/6;trace.append(a.copy())
  wall=time.perf_counter()-start;trace=np.asarray(trace);expected=ref(factor);err=trace-expected;arrays[f'factor{factor}_trace']=trace;arrays[f'factor{factor}_reference']=expected;arrays[f'factor{factor}_common']=trace[factor-1::factor]
  c=dict(factor=factor,dt_s=dt,rhs_vector_calls=4*len(trace),euler_same_dt_rhs_calls=len(trace),python_single_run_wall_s=wall,activation_min=float(trace.min()),activation_max=float(trace.max()),stage_min=stage_min,stage_max=stage_max,stage_outside_count=outside,max_abs_error=float(abs(err).max()),rms_error=float(np.sqrt(np.mean(err**2))),euler_max_abs_error=old['cases'][len(cases)]['max_abs_error'])
  c['range_pass']=trace.min()>=-1e-12 and trace.max()<=1+1e-12 and outside==0;c['accuracy_pass']=c['max_abs_error']<=.01 and c['rms_error']<=.001;cases.append(c)
 for i,c in enumerate(cases):
  c['next_finer_common_max_difference']=float(abs(arrays[f"factor{c['factor']}_common"]-arrays[f"factor{cases[i+1]['factor']}_common"]).max()) if i+1<len(cases) else None;c['qualified_isolated_activation']=bool(c['range_pass'] and c['accuracy_pass'] and c['next_finer_common_max_difference'] is not None and c['next_finer_common_max_difference']<=.005)
 assert all(sha(p)==h for p,h in hashes.items());result=dict(scope='CLASSICAL_RK4_ACTIVATION_ONLY_NOT_NATIVE_WHOLE_BODY',reference_old_common_max_difference=refdiff,cases=cases,coarsest_qualified_dt_s=next((c['dt_s'] for c in cases if c['qualified_isolated_activation']),None),hashes=hashes,hashes_unchanged=True)
 np.savez_compressed(out/'rk4.npz',**arrays);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4));ax.loglog([c['dt_s']*1e6 for c in cases],[c['max_abs_error'] for c in cases],'o-',label='classical RK4');ax.loglog([c['dt_s']*1e6 for c in cases],[c['euler_max_abs_error'] for c in cases],'s-',label='Euler prior');ax.axhline(.01,color='gray',ls='--',label='declared max error');ax.set(xlabel='dt (us)',ylabel='all-channel all-endpoint max error',title='Isolated activation only — NOT whole-body integrator qualification');ax.legend();fig.tight_layout();fig.savefig(out/'comparison.png');plt.close(fig);print(out);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
