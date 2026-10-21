"""Independent activation-only fixed-command integration; no body model load."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import mujoco
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parent
P=Path('/home/sensen/Desktop/engine修复验收/B-local-replay-20261014T213817')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rhs(a,u):
 s=.5+1.5*np.clip(a,0,1);d=np.clip(u,0,1)-a;return d/np.maximum(np.where(d>0,.0001*s,.0004/s),1e-15)
def main():
 out=Path('/home/sensen/Desktop/engine修复验收')/('S1-fixed-activation-'+time.strftime('%Y%m%dT%H%M%S'));out.mkdir(exist_ok=False)
 with np.load(P/'baseline.npz') as z:commands=z['applied_controls'].copy();archived=z['act'].copy()
 meta=ROOT/'outputs/s2_fixed_stimulus/20261014_100204_7b35ada0e37c/native_floor/metadata.json';m=json.loads(meta.read_text());assert np.array_equal(np.asarray(m['actuator_dynprm'])[:,:3],np.tile([.0001,.0004,0],(90,1)));assert np.array_equal(m['actuator_actadr'],np.arange(90));assert not any(m['actuator_actlimited']);assert m['integrator']==0
 nativeerr=max(abs(float(rhs(np.array([a]),u)[0])-mujoco.mju_muscleDynamics(u,a,np.array([.0001,.0004,0]))) for a in (-.2,0,.2,.8,1,1.5,2) for u in (0,.2,.8,1));assert nativeerr<=1e-9
 def reference(tight):
  a=np.zeros(90);sols=[]
  for k,u in enumerate(commands):
   sol=solve_ivp(lambda t,y:rhs(y,u),(k*.005,(k+1)*.005),a,method='DOP853',rtol=1e-12 if tight else 1e-11,atol=1e-14 if tight else 1e-13,max_step=1e-6 if tight else 2e-6,dense_output=True);assert sol.success;sols.append(sol);a=sol.y[:,-1]
  return sols
 refs=reference(False);tight=reference(True)
 def evaluate(sols,steps):
  per=steps//10;return np.concatenate([s.sol((k*per+np.arange(1,per+1))*(.05/steps)).T for k,s in enumerate(sols)])
 refcommon=evaluate(refs,500);tightcommon=evaluate(tight,500);referr=float(abs(refcommon-tightcommon).max());assert referr<=1e-8
 raw=dict(commands=commands,reference_common=refcommon,reference_tight_common=tightcommon);results=[]
 for factor in (1,2,4,8,16):
  dt=.0001/factor;a=np.zeros(90);trace=[]
  for i in range(500*factor):a=a+dt*rhs(a,commands[i//(50*factor)]);trace.append(a.copy())
  trace=np.array(trace);expected=evaluate(tight,500*factor);err=trace-expected;common=trace[factor-1::factor];raw[f'factor{factor}_common']=common
  item=dict(factor=factor,dt_s=dt,activation_min=float(trace.min()),activation_max=float(trace.max()),above_one_count=int((trace>1+1e-12).sum()),max_abs_error=float(abs(err).max()),rms_error=float(np.sqrt(np.mean(err**2))),range_pass=bool(trace.min()>=-1e-12 and trace.max()<=1+1e-12));item['accuracy_pass']=item['max_abs_error']<=.01 and item['rms_error']<=.001;results.append(item)
  if factor==1:
   raw['euler100us_all']=trace;raw['archived_act']=archived;replayerr=float(abs(trace-archived).max());exact=bool(np.array_equal(trace,archived));assert replayerr<=1e-12
 for i,c in enumerate(results):
  c['next_finer_common_max_difference']=float(abs(raw[f"factor{c['factor']}_common"]-raw[f"factor{results[i+1]['factor']}_common"]).max()) if i+1<len(results) else None;c['qualified_isolated_activation']=c['range_pass'] and c['accuracy_pass'] and c['next_finer_common_max_difference'] is not None and c['next_finer_common_max_difference']<=.005
 paths=[Path(__file__),ROOT/'S1_FIXED_COMMAND_ACTIVATION_PROTOCOL.zh-CN.md',P/'baseline.npz',meta,ROOT/'run_activation_timestep_audit.py'];result=dict(scope='ACTIVATION_ONLY_NO_BODY_OR_NEURAL_SIMULATION',native_rhs_max_error=nativeerr,reference_selfcheck_max_error=referr,archived_euler_exact=exact,archived_euler_max_error=replayerr,cases=results,hashes={str(p):sha(p) for p in paths},environment=dict(mujoco=mujoco.__version__,numpy=np.__version__))
 np.savez_compressed(out/'activation.npz',**raw);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,1,figsize=(9,6));t=np.arange(1,501)*.1
 for factor in (1,2,4,8,16):axes[0].plot(t,raw[f'factor{factor}_common'][:,14],label=f'{100/factor:g}us Euler')
 axes[0].plot(t,tightcommon[:,14],'k--',label='DOP853 reference');axes[0].set(xlabel='time (ms)',ylabel='activation channel14');axes[0].legend(ncol=3)
 axes[1].loglog([c['dt_s']*1e6 for c in results],[c['max_abs_error'] for c in results],'o-');axes[1].axhline(.01,color='gray',ls='--');axes[1].set(xlabel='Euler dt (us)',ylabel='all-channel all-step max error');fig.suptitle('Fixed archived neural commands: activation-only qualification, NOT behavior');fig.tight_layout();fig.savefig(out/'activation.png');plt.close(fig)
 print(out);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
