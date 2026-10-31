"""S160 matched 25us-grid comparison with preserved additional half samples."""
from pathlib import Path
import argparse,json,csv,xml.etree.ElementTree as ET
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_muscle_reconstruction import evaluate
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
def load(p):
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;a=load(REF);b=load(p);n=min(len(a['time']),(len(b['time'])+1)//2);assert np.allclose(a['time'][:n],b['time'][:2*n:2],rtol=0,atol=1e-14);t=a['time'][:n];nodes=list(ET.parse('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml').getroot().find('actuator'));fields={};summary={};fig,axs=plt.subplots(4,3,figsize=(15,14));records=[]
 for name,x,stride in [('S154 25us',a,1),('S160 12.5us',b,2)]:
  L=x['actuator_length'];V=x['actuator_velocity'];act=x['act'];active=np.array([[evaluate(L[k,j],V[k,j],act[k,j],node)[1]*act[k,j] for j,node in enumerate(nodes)] for k in range(len(x['time']))]);power=(active*V).sum(1);penetration=np.array([max(0.,-float(x['contact_dist'][k,:int(x['contact_ncon'][k])].min())) for k in range(len(x['time']))]);idx=np.arange(n)*stride
  # Primary work comparison uses identical 25us sampled times for both arms.
  pw=power[idx];work=np.concatenate(([0.],np.cumsum((pw[1:]+pw[:-1])*.5*np.diff(t))));v=dict(foot_Fz=x['foot_force'][idx,:,2],penetration=penetration[idx],COMz=x['com_position_world'][idx,2],COMvz=x['com_velocity_world'][idx,2],COMaz=x['com_acceleration_world'][idx,2],activation=x['act'][idx],active_power=pw,active_work=work,load_ratio=x['load_ratio'][idx],limit_excess=x['limit_excess'][idx]);fields[name]=v
  for j,leg in enumerate(['lf','lm','lh','rf','rm','rh']):axs.flat[j].plot(t*1000,v['foot_Fz'][:,j],label=name);axs.flat[j].set_title(leg+' Fz (native)')
  for j,y,title in [(6,v['COMvz'],'COM vz'),(7,v['COMz']-v['COMz'][0],'COM height change'),(8,work,'Active signed work (common 25us grid)'),(9,v['activation'].max(1),'Max actual activation'),(10,v['COMaz'],'COM az'),(11,v['penetration'],'Max penetration')]:axs.flat[j].plot(t*1000,y,label=name);axs.flat[j].set_title(title)
  mask=x['time']<=t[-1]+1e-14;nativework=float(np.trapezoid(power[mask],x['time'][mask]));summary[name]=dict(common_grid_active_work=float(work[-1]),own_grid_same_window_active_work=nativework,first_failure_ms=float(x['time'][-1]*1000),first_failure_type='LOAD_RATIO_LT_0.8' if not x['six_gates'][-1,3] else 'OTHER',min_COMvz=float(x['com_velocity_world'][:,2].min()),max_limit_excess=float(x['limit_excess'].max()))
  for k in range(n):records.append(dict(arm=name,time_s=float(t[k]),load_ratio=float(v['load_ratio'][k]),COMz=float(v['COMz'][k]),COMvz=float(v['COMvz'][k]),COMaz=float(v['COMaz'][k]),penetration=float(v['penetration'][k]),active_work=float(work[k]),**{leg:float(v['foot_Fz'][k,j]) for j,leg in enumerate(['lf','lm','lh','rf','rm','rh'])}))
 d={key:dict(max_abs=float(abs(fields['S160 12.5us'][key]-fields['S154 25us'][key]).max()),rms=float(np.sqrt(np.mean((fields['S160 12.5us'][key]-fields['S154 25us'][key])**2)))) for key in fields['S154 25us']};odd=np.arange(1,len(b['time']),2);badodd=odd[~b['six_gates'][odd].all(1)];delta=float(b['time'][-1]-a['time'][-1]);result=dict(common_samples=n,common_end_ms=float(t[-1]*1000),arms=summary,whole_curve_difference=d,failure_time_delta_us=delta*1e6,failure_time_delta_in_original_steps=delta/25e-6,halfstep_bad_indices=badodd.tolist(),halfstep_samples_preserved=len(odd),strict_convergence_claim=False)
 with (p/'TIMESTEP_COMPARISON.json').open('x') as f:json.dump(result,f,indent=2)
 with (p/'TIMESTEP_COMMON_GRID.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 for ax in axs.flat:ax.legend();ax.set_xlabel('Physical time (ms), matched 25us samples')
 fig.suptitle('S154 vs S160 RK4 timestep half contrast; not strict convergence');fig.tight_layout();fig.savefig(p/'TIMESTEP_COMPARISON.png',dpi=150);print(json.dumps(result))
if __name__=='__main__':main()
