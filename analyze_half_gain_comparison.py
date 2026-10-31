"""S149 saved-array two-window comparison; no simulation."""
from pathlib import Path
import argparse,json,csv
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_muscle_reconstruction import evaluate
R=Path('/home/sensen/Desktop/engine修复验收');REF=R/'s146-secant-pose-free-physical/20261024T203645678947Z';ZERO=R/'s148-zero-control-ablation/20261024T212511892865Z'
def load(p):
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;data={'S146 gain1':load(REF),'S148 gain0':load(ZERO),'S149 gain0.5':load(p)};nodes=list(ET.parse('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml').getroot().find('actuator'));summary={};records=[]
 for window,n in [('early_common',min(len(x['time']) for x in data.values())),('full_common',min(len(data[k]['time']) for k in ('S146 gain1','S149 gain0.5')))]:
  fig,axs=plt.subplots(2,2,figsize=(12,8));summary[window]={}
  for name,a in data.items():
   if window=='full_common' and name=='S148 gain0':continue
   t=a['time'][:n];active=np.zeros((n,90));bias=active.copy()
   for i in range(n):
    for j,node in enumerate(nodes):
     f,g,b=evaluate(a['actuator_length'][i,j],a['actuator_velocity'][i,j],a['act'][i,j],node);assert abs(f-a['actuator_force'][i,j])<1e-10;active[i,j]=g*a['act'][i,j];bias[i,j]=b
   powers=dict(active=np.sum(active*a['actuator_velocity'][:n],1),muscle_bias=np.sum(bias*a['actuator_velocity'][:n],1),contact=a['signed_power_contact_rows'][:n],joint_passive=a['signed_power_passive'][:n]);cum=np.r_[0,np.cumsum((powers['active'][1:]+powers['active'][:-1])/2*np.diff(t))]
   summary[window][name]=dict(duration_ms=float(t[-1]*1000),load_final=float(a['load_ratio'][n-1]),COM_vz_final=float(a['com_velocity_world'][n-1,2]),COM_vz_max=float(a['com_velocity_world'][:n,2].max()),COM_vz_min=float(a['com_velocity_world'][:n,2].min()),COM_az_final=float(a['com_acceleration_world'][n-1,2]),max_limit_excess=float(a['limit_excess'][:n].max()),root_z_change=float(a['qpos'][n-1,2]-a['qpos'][0,2]),COM_z_change=float(a['com_position_world'][n-1,2]-a['com_position_world'][0,2]),work_native={k:float(np.trapezoid(v,t)) for k,v in powers.items()})
   for i in range(n):records.append([window,name,i,float(t[i]),float(a['load_ratio'][i]),float(a['com_velocity_world'][i,2]),float(a['com_acceleration_world'][i,2]),*[float(v[i]) for v in powers.values()]])
   axs[0,0].plot(t*1000,a['load_ratio'][:n],label=name);axs[0,1].plot(t*1000,a['com_velocity_world'][:n,2],label=name);axs[1,0].plot(t*1000,cum,label=name);axs[1,1].plot(t*1000,a['com_acceleration_world'][:n,2],label=name)
  axs[0,0].axhline(.8,color='red')
  for ax,title in zip(axs.flat,['Foot load ratio','COM vertical velocity (native)','Active muscle signed work (native)','COM vertical acceleration (native)']):ax.set_title(title);ax.set_xlabel('Actual physical ms');ax.legend()
  fig.suptitle('S149 fixed half gain: '+window+'; diagnostic only');fig.tight_layout();fig.savefig(p/(window.upper()+'.png'),dpi=150);plt.close(fig)
 with (p/'COMPARISON.csv').open('x') as f:
  w=csv.writer(f);w.writerow(['window','arm','sample','time_s','load_ratio','COM_vz','COM_az','power_active','power_muscle_bias','power_contact','power_joint_passive']);w.writerows(records)
 with (p/'COMPARISON.json').open('x') as f:json.dump(dict(passed=True,windows=summary,prewarm_success=False,physical_units='NOT_ESTABLISHED'),f,indent=2)
 print(json.dumps(summary))
if __name__=='__main__':main()
