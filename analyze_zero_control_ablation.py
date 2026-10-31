"""S148 common-window signed work and actual force comparison, saved arrays only."""
from pathlib import Path
import json,csv,hashlib,xml.etree.ElementTree as ET
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_muscle_reconstruction import evaluate
P=Path('/home/sensen/Desktop/engine修复验收/s148-zero-control-ablation/20261024T212511892865Z');REF=Path('/home/sensen/Desktop/engine修复验收/s146-secant-pose-free-physical/20261024T203645678947Z');XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=load(P/'init80ms/TRAJECTORY.npz');b=load(REF/'init80ms/TRAJECTORY.npz');n=len(a['time']);t=a['time'];nodes=list(ET.parse(XML).getroot().find('actuator'));summary={};records=[];fig,axs=plt.subplots(2,2,figsize=(12,8))
 for name,x in [('S146 genuine drive',b),('S148 zero ctrl',a)]:
  active=np.zeros((n,90));bias=active.copy()
  for i in range(n):
   for j,node in enumerate(nodes):
    f,g,v=evaluate(x['actuator_length'][i,j],x['actuator_velocity'][i,j],x['act'][i,j],node);assert abs(f-x['actuator_force'][i,j])<=1e-10;active[i,j]=g*x['act'][i,j];bias[i,j]=v
  powers=dict(active=np.sum(active*x['actuator_velocity'][:n],1),muscle_bias=np.sum(bias*x['actuator_velocity'][:n],1),joint_passive=x['signed_power_passive'][:n],contact=x['signed_power_contact_rows'][:n]);work={k:float(np.trapezoid(v,t)) for k,v in powers.items()};cum=np.r_[0,np.cumsum((powers['active'][1:]+powers['active'][:-1])/2*np.diff(t))]
  summary[name]=dict(work_native=work,final_load=float(x['load_ratio'][n-1]),final_COM_vz=float(x['com_velocity_world'][n-1,2]),final_COM_az=float(x['com_acceleration_world'][n-1,2]),max_joint_excess=float(x['limit_excess'][:n].max()))
  for i in range(n):records.append([name,i,float(t[i]),float(x['load_ratio'][i]),float(x['com_acceleration_world'][i,2]),float(x['com_velocity_world'][i,2]),*[float(v[i]) for v in powers.values()]])
  axs[0,0].plot(t*1000,x['load_ratio'][:n],label=name);axs[0,1].plot(t*1000,x['com_velocity_world'][:n,2],label=name);axs[1,0].plot(t*1000,cum,label=name);axs[1,1].plot(t*1000,x['com_acceleration_world'][:n,2],label=name)
 axs[0,0].axhline(.8,color='red');axs[0,0].set_title('Foot load ratio');axs[0,1].set_title('COM vertical velocity (native)');axs[1,0].set_title('Cumulative active muscle signed work (native)');axs[1,1].set_title('COM vertical acceleration (native)')
 for ax in axs.flat:ax.set_xlabel('Actual physical ms');ax.legend()
 fig.suptitle('S148 fixed initial state, zero ctrl ablation: common window only, NOT a controller');fig.tight_layout();fig.savefig(P/'ZERO_CONTROL_VS_S146.png',dpi=150)
 with (P/'COMMON_WINDOW.csv').open('x') as f:
  w=csv.writer(f);w.writerow(['arm','sample','time_s','load_ratio','COM_az','COM_vz','power_active','power_muscle_bias','power_joint_passive','power_contact']);w.writerows(records)
 report=dict(passed=True,common_duration_ms=float(t[-1]*1000),summary=summary,interpretation='Removal of active drive weakens early upward acceleration in this fixed state/window. Difference includes nonlinear contact and pose response; not additive muscle attribution or proof of elastic energy release.',physical_units='NOT_ESTABLISHED',prewarm_success=False,hashes={str(x):sha(x) for x in (Path(__file__),P/'init80ms/TRAJECTORY.npz',REF/'init80ms/TRAJECTORY.npz',XML)})
 with (P/'COMMON_WINDOW.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report))
if __name__=='__main__':main()
