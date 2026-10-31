"""S161 matched dynamic zero drive vs S154; saved arrays only."""
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
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output;a=load(REF);b=load(p);n=min(len(a['time']),len(b['time']));t=a['time'][:n];assert np.allclose(t,b['time'][:n],rtol=0,atol=1e-14);nodes=list(ET.parse('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml').getroot().find('actuator'));fig,axs=plt.subplots(4,3,figsize=(15,14));summary={};derived={};records=[]
 for name,x in [('S154 gain0.5',a),('S161 gain0',b)]:
  active=np.zeros_like(x['act']);bias=np.zeros_like(x['act'])
  for k in range(len(x['time'])):
   for j,node in enumerate(nodes):force,g,bf=evaluate(x['actuator_length'][k,j],x['actuator_velocity'][k,j],x['act'][k,j],node);active[k,j]=g*x['act'][k,j];bias[k,j]=bf;assert abs(force-x['actuator_force'][k,j])<1e-10
  derived[name+'_active_force']=active;derived[name+'_bias_force']=bias
  aw=(active*x['actuator_velocity']).sum(1);bw=(bias*x['actuator_velocity']).sum(1)
  def cumulative(y):return np.concatenate(([0.],np.cumsum((y[1:]+y[:-1])*.5*np.diff(x['time'][:n]))))
  ac=cumulative(aw[:n]);bc=cumulative(bw[:n]);penetration=np.array([max(0.,-float(x['contact_dist'][k,:int(x['contact_ncon'][k])].min())) for k in range(n)]);loadratio=x['foot_force'][:n,:,2].sum(1)/abs(x['gravity_force_world'][:n,2]);peak=int(loadratio.argmax())
  for j,leg in enumerate(['lf','lm','lh','rf','rm','rh']):axs.flat[j].plot(t*1000,x['foot_force'][:n,j,2],label=name);axs.flat[j].set_title(leg+' Fz (native)')
  for j,y,title in [(6,x['com_velocity_world'][:n,2],'COM vz'),(7,x['com_position_world'][:n,2]-x['com_position_world'][0,2],'COM height change'),(8,ac,'Active muscle signed work'),(9,bc,'Bias muscle signed work (not all passive forces)'),(10,x['com_acceleration_world'][:n,2],'COM az'),(11,loadratio,'Total foot load / weight')]:axs.flat[j].plot(t*1000,y,label=name);axs.flat[j].set_title(title)
  summary[name]=dict(first_failure_ms=float(x['time'][-1]*1000),common_active_work=float(ac[-1]),common_bias_work=float(bc[-1]),common_load_peak=float(loadratio[peak]),common_load_peak_ms=float(t[peak]*1000),common_final_load=float(loadratio[-1]),common_COMvz_minmax=[float(x['com_velocity_world'][:n,2].min()),float(x['com_velocity_world'][:n,2].max())],common_COMz_change=float(x['com_position_world'][n-1,2]-x['com_position_world'][0,2]),common_penetration_start_end=[float(penetration[0]),float(penetration[-1])],max_abs_bias_force=float(abs(bias).max()),max_abs_active_force=float(abs(active).max()))
  for k in range(n):records.append(dict(arm=name,time_s=float(t[k]),load_ratio=float(loadratio[k]),COMvz=float(x['com_velocity_world'][k,2]),COMz=float(x['com_position_world'][k,2]),COMaz=float(x['com_acceleration_world'][k,2]),penetration=float(penetration[k]),active_work=float(ac[k]),bias_work=float(bc[k]),**{leg:float(x['foot_force'][k,j,2]) for j,leg in enumerate(['lf','lm','lh','rf','rm','rh'])}))
 with (p/'ACTIVE_BIAS_FORCE_SPLIT.npz').open('xb') as f:np.savez_compressed(f,**derived)
 with (p/'ZERO_COMPARISON.json').open('x') as f:json.dump(dict(common_samples=n,common_duration_ms=float(t[-1]*1000),arms=summary,interpretation='Difference is total matched intervention response; muscle work is internal, not COM external force or metabolic cost'),f,indent=2)
 with (p/'ZERO_COMMON_TIMESERIES.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 for ax in axs.flat:ax.legend();ax.set_xlabel('Release aligned physical time (ms), common window')
 fig.suptitle('S161 zero interface matched control; zero ctrl does NOT remove bias muscle force/contact dynamics');fig.tight_layout();fig.savefig(p/'ZERO_DRIVE_COMPARISON.png',dpi=150);print(json.dumps(summary))
if __name__=='__main__':main()
