"""S157 saved-array S154 comparison and requested fixed-time midleg tests."""
from pathlib import Path
import argparse,types,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
def load(p):
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 src=(ROOT/'analyze_activation_rate_probe.py').read_text();base=(ROOT/'analyze_lf_channel_gain.py').read_text();base=base.replace('s149-half-gain-physical/20261024T213108429114Z','s154-activation-rate/20261024T215751775489Z').replace('S149 all0.5','S154 front0.5').replace('S151 LF4=0.25','S157 front0.45').replace('S151 LF channel4 intervention:','S157 paired foreleg gain:').replace('LF_COMPARISON','FORELEG_COMPARISON')
 base=base.replace('plt.subplots(3,3,figsize=(15,11))','plt.subplots(4,3,figsize=(15,14))');old="   axs.flat[6].plot(t*1000,a['com_velocity_world'][:n,2],label=name)";assert old in base
 base=base.replace(old,"   axs.flat[9].plot(t*1000,a['act'][:n].max(1),label=name);axs.flat[9].set_title('Actual maximum activation')\n   axs.flat[10].plot(t*1000,a['com_acceleration_world'][:n,2],label=name);axs.flat[10].set_title('COM vertical acceleration (native)')\n   axs.flat[11].plot(t*1000,a['com_position_world'][:n,2]-a['com_position_world'][0,2],label=name);axs.flat[11].set_title('COM vertical displacement (native)')\n"+old)
 mod=types.ModuleType('_s157_compare');mod.__file__=str(Path(__file__));exec(compile(base,str(Path(__file__)),'exec'),mod.__dict__);oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=oldargs
 a=load(REF);b=load(p);n=min(len(a['time']),len(b['time']));points={}
 for k in (12,60):
  if k<n:points[str(k)]=dict(time_ms=float(a['time'][k]*1000),reference_six_Fz=a['foot_force'][k,:,2].tolist(),new_six_Fz=b['foot_force'][k,:,2].tolist(),delta_six_Fz=(b['foot_force'][k,:,2]-a['foot_force'][k,:,2]).tolist())
 delta=b['foot_force'][:n,:,2]-a['foot_force'][:n,:,2]
 result=dict(points=points,common_duration_ms=float(a['time'][n-1]*1000),delta_LM_minmax=[float(delta[:,1].min()),float(delta[:,1].max())],delta_RM_minmax=[float(delta[:,4].min()),float(delta[:,4].max())],both_midlegs_nondecreased_every_sample=bool(np.all(delta[:,[1,4]]>=-1e-10)),new_actual_activation_max=float(b['act'].max()),new_first_fail_ms=float(b['time'][-1]*1000),new_min_COM_vz=float(b['com_velocity_world'][:,2].min()),new_COM_height_change=float(b['com_position_world'][-1,2]-b['com_position_world'][0,2]),new_final_load=float(b['load_ratio'][-1]))
 with (p/'FORELEG_PREREGISTERED_POINTS.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
