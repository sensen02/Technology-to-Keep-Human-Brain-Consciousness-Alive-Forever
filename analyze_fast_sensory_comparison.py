"""S162 release-aligned saved trajectory comparison, no engine execution."""
from pathlib import Path
import argparse,types,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 base=(ROOT/'analyze_lf_channel_gain.py').read_text();base=base.replace('s149-half-gain-physical/20261024T213108429114Z','s154-activation-rate/20261024T215751775489Z').replace('S149 all0.5','S154 sensory5ms').replace('S151 LF4=0.25','S162 sensory0.5ms').replace('S151 LF channel4 intervention:','S162 sensory refresh:').replace('LF_COMPARISON','SENSORY_COMPARISON')
 base=base.replace('plt.subplots(3,3,figsize=(15,11))','plt.subplots(4,3,figsize=(15,14))');old="   axs.flat[6].plot(t*1000,a['com_velocity_world'][:n,2],label=name)";assert old in base
 base=base.replace(old,"   axs.flat[9].plot(t*1000,a['act'][:n].max(1),label=name);axs.flat[9].set_title('Actual maximum activation')\n   axs.flat[10].plot(t*1000,a['com_position_world'][:n,2]-a['com_position_world'][0,2],label=name);axs.flat[10].set_title('COM vertical displacement')\n   axs.flat[11].plot(np.arange(len(a['decoded']))*.5,a['decoded'].max(1),label=name);axs.flat[11].set_title('Max decoded per neural tick')\n"+old)
 mod=types.ModuleType('_s162_compare');mod.__file__=str(Path(__file__));exec(compile(base,str(Path(__file__)),'exec'),mod.__dict__);oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=oldargs
 import matplotlib.pyplot as plt
 with np.load(p/'init80ms/TRAJECTORY.npz') as z:a={k:z[k].copy() for k in z.files}
 with np.load(REF/'init80ms/TRAJECTORY.npz') as z:b={k:z[k].copy() for k in z.files}
 n=min(len(a['time']),len(b['time']));delta=a['foot_force'][:n,:,2]-b['foot_force'][:n,:,2];fig,axs=plt.subplots(3,1,figsize=(10,9));common=min(len(a['decoded']),len(b['decoded']));ticks=np.arange(common)*.5
 axs[0].plot(ticks,np.max(abs(a['current'][:common]-b['current'][:common]),axis=1),'o-');axs[0].set_title('Fresh encoded neural current max difference vs S154')
 axs[1].plot(ticks,np.count_nonzero(a['spikes'][:common]!=b['spikes'][:common],axis=1),'o-');axs[1].set_title('Number of differing spikes per tick')
 axs[2].plot(ticks,np.max(abs(a['decoded'][:common]-b['decoded'][:common]),axis=1),'o-');axs[2].set_title('Decoded command max difference')
 for ax in axs:ax.set_xlabel('Physical observation time (ms)');ax.grid()
 fig.tight_layout();fig.savefig(p/'SENSORY_NEURAL_CHAIN.png',dpi=150)
 result=dict(common_end_ms=float(a['time'][n-1]*1000),LM_delta_minmax=[float(delta[:,1].min()),float(delta[:,1].max())],RM_delta_minmax=[float(delta[:,4].min()),float(delta[:,4].max())],both_midlegs_nondecreased_all_common_samples=bool(np.all(delta[:,[1,4]]>=-1e-10)),new_firstfail_ms=float(a['time'][-1]*1000),new_min_COMvz=float(a['com_velocity_world'][:,2].min()),new_COMdz=float(a['com_position_world'][-1,2]-a['com_position_world'][0,2]))
 (p/'SENSORY_PREREGISTERED_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
