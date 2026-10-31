"""S159 release-aligned comparison and application-alignment availability."""
from pathlib import Path
import argparse,types,sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 base=(ROOT/'analyze_lf_channel_gain.py').read_text();base=base.replace('s149-half-gain-physical/20261024T213108429114Z','s154-activation-rate/20261024T215751775489Z').replace('S149 all0.5','S154 immediate').replace('S151 LF4=0.25','S159 delayed').replace('S151 LF channel4 intervention:','S159 release aligned timing:').replace('LF_COMPARISON','TIMING_COMPARISON')
 base=base.replace('plt.subplots(3,3,figsize=(15,11))','plt.subplots(4,3,figsize=(15,14))');old="   axs.flat[6].plot(t*1000,a['com_velocity_world'][:n,2],label=name)";assert old in base
 base=base.replace(old,"   axs.flat[9].plot(t*1000,a['act'][:n].max(1),label=name);axs.flat[9].set_title('Actual maximum activation')\n   axs.flat[10].plot(t*1000,a['ctrl'][:n].max(1),label=name);axs.flat[10].set_title('Actual maximum ctrl (release aligned)')\n   axs.flat[11].plot(t*1000,a['com_position_world'][:n,2]-a['com_position_world'][0,2],label=name);axs.flat[11].set_title('COM vertical displacement (native)')\n"+old)
 mod=types.ModuleType('_s159_compare');mod.__file__=str(Path(__file__));exec(compile(base,str(Path(__file__)),'exec'),mod.__dict__);oldargs=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=oldargs
 import matplotlib.pyplot as plt
 with np.load(p/'init80ms/TRAJECTORY.npz') as a:
  n=len(a['decoded']);physical=np.arange(n)*.5;available=(np.arange(n)+1)*.5;fig,ax=plt.subplots(figsize=(10,4));ax.scatter(available,np.arange(n),label='Command available/planned application (ms)');ax.hlines(np.arange(n),physical,available,label='Source tick (physical offset)');ax.axvline(a['time'][-1]*1000,color='r',label='Actual first-failure stop');ax.axvspan(0,min(.5,float(a['time'][-1]*1000)),alpha=.2,color='gray',label='Actual ctrl=0 initial interval');nonzero=bool(a['ctrl'].any());ax.set_title('S159 command timing: '+('post-application data available' if nonzero else 'No nonzero command applied; post-application alignment unavailable'));ax.set_xlabel('Physical time (ms)');ax.set_ylabel('Release neural tick');ax.legend();fig.tight_layout();fig.savefig(p/'COMMAND_APPLICATION_TIMING.png',dpi=150)
  result=dict(post_application_alignment_available=nonzero,reason='First failure before first planned nonzero application' if not nonzero else 'See saved ctrl onset',first_planned_application_s=.0005,actual_stop_s=float(a['time'][-1]),actual_max_activation=float(a['act'].max()))
  (p/'POST_APPLICATION_ALIGNMENT.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
