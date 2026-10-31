"""S155 separate pairwise common-window analyses, no physical calls."""
from pathlib import Path
import argparse,types,sys,json
ROOT=Path(__file__).resolve().parent
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 for ref in ['S149','S154']:
  # Build actual underlying comparison text in-memory, preserving historical sources.
  base=(ROOT/'analyze_lf_channel_gain.py').read_text();base=base.replace('S151 LF4=0.25','S155 tau_act=2ms').replace('S151 LF channel4 intervention:','S155 activation dose vs '+ref+':').replace('LF_COMPARISON',ref+'_DOSE_COMPARISON').replace("window.upper()+'.png'","'"+ref+"_'+window.upper()+'.png'")
  if ref=='S154':base=base.replace('s149-half-gain-physical/20261024T213108429114Z','s154-activation-rate/20261024T215751775489Z').replace('S149 all0.5','S154 tau_act=1ms')
  base=base.replace('plt.subplots(3,3,figsize=(15,11))','plt.subplots(4,3,figsize=(15,14))');old="   axs.flat[6].plot(t*1000,a['com_velocity_world'][:n,2],label=name)";assert old in base
  base=base.replace(old,"   axs.flat[9].plot(t*1000,a['act'][:n].max(1),label=name);axs.flat[9].set_title('Actual maximum activation')\n   axs.flat[10].plot(t*1000,a['com_acceleration_world'][:n,2],label=name);axs.flat[10].set_title('COM vertical acceleration (native)')\n   axs.flat[11].plot(t*1000,a['com_position_world'][:n,2]-a['com_position_world'][0,2],label=name);axs.flat[11].set_title('COM vertical displacement (native)')\n"+old)
  base=base.replace('duration_ms=float(t[-1]*1000),legs=perleg',"duration_ms=float(t[-1]*1000),actual_activation_max=float(a['act'][:n].max()),actual_activation_at_25us=float(a['act'][1].max()),legs=perleg")
  mod=types.ModuleType('_s155_compare_'+ref);mod.__file__=str(Path(__file__));exec(compile(base,str(Path(__file__)),'exec'),mod.__dict__);oldargs=sys.argv
  try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
  finally:sys.argv=oldargs
if __name__=='__main__':main()
