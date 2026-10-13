from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'outputs/workspace_recheck'
d=json.loads((OUT/'pose_command_cross.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5))
for tr in d['trajectories']:
    r=tr['samples']; time=[v['time_s']*1000 for v in r]
    for ax,key,idx in [(axes[0],'root_z_mm',None),(axes[1],'COM_position_mm',2)]:
        z=[v[key] if idx is None else v[key][idx] for v in r]
        ax.plot(time,[z[0]-a for a in z],label=tr['label'])
for ax,title in zip(axes,['Root height drop','Whole-body COM height drop']):
    ax.set(title=title,xlabel='Time (ms)',ylabel='Drop from own initial height (mm)');ax.legend();ax.grid(alpha=.2)
fig.suptitle('2 poses x 2 constant commands: same native physics, no reoptimization')
fig.text(.5,.015,'P_A: original pose; P_B: root lower by 0.01mm. C_A/C_B: saved commands, including initial activation. Research only.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.93));p=OUT/'pose_command_cross.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)
