"""Plot dynamic audit samples, preserve physical units and acceleration/velocity distinction."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'outputs/workspace_recheck'

def main():
    d=json.loads((OUT/'dynamic_support_audit.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for tr in d['trajectories']:
        rows=tr['samples']; t=[1000*r['time_s'] for r in rows]
        label=tr['label']
        axes[0,0].plot(t,[r['support_over_weight'] for r in rows],label=label)
        axes[0,1].plot(t,[r['COM_velocity_mm_s'][2] for r in rows],label=label)
        axes[1,0].plot(t,[r['COM_acceleration_mm_s2'][2] for r in rows],label=label)
        z=[r['root_z_mm'] for r in rows]
        axes[1,1].plot(t,[v-z[0] for v in z],label=label)
    axes[0,0].axhline(1,color='black',ls='--',lw=1)
    axes[0,1].axhline(0,color='black',ls='--',lw=1)
    axes[1,0].axhline(0,color='black',ls='--',lw=1)
    for ax,title,ylabel in zip(axes.flat,['Ground support','Whole-body COM velocity','Whole-body COM acceleration','Root height loss'],['Vertical ground force / weight','Vertical velocity (mm/s)','Vertical acceleration (mm/s²)','Height change from initial (mm)']):
        ax.set(title=title,xlabel='Time (ms)',ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Dynamic diagnostics: unchanged constant commands and native physical model')
    fig.text(.5,.015,'Positive acceleration can brake an ongoing fall; it does not imply upward velocity or stable standing.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.95))
    p=OUT/'dynamic_support_audit.png'; fig.savefig(p,dpi=140,bbox_inches='tight'); plt.close(fig); print(p)
if __name__=='__main__':main()
