"""Render saved ANALYSIS.json event/decode/write/activation chain; no dynamics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SOURCE = Path('/home/sensen/Desktop/engine修复验收/s4-command1ms/20261027T222000Z/ANALYSIS.json')
OUTPUT = Path('/home/sensen/Desktop/S4_COMMAND1MS_EARLY_CHAIN.png')
LEGS = ['lf','lm','lh','rf','rm','rh']


def main():
    report = json.loads(SOURCE.read_text())
    fig, axs = plt.subplots(1,2,figsize=(17,8),sharex=True,sharey=True)
    specs = [('first_motor_event_s','Motor event','o','#222222',.24),
             ('first_saved_decode_s','Logical filter sample (new only)','s','#2471a3',.14),
             ('first_effective_motor_readout_s','Actual completed-block decode/readout','D','#148f77',.04),
             ('first_write_boundary_s','Write boundary','v','#d68910',-.06),
             ('first_body_command_sample_s','Actual BODY command sample','>','#884ea0',-.16),
             ('first_activation_above_native_floor_s','Activation > .0001 + 1e-12','*','#c0392b',-.26)]
    for ax, arm in zip(axs,['old5ms','new1ms']):
        for j,leg in enumerate(LEGS):
            r = report['arms'][arm]['perleg'][leg]
            ax.axhspan(j-.36,j+.36,color='#f3f4f5' if j%2==0 else 'white',zorder=0)
            points=[]
            for key,label,marker,color,offset in specs:
                if arm=='old5ms' and key=='first_saved_decode_s':
                    continue
                t=r[key]
                if t is not None:
                    x=t*1000;y=j+offset
                    ax.scatter(x,y,s=95 if marker=='*' else 55,marker=marker,color=color,zorder=4)
                    points.append((x,y))
            if points:
                ax.plot([p[0] for p in points],[p[1] for p in points],color='#aaaaaa',lw=1,zorder=1)
            event=r['first_motor_event_s'];write=r['first_write_boundary_s']
            if event is not None and write is not None:
                ax.text(.3,j-.26,f'event→write {(write-event)*1000:.3g} ms',fontsize=8,color='#555555')
            if arm=='old5ms':
                ax.text(15.8,j+.16,'tick decode\nnot saved',ha='right',va='center',fontsize=8,color='#2471a3')
        ax.set_title('Historical 5ms command' if arm=='old5ms' else 'New 1ms command / 0.5ms neural ticks')
        ax.set_xlim(0,16);ax.set_xticks(range(0,17,2));ax.set_xlabel('Physical / neural time (ms)')
        ax.set_yticks(range(6),[l.upper() for l in LEGS]);ax.grid(axis='x',alpha=.3)
        ax.tick_params(axis='y',labelleft=True)
    axs[0].set_ylim(5.6,-.6)
    handles=[Line2D([],[],marker=marker,linestyle='None',color=color,label=label,markersize=9) for _,label,marker,color,_ in specs]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=10,bbox_to_anchor=(.5,.095))
    fig.suptitle('Saved event → decode → write → native activation timeline / six legs',fontsize=16)
    fig.text(.5,.015,'Logical6.5ms filter sample is produced during consume at7ms, NOT a decoder call at6.5ms.\nOffsets separate coincident markers; BODY follows boundary by12.5µs. Historical subwindow samples unavailable.\nClosed-loop feedback diverges; not pure fixed neural-delay attribution or biological/stable-walking certification.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.18,1,.94]);fig.savefig(OUTPUT,dpi=160);plt.close(fig)
    print(OUTPUT)


if __name__=='__main__':
    main()
