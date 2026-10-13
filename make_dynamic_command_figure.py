"""Render compact plotting data derived from saved dynamic command experiment."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent/'outputs/workspace_recheck'

def render(d):
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    c=d['candidates']; axes[0].barh([r['label'] for r in c],[r['score'] for r in c],color=['steelblue' if r['eligible'] else 'gray' for r in c]);axes[0].invert_yaxis()
    axes[0].set(title='50ms bounded candidates',xlabel='Declared dimensionless dynamic score (lower better)')
    axes[0].tick_params(axis='y',labelsize=8)
    for r in d['validation']:
        z=r['COM_z']; line,=axes[1].plot([1000*t for t in r['time_s']],[z[0]-v for v in z],label=r['label'])
        if r['first_body_contact_ms'] is not None:
            axes[1].axvline(r['first_body_contact_ms'],color=line.get_color(),ls=':',label=f"body contact {r['first_body_contact_ms']:.1f}ms")
    axes[1].axvline(50,color='gray',ls='--',label='selection window ends')
    axes[1].set(title='Independent 100ms constant-command replay',xlabel='Time (ms)',ylabel='Whole-body COM height drop (mm)');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('Dynamic evaluation on fixed pose and frozen native physics')
    fig.text(.5,.01,'Engineering score and finite candidates only. Offline muscle commands are not neural outputs or standing proof.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));p=OUT/'dynamic_command_search.png';fig.savefig(p,dpi=140,bbox_inches='tight');plt.close(fig);print(p)

if __name__=='__main__':
    d=json.loads((OUT/'dynamic_command_search.json').read_text())
    compact=dict(candidates=[dict(label=r['label'],score=r['metrics']['score'],eligible=r['eligible']) for r in d['candidates']],validation=[dict(label=r['role']+': '+r['candidate_label'],time_s=r['trace']['time_s'],COM_z=[v[2] for v in r['trace']['COM_position_mm']],first_body_contact_ms=next((1000*t for t,n in zip(r['native_trace']['time_s'],r['native_trace']['nonfoot_floor_contacts']) if n),None)) for r in d['validation100ms']])
    render(compact)
