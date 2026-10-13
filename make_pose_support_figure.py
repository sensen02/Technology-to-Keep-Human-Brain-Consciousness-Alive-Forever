"""Render pose-search evidence from saved artifact, not hard-coded outcome values."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'outputs/workspace_recheck'

def main():
    d=json.loads((OUT/'pose_support_search.json').read_text())
    rows=[r for r in d['rows'] if 'objective' in r]
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    rows=sorted(rows,key=lambda r:r['objective'])
    axes[0].barh([r['label'] for r in rows],[r['objective'] for r in rows],color='steelblue')
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Same dimensionless squared residual objective (lower better)')
    axes[0].set_title('Feasible bounded candidates; not a global search')
    axes[0].tick_params(axis='y',labelsize=8)
    for item in d['integrations']:
        tr=item['result']['trace']
        z=tr['root_height_mm']
        axes[1].plot([1000*t for t in tr['time_s']],[v-z[0] for v in z],label=item['label'])
    axes[1].set(xlabel='Time (ms)',ylabel='Root height change from initial (mm)',title='Independent constant-command integration')
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=.2)
    fig.suptitle('Bounded pose/support exploration: native model parameters unchanged')
    fig.text(.5,.01,'L=1 mm engineering scale; COM weight=1. Offline mechanics, not neural standing or infeasibility proof.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    p=OUT/'pose_support_search.png'; fig.savefig(p,dpi=140,bbox_inches='tight'); plt.close(fig); print(p)

if __name__=='__main__':main()
