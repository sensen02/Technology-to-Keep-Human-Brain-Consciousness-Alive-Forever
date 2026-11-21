"""Plot development learning curves and the actual safe-side trajectory margins."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'20261119T084744Z-closed-loop-forage'


def slots(folder):
    return [json.loads(line) for line in (HERE/folder/'UPDATES.jsonl').read_text().splitlines()]


def trace(directory):
    with np.load(directory/'BODY.npz',allow_pickle=False) as data:
        return data['trace']


def main():
    selected=json.loads((HERE/'SELECTION.json').read_text())
    physical=json.loads((HERE/'PHYSICAL_AUDIT.json').read_text())
    fig,ax=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    top=ax[0,0]
    for label,tag,color in (('TRAIN','Unsafe advance cost','#175966'),('GATED','Food credit gated','#8a4859')):
        data=slots(label)
        top.plot([a['step']/1000. for a in data],
                 [a['stats']['wide_bridge']['success'] for a in data],lw=1.4,label=tag,color=color)
    top.axvline(selected['checkpoint']['step']/1000.,c='#545454',ls=':',lw=1.)
    top.set(xlabel='SAC decisions (thousands)',ylabel='Safe meals / 24',
            title='Wide-bridge development performance',ylim=(0,24))
    top.legend(loc='upper right',frameon=False,fontsize=8)
    rewards=json.loads((HERE/'REWARD_REPLAY.json').read_text())['aggregate']
    panel=ax[0,1]
    panel.bar([0,1],[rewards['old_mean'],rewards['new_mean']],color=['#b87b5d','#4d8c93'])
    panel.axhline(0.,c='#666666',lw=.8)
    panel.set(xticks=[0,1],xticklabels=['Old','Repaired'],ylabel='Mean shaping reward',
              title='11 unsafe approach intervals in archived runs')
    comparisons={'REPEAT_L':'WIDE_L','REPEAT_R':'WIDE_R'}
    for name,graph in zip(('REPEAT_L','REPEAT_R','HOLD_L','HOLD_R'),ax[1:].ravel(),strict=True):
        if name not in physical['cases']:
            graph.axis('off')
            continue
        row=physical['cases'][name]
        t=row['geometry']
        side=t['side']
        # Plot both mirrors with y reflected into safe-side coordinates.
        graph.add_patch(Rectangle((10*t['near'],-14.),10*(t['far']-t['near']),
                                  14.+10*t['edge'],facecolor='#6380b5',alpha=.35))
        graph.hlines(10*(t['edge']+.112),10*t['near'],10*t['far'],
                     color='#435779',ls=':',label='Target foot margin')
        graph.add_patch(Circle((10*t['goal_x'],10*side*t['goal_y']),4.9,
                                color='#ab5a56',alpha=.22))
        if name in comparisons:
            z=trace(PRIOR/comparisons[name])
            graph.plot(10*z[:,1],10*side*z[:,2],c='#a15a43',ls='--',label='Previous')
        z=trace(HERE/name)
        graph.plot(10*z[:,1],10*side*z[:,2],c='#175966',label='Repaired reward')
        graph.scatter([10*z[-1,1]],[10*side*z[-1,2]],c='#175966',s=12)
        graph.set(xlim=(-.8,24),ylim=(-8,9),xlabel='Forward (mm)',ylabel='Toward safe side (mm)',
                  title=f'{name}    intake {row["intake"]:.3f}')
        graph.legend(loc='upper left',frameon=False,fontsize=8)
    fig.savefig(HERE/'EVIDENCE.png',dpi=148,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
