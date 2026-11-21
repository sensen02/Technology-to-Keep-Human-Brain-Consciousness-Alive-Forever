"""Independently summarize the open-field control matrix and draw evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


CASES = ('FORMAL_L','FORMAL_R','CUT_L','VISION_L','ODOR_L','BLANK_L')
COLORS = ('#27846e','#3763ad','#b05536','#8c67a2','#bc852e','#6d6d74')
LABELS = ('Both cues, left','Both cues, right','BANC DN cut','Vision only','Odor only','Both cues off')


def read(path):
    return json.loads(path.read_text())


def run(root):
    data = {}
    for case in CASES:
        folder = root/case
        meta = read(folder/'RESULT.json')
        with np.load(folder/'TRACE.npz',allow_pickle=False) as bundle:
            n=meta['qpos_columns']
            motion=bundle['trace'][:,n:]
            expected=np.cumsum(motion[:,7])
            assert np.allclose(expected,motion[:,8],rtol=0.,atol=1e-12)
            assert np.all(motion[motion[:,7]>0.,6] <= .6)
            assert expected[-1] == meta['ingested']
            assert int(bundle['arrived'].sum()) == meta['actual_neck_arrival']
            assert int(bundle['emitted'].sum()) == meta['eon_source']
            if case == 'CUT_L':
                assert not bundle['arrived'].any() and bundle['emitted'].any()
            data[case]=dict(meta=meta,motion=motion.copy(),arrived=bundle['arrived'].copy(),
                            eyes=bundle['eyes'].copy())
    fit=read(root/'SENSOR_FIT/FIT.json')
    passed=(
        all(data[key]['meta']['navigation_accepted'] and data[key]['meta']['actual_neck_arrival']>0
            for key in CASES[:2])
        and all(not data[key]['meta']['ingested'] for key in ('CUT_L','BLANK_L'))
    )

    fig,axes=plt.subplots(2,2,figsize=(11.8,8.8),layout='constrained')
    for case,color,label in zip(CASES,COLORS,LABELS,strict=True):
        rec=data[case]
        a=rec['motion']
        axes[0,0].plot(a[:,1],a[:,2],label=label,color=color,lw=1.65)
        axes[0,1].plot(a[:,0],a[:,6],color=color,lw=1.5)
        axes[1,0].plot(a[:,0],a[:,8],color=color,lw=1.5)
        time=np.arange(len(rec['arrived']))*.0001
        axes[1,1].plot(time,rec['arrived'].sum(axis=1).cumsum(),color=color,lw=1.5)
    for case,color in zip(CASES[:2],COLORS[:2],strict=True):
        x,y,_=data[case]['meta']['fruit']
        axes[0,0].add_patch(Circle((x,y),.6,facecolor=color,alpha=.2,edgecolor=color))
    axes[0,0].scatter([0.],[0.],color='#111111',s=23,zorder=5)
    axes[0,0].set(xlabel='World x (mm)',ylabel='World y (mm)',title='Actual six-leg trajectories')
    axes[0,0].set_aspect('equal')
    axes[0,0].legend(frameon=False,fontsize=8,loc='upper left')
    axes[0,1].axhline(.6,color='#777777',ls='--',lw=1.0)
    axes[0,1].set(xlabel='Time (s)',ylabel='Mouth to fruit (mm)',title='Local contact required for intake')
    axes[1,0].set(xlabel='Time (s)',ylabel='Cumulative intake (units)',title='Measured food transfer')
    axes[1,1].set(xlabel='Time (s)',ylabel='DN arrivals',title='Actual BANC delivery')
    for a in axes.ravel():
        a.grid(alpha=.2)
        a.spines[['top','right']].set_visible(False)
    fig.savefig(root/'OPEN_FRUIT.png',dpi=151)
    plt.close(fig)

    eyes=data['FORMAL_L']['eyes']
    fig,pairs=plt.subplots(1,4,figsize=(10.8,2.5),layout='constrained')
    for i,ax in enumerate(pairs):
        if i<len(eyes):
            ax.imshow(np.hstack(eyes[i]))
            ax.set_title(f'{i*.2:.1f} s')
        ax.axis('off')
    fig.savefig(root/'FRUIT_EYES.png',dpi=150)

    fields=('scene','mode','first_intake_s','ingested','min_oral_distance','eon_source',
            'actual_neck_arrival','ascending_arrival','min_upright','navigation_accepted')
    results={case:{key:entry['meta'][key] for key in fields} for case,entry in data.items()}
    result=dict(status='OPEN_FRUIT_INTAKE_PASS' if passed else 'OPEN_FRUIT_INTAKE_NOT_QUALIFIED',
        cases=results, sensor_fit=fit['metrics'], wall_validated=False)
    with (root/'ACCEPTANCE.json').open('x') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    run(p.parse_args().run)
