"""Evaluate frozen parent and continuation from the unchanged project entry point."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from train import HERE, PRIOR, START, VELOCITIES, HELD, HANDOFF, PARENT, sha, dump
from freeze import verify_proposal

FIELDS=('exc_fraction','inh_fraction','load_gain','feco_gain','vnc_gain_uS','motor_gain')


def run_case(case):
    name, model, velocity, mode=case
    path=HERE/'validation'/name
    subprocess.run([sys.executable,'-B',str(PARENT.parent/'run_stance.py'),str(path),'--mode',mode,
                    '--parameters',str(model),'--initial-vx',str(velocity)],check=True,stdout=subprocess.DEVNULL)
    report=json.loads((path/(mode+'.json')).read_text())
    assert report['checkpoint_sha256']==sha(model) and report['post_teacher_ticks']==0 and report['warm_all_six']
    with np.load(path/(mode+'.npz')) as data:
        assert not np.any(data['teacher_on'][HANDOFF:])
        cmd=data['command'][HANDOFF:]
        if mode=='warm_release':
            np.testing.assert_array_equal(cmd,np.repeat(data['neural_command'][HANDOFF//5-1:-1],5,axis=0))
        else:
            assert not np.any(cmd)
        bad=np.flatnonzero(~data['stand'][HANDOFF:])
        loads=data['loads'][HANDOFF:]
        height=data['torso'][:,0]/data['torso'][0,0]
        report['first_failure_detail']=None
        if len(bad):
            t=int(bad[0])
            feet=loads[t,:6]
            report['first_failure_detail']=dict(feet=feet.tolist(),total=float(feet.sum()),
                loaded=int((feet>.01).sum()),nonfoot=float(loads[t,-1]),
                com_ratio=float(height[HANDOFF+t]),upright=float(data['torso'][HANDOFF+t,1]),
                depth_mm=float(data['depth_mm'][HANDOFF+t]))
        report['first_step_load']=float(loads[1,:6].sum())
    return name,report


def audit(status):
    last=status['parent']
    updates=0
    count=0
    for k in range(status['completed_blocks']):
        logs=json.loads((HERE/f'generation_{k:04d}.json').read_text())
        updates+=len(logs)
        count+=sum(len(row['metrics']) for row in logs)
        with np.load(HERE/f'block_{k:04d}.npz') as block:
            if str(block['anchor'])!=last:
                assert str(block['previous_primary'])==last
                adopted=set()
                for file in (HERE/'explore').glob('block_*.npz'):
                    with np.load(file) as branch:
                        if branch['winner'].size:
                            adopted.add(str(branch['accepted']))
                assert str(block['anchor']) in adopted
                last=str(block['anchor'])
            nxt=str(block['candidate'])
            if nxt!=last:
                field=str(block['field'])
                verify_proposal(last, field, block['ids'], block['groups'], block['selected'], nxt)
                with np.load(HERE/'checkpoints'/(last+'.npz')) as old,np.load(HERE/'checkpoints'/(nxt+'.npz')) as new:
                    mask=np.ones(old[field].shape,bool)
                    mask[block['ids']]=False
                    for key in old.files:
                        if key==field:
                            np.testing.assert_array_equal(old[key][mask],new[key][mask])
                        else:
                            np.testing.assert_array_equal(old[key],new[key])
                    for gate in ('exc_fraction','inh_fraction'):
                        assert np.max(abs(new[gate]))<=.08
            last=nxt
    assert sha(HERE/'CANDIDATE.npz')==last==status['best']
    assert updates==status['updates'] and count==status['candidates'] and count*3==status['episodes']
    assert status['training_seconds']>=json.loads((HERE/'BUDGET.json').read_text())['target_training_s']


def main():
    status=json.loads((HERE/'progress.json').read_text())
    audit(status)
    new=HERE/'SELECTED.npz'
    frozen=json.loads((HERE/'FROZEN.json').read_text())
    assert sha(new)==frozen['sha256']
    checks=[(f'{label}_{i}',model,v,'warm_release') for i,v in enumerate((*VELOCITIES,*HELD))
            for label,model in [('parent',START),('new',new)]]
    checks += [('replay',new,0.,'warm_release'),('cut',new,0.,'motor_cut')]
    with ThreadPoolExecutor(3) as pool:
        results=dict(pool.map(run_case,checks))
    ref=HERE/'validation'
    with np.load(ref/'new_0/warm_release.npz') as main,np.load(ref/'replay/warm_release.npz') as replay:
        for key in main.files:
            np.testing.assert_array_equal(main[key],replay[key])
        arrays=len(main.files)
    for i,v in enumerate(VELOCITIES):
        measured=results[f'new_{i}']['first_stance_violation_ms']
        assert (measured if measured is not None else 250.)==frozen['metrics'][i][0]
    with np.load(START) as a,np.load(new) as b:
        changes={k:int(np.count_nonzero(a[k]!=b[k])) for k in FIELDS}
    budget=json.loads((HERE/'BUDGET.json').read_text())
    dump(HERE/'ACCEPTANCE.json',dict(time_multiplier=status['training_seconds']/budget['previous_total_s'],
         budget=budget,training=status,frozen=frozen,replay_arrays=arrays,changed_from_parent=changes,cases=results))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,layout='constrained')
    for i,ax in enumerate(axes.ravel()):
        for label,title in [('parent','Original'),('new','10x training')]:
            with np.load(ref/f'{label}_{i}'/'warm_release.npz') as d:
                load=d['loads'][HANDOFF:,:6].sum(axis=1)
                line,=ax.plot(np.arange(len(load))*.1,load,label=title,linewidth=1)
                first=results[f'{label}_{i}']['first_stance_violation_ms']
                if first is not None:
                    ax.plot(first,load[round(first/.1)],'o',color=line.get_color(),markersize=4)
        ax.axhspan(.8,1.2,color='gray',alpha=.15)
        ax.set(xlabel='Time after release (ms)',ylabel='Foot load / body weight',
               title=f'{"Development" if i<3 else "Held out"}   |   vx {(*VELOCITIES,*HELD)[i]:+.3f} mm/s')
    axes[0,0].legend()
    fig.suptitle('Foot loading after handoff   •   Dots mark the first failed stance criterion')
    fig.savefig(HERE/'FOOT_LOAD.png',dpi=140)
    for name,m in results.items():
        print(name,m['first_stance_violation_ms'],m['first_nonfoot_ms'],m['post_tail_feet'],flush=True)


if __name__=='__main__':
    main()
