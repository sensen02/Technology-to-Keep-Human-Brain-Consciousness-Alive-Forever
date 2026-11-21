"""Choose by nominal full-rollout gates, then replay and run held-out conditions and ablations."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import numpy as np
from bench import BASE, HERE, PARENT, HANDOFF, sha
from validate import run

VAL=HERE/'validation'


def verify_freeze(start, folder, tags):
    with np.load(start) as a:
        parent={k:a[k] for k in a.files}
    for tag in tags:
        with np.load(folder/(tag+'_block.npz')) as spec, np.load(folder/(tag+'.npz')) as a:
            field=spec['field'].item()
            allowed=np.zeros(parent[field].shape,dtype=bool)
            allowed[spec['index']]=True
            for k in a.files:
                if k==field:
                    np.testing.assert_array_equal(a[k][~allowed],parent[k][~allowed])
                else:
                    np.testing.assert_array_equal(a[k],parent[k])
            parent={k:a[k] for k in a.files}
            assert max(abs(parent['exc_fraction']).max(),abs(parent['inh_fraction']).max())<=.08


def brief(report):
    keys=('first_stance_violation_ms','first_com_drop_10pct_ms','first_nonfoot_ms','post_tail_feet',
          'post_entire_stand','post_tail_stand','post_teacher_ticks','warm_spikes','warm_motor_spikes')
    return {k:report[k] for k in keys}


def read(name):
    return json.loads((VAL/name/'warm_release.json').read_text())


def main():
    stages={'base':BASE,'stage1':HERE/'training/trained.npz','stage2':HERE/'strict/trained.npz',
            'impact':HERE/'transfer/impact1.npz','stage3':HERE/'transfer/trained.npz'}
    for i in (0,1):
        f=HERE/f'sweep{i}/trained.npz'
        if f.exists():
            stages['sweep'+str(i)]=f
    stages['correction']=HERE/'correction/trained.npz'
    for name,path in stages.items():
        if not (VAL/name/'warm_release.json').exists():
            run(name,path,0)
        assert read(name)['checkpoint_sha256']==sha(path)
    def rank(tag):
        a=read(tag)
        return tuple(250.1 if a[k] is None else a[k] for k in ('first_stance_violation_ms','first_nonfoot_ms'))
    name=max(stages,key=rank)
    selected=stages[name]
    shutil.copyfile(selected,HERE/'CANDIDATE.npz')
    ablations={}
    with np.load(selected) as current,np.load(BASE) as old:
        changes={}
        for k in ('exc_fraction','inh_fraction','feco_gain','load_gain','vnc_gain_uS','motor_gain'):
            changes[k]={'n':int(np.count_nonzero(current[k]!=old[k])), 'min':float(current[k].min()), 'max':float(current[k].max())}
        for label,fields in [('reset_existing',['exc_fraction','inh_fraction']),
                             ('reset_interfaces',['feco_gain','load_gain','vnc_gain_uS','motor_gain'])]:
            path=HERE/(label+'.npz')
            np.savez_compressed(path, **{k:old[k] if k in fields else current[k] for k in old.files})
            ablations[label]=path
    tests=[('replay',selected,0.,'warm_release'),('positive',selected,.03,'warm_release'),
           ('negative',selected,-.03,'warm_release'),('cut',selected,0.,'motor_cut')]
    tests.extend((k,v,0.,'warm_release') for k,v in ablations.items())
    with ThreadPoolExecutor(max_workers=3) as pool:
        checked=list(pool.map(lambda x:run(*x),tests))
    with np.load(VAL/name/'warm_release.npz') as original,np.load(VAL/'replay/warm_release.npz') as other:
        for k in original.files:
            np.testing.assert_array_equal(original[k],other[k])
        replay_fields=len(original.files)
    gates={}
    with np.load(VAL/name/'warm_release.npz') as a:
        stand=a['stand'][HANDOFF:]
        bad=np.flatnonzero(~stand)
        if len(bad):
            t=HANDOFF+bad[0]
            gates=dict(first_bad_feet=float(a['loads'][t,:6].sum()),first_bad_nonfoot=float(a['loads'][t,-1]),
                       first_bad_loaded=int((a['loads'][t,:6]>.01).sum()),first_bad_com=float(a['torso'][t,0]/a['torso'][0,0]),
                       first_bad_upright=float(a['torso'][t,1]),first_bad_depth_mm=float(a['depth_mm'][t]))
        gates['first_step_feet']=float(a['loads'][HANDOFF+1,:6].sum())
    verify_freeze(HERE/'strict/start.npz',HERE/'strict',['family','exc_fraction','inh_fraction','legs0','legs1','legs2'])
    verify_freeze(HERE/'strict/trained.npz',HERE/'transfer',['impact0','impact1','family','feco','vnc'])
    verify_freeze(HERE/'sweep1/trained.npz',HERE/'correction',['teacher0','teacher1','teacher2','fine'])
    for i in (0,1):
        folder=HERE/f'sweep{i}'
        if folder.exists():
            start=HERE/('transfer' if i==0 else 'sweep0')/'trained.npz'
            verify_freeze(start,folder,['units0','units1','units2','inh_fraction','exc_fraction','arms0','arms1','arms2'])
    warm_log=[json.loads(s) for s in (HERE/'training/TRAIN.jsonl').read_text().splitlines()]
    warm_evals=sum(x['generations']*10 for x in warm_log if x.get('stage')=='warm_motor')
    physical_evals=sum(len(x['population']) for x in warm_log if 'population' in x)
    for f in HERE.glob('*/*.jsonl'):
        if f.parent.name in ('strict','transfer','sweep0','sweep1','correction'):
            physical_evals+=sum(len(json.loads(line)['scores']) for line in f.read_text().splitlines())
    result=dict(selected=name,checkpoint_sha256=sha(selected),replay_fields=replay_fields,changes=changes,
                warm_cma_candidates=warm_evals,closed_loop_cma_candidates=physical_evals,
                correction_label_candidates=sum(x['evaluations'] for x in json.loads((HERE/'correction/fit.json').read_text())),
                nominal={k:brief(read(k)) for k in stages},checks={x[0]:brief(y) for x,y in zip(tests,checked)},
                heldout_baselines={k:brief(read(k)) for k in ('base_0','base_1')},**gates)
    (HERE/'ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
