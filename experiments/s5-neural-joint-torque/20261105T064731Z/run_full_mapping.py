"""Full model-supported fixed sign readout, isolated from archived dominant mapping."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,importlib.util,json,time
from pathlib import Path
import numpy as np
PREV=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py')
spec=importlib.util.spec_from_file_location('previous_pilot',PREV);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T064731Z')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T062800Z')
base.O=O
js=base.js;dump=base.dump;sha=base.sha

def sources():return base.sources()+[O/'FULL_MAPPING.npz',O/'FULL_COVERAGE.json',OLD/'SUMMARY.json',Path(__file__)]
def prereg():
    assert (O/'MAPPING_AUDIT.json').exists() and not (O/'PREREGISTRATION.json').exists()
    cfg=js(OLD/'PREREGISTRATION.json')
    cfg.update(mapping='fixed sign of summed muscle activation torque increments for each existing unique pool at keyframe; per-row relative tolerance 1e-8, floor 1e-12; 107 restored coefficients, no existing sign flips',
        sign='full fixed model-supported sign matrix; 42 covered, 35 bidirectional, 7 one-direction; 9 entries change sign in sampled warmup poses, fixed-keyframe approximation disclosed',
        budget=dict(total_cap=200,pilot_training_per_arm=17,total_this_round=40,description='P0/P2 each: exact dominant baseline replay + full-map baseline and 16 search proposals + best repeat + readout cut. Same seed, sigma and parameter bounds as prior pilot.'),
        comparison='dominant zero-vector replay must reproduce old J and qpos exactly before full-map simulation. Main causal comparison is zero-vector full vs dominant. Training groups same budget; no cross-plant score claims.',
        physical_model='unchanged assist-matched joint torque plant; no torque limit change',
        importance='核心必需')
    dump(O/'PREREGISTRATION.json',cfg)
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in sources()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(OLD/'COVERAGE.json'),old_sha256=sha(OLD/'COVERAGE.json'),new_file=str(O/'FULL_COVERAGE.json'),new_sha256=sha(O/'FULL_COVERAGE.json'),reason='Restore non-dominant joint effects already present in muscle model via unchanged pool identities; no fabricated synapses or added neurons. Full mapping summed within shared pools before taking sign.',old_registration=str(OLD/'PREREGISTRATION.json'),old_registration_sha256=sha(OLD/'PREREGISTRATION.json'),new_registration=str(O/'PREREGISTRATION.json'),new_registration_sha256=sha(O/'PREREGISTRATION.json')))
    print('PREREGISTERED 40 evaluations',flush=True)

def run():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in sources()}
    previous=js(OLD/'SUMMARY.json');t=base.Trial();dominant=t.S.copy();full=np.array(js(O/'FULL_COVERAGE.json')['sign_matrix']);assert np.array_equal(dominant,np.load(O/'FULL_MAPPING.npz')['dominant_sign_matrix'])
    (O/'COVERAGE.json').rename(O/'DOMINANT_COVERAGE.json');dump(O/'COVERAGE.json',js(O/'FULL_COVERAGE.json'))
    results={};history=[]
    try:
        for arm in ['P0','P2']:
            t.warm(arm);t.S=dominant.copy();ref=t.evaluate(arm,np.zeros(108),arm+'_dominant_replay')
            oldref=previous[arm]['baseline'];assert ref['J']==oldref['J']
            a=np.load(Path(ref['folder'])/'BODY.npz');b=np.load(Path(oldref['folder'])/'BODY.npz')
            assert all(np.array_equal(a[k],b[k]) for k in a.files)
            dump(Path(ref['folder'])/'MAPPING_IDENTITY.json',dict(mapping=str(O/'DOMINANT_COVERAGE.json'),sha256=sha(O/'DOMINANT_COVERAGE.json'),exact_historical_reproduction=True))
            t.S=full.copy();x=np.zeros(108);baseline=t.evaluate(arm,x,arm+'_baseline');best=baseline;bestx=x.copy();rng=np.random.default_rng(20261105);history.append(baseline)
            for gen,sigma in enumerate([.35,.20]):
                center=bestx.copy();prev=str(Path(best['folder'])/'PARAMETERS.json')
                for i in range(8):
                    z=np.clip(center+sigma*rng.standard_normal(108),-2,2)
                    r=t.evaluate(arm,z,f'{arm}_g{gen}_i{i}',previous=prev);history.append(r)
                    if r['J']<best['J']:best=r;bestx=z.copy()
                    dump(O/'HISTORY.json',history)
            repeat=t.evaluate(arm,bestx,arm+'_repeat');cut=t.evaluate(arm,bestx,arm+'_P1_cut',cut=True)
            a=np.load(Path(best['folder'])/'BODY.npz');b=np.load(Path(repeat['folder'])/'BODY.npz')
            exact=all(np.array_equal(a[k],b[k]) for k in a.files) and best['J']==repeat['J'];assert exact
            results[arm]=dict(dominant_replay=ref,baseline=baseline,best=best,repeat=repeat,cut=cut,exact_repeat=exact,training_evaluations=17,best_z=bestx)
            dump(O/'SUMMARY.json',results)
        print('FINISHED full mapping 40 evaluations',flush=True)
    finally:
        after={str(p):sha(p) for p in sources()};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after
if __name__=='__main__':
    if sys.argv[1]=='prereg':prereg()
    elif sys.argv[1]=='run':run()
