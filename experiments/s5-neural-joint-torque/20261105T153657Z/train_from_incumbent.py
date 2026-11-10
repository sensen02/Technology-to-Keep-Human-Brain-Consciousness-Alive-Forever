"""Preserve best live closed-loop incumbent; train all existing readout strengths."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,importlib.util
from pathlib import Path
import numpy as np
SRC=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T153224Z/train_edge_readout.py')
spec=importlib.util.spec_from_file_location('edge_train',SRC);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T153657Z')
PRIOR=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T153224Z')
e.O=O;e.b.O=O
js=e.js;dump=e.dump;sha=e.sha

def sources():return e.inputs()+[PRIOR/'SUMMARY.json',SRC,Path(__file__)]
def prereg():
    assert (PRIOR/'SUMMARY.json').exists()
    O.mkdir(parents=True,exist_ok=False)
    dump(O/'PREREGISTRATION.json',dict(stage='S5 independent readout refinement from verified live incumbent',reason='Offline teacher fit can worsen closed-loop performance and yields exact zero coefficients; retain best actual rollout and allow reactivation instead of discarding prior best.',
        seed='per arm choose lower-J of previous full-map best reproduction and offline-fit-trained best; same cumulative budget in both arms',
        readout='same fixed 174 signed model-supported entries; independent nonnegative coefficient c; tau=clip(limit*(S*c)@rates/100,+/-limit); no bias/controller/reference',
        search=dict(seed=20261105,generations=3,population=8,sigma=[.15,.08,.04],proposal='clip(center + sigma*(center+0.1)*N(0,1),0,32) on 174 allowed entries; all other entries zero; fixed generation center; preserve best incumbent',training_per_arm=25),
        controls='baseline+24 candidates+exact repeat+readout cut+sensory cut per arm, total56',
        evaluation='unchanged .2s release, 1.3s algorithm warmup, body12.5us, neural0.5ms, command1ms, same D1-D8 and loss; no support gate changes',
        units='NOT_ESTABLISHED',resource='single process',importance='核心必需'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in sources()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(PRIOR/'PREREGISTRATION.json'),old_sha256=sha(PRIOR/'PREREGISTRATION.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Preserve closed-loop best; bounded additive independent coefficient proposals can reactivate zeros. No actuator-limit increase.'))

def run():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in sources()}
    prev=js(PRIOR/'SUMMARY.json');t=e.dmod.DiagnosticTrial();(O/'COVERAGE.json').rename(O/'DOMINANT_COVERAGE.json');cover=js(e.SOURCE/'FULL_COVERAGE.json');dump(O/'COVERAGE.json',cover);S=np.array(cover['sign_matrix']);mask=S!=0;history=[];summary={}
    try:
        for arm in ['P0','P2']:
            t.sensory_cut=False;t.warm(arm)
            chosen=min([prev[arm]['old_best_replay'],prev[arm]['best']],key=lambda r:r['J']);matrix=np.array(js(chosen['matrix_file'])['matrix']);C=abs(matrix)
            baseline=e.evaluate(t,arm,matrix,arm+'_baseline',previous=chosen['matrix_file']);assert baseline['J']==chosen['J']
            a=np.load(Path(baseline['folder'])/'BODY.npz');bb=np.load(Path(chosen['folder'])/'BODY.npz');assert all(np.array_equal(a[k],bb[k]) for k in a.files)
            best=baseline;bestC=C.copy();history.append(baseline);rng=np.random.default_rng(20261105)
            for gen,sigma in enumerate([.15,.08,.04]):
                center=bestC.copy();parent=best['matrix_file']
                for i in range(8):
                    cc=center.copy();cc[mask]=np.clip(center[mask]+sigma*(center[mask]+.1)*rng.standard_normal(int(mask.sum())),0,32)
                    r=e.evaluate(t,arm,S*cc,f'{arm}_g{gen}_i{i}',previous=parent);history.append(r)
                    if r['J']<best['J']:best=r;bestC=cc.copy()
                    dump(O/'HISTORY.json',history)
            repeat=e.evaluate(t,arm,S*bestC,arm+'_repeat',previous=best['matrix_file']);cut=e.evaluate(t,arm,S*bestC,arm+'_P1_cut',cut=True,previous=best['matrix_file']);sens=e.evaluate(t,arm,S*bestC,arm+'_sensory_cut',sensory_cut=True,previous=best['matrix_file'])
            a=np.load(Path(best['folder'])/'BODY.npz');bb=np.load(Path(repeat['folder'])/'BODY.npz');assert best['J']==repeat['J'] and all(np.array_equal(a[k],bb[k]) for k in a.files)
            summary[arm]=dict(baseline=baseline,best=best,repeat=repeat,cut=cut,sensory_cut=sens,exact_repeat=True,training_evaluations=25,best_coefficients=bestC,incumbent_source=chosen)
            dump(O/'SUMMARY.json',summary)
        print('INCUMBENT REFINEMENT COMPLETE 56 rollouts',flush=True)
    finally:
        after={str(p):sha(p) for p in sources()};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after
if __name__=='__main__':
    if sys.argv[1]=='prereg':prereg()
    elif sys.argv[1]=='run':run()
