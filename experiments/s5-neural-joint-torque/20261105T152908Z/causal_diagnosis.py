"""Frozen-best causal interventions, followed only by separately preregistered escalation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,importlib.util,copy
from pathlib import Path
import numpy as np
PREV=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py')
spec=importlib.util.spec_from_file_location('pilot',PREV);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T152908Z')
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T064731Z')
b.O=O;js=b.js;dump=b.dump;sha=b.sha

def inputs():return b.sources()+[SOURCE/'SUMMARY.json',SOURCE/'FULL_COVERAGE.json',Path(__file__)]
def prereg():
    O.mkdir(parents=True,exist_ok=False)
    dump(O/'PREREGISTRATION.json',dict(stage='S5 frozen-best causal diagnosis',source=str(SOURCE),body_dt_s=b.DT,neural_dt_s=b.NDT,command_dt_s=b.CDT,warmup_s=1.3,release_s=.2,
        baseline='P0 full-mapping best, exact reproduction required; P2 not retrained in diagnosis',
        interventions=['baseline','global_output_gain_x0.25','global_output_gain_x0.5','global_output_gain_x2','release_sensory_current_zero','release_readout_cut','one_leg_pool_readout_cut_each_of_6'],
        isolation='one intervention per arm, identical full physical/neural/filter initial states; network evolves with actual post-intervention body; readout cuts start exactly at release, no pose reset',
        hypotheses=dict(saturation='lower output gain improves support if excessive amplitude contributes; gain-up tests weakness alternative',feedback='zero release sensory current vs baseline; not a claim about feedback timing correctness',legs='zero only columns of pools belonging to each anatomical leg; no neuron/synapse removed'),
        scores='unchanged J and D1-D8; report first nonfoot contact, load, height, saturation and motion. Positive improvement alone is not support success.',budget=12,seed=0,
        escalation='If none passes all dynamic gates, authorize next separate preregistration: nonnegative independent gains on 174 existing model-supported readout entries, fit only offline from warmup neural rates to torque, then live neural-only release with P2 matched-budget and P1 cut. No biases, PD, phase/time inputs, added neurons or synapses.',
        resource='single process, heavy jobs serialized',units='NOT_ESTABLISHED',importance='核心必需'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in inputs()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(SOURCE/'SUMMARY.json'),old_sha256=sha(SOURCE/'SUMMARY.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='User requested causal diagnosis and more aggressive neural/readout training if diagnosis does not resolve support.'))

class DiagnosticTrial(b.Trial):
    sensory_cut=False
    def observe_neural(self,d,net,enc,rates):
        if not self.sensory_cut:return super().observe_neural(d,net,enc,rates)
        cur=np.zeros(12000);counts=np.zeros(12000,dtype=np.int32);decay=np.exp(-b.NDT/.005)
        for _ in range(2):
            sp=net.step(current_nA=cur);counts+=sp
            rates=decay*rates+(1-decay)*np.asarray(self.poolmat@sp,dtype=float)/b.NDT
        return rates,counts,cur

def evaluate(t,z,label,matrix,sensory_cut=False,cut=False):
    t.S=matrix.copy();t.sensory_cut=sensory_cut
    r=t.evaluate('P0',z,label,cut=cut,previous=str(SOURCE/'SUMMARY.json'))
    f=Path(r['folder']);dump(f/'READOUT_MATRIX.json',dict(matrix=t.S,sensory_cut=sensory_cut,cut=cut,importance='核心必需'))
    params=f/'PARAMETERS.json';dump(f/'CHANGE_LEDGER.json',dict(old_file=str(SOURCE/'SUMMARY.json'),old_sha256=sha(SOURCE/'SUMMARY.json'),new_file=str(params),new_sha256=sha(params),matrix_file=str(f/'READOUT_MATRIX.json'),matrix_sha256=sha(f/'READOUT_MATRIX.json'),reason=label))
    return r

def run():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in inputs()}
    t=DiagnosticTrial();(O/'COVERAGE.json').rename(O/'DOMINANT_COVERAGE.json');full=js(SOURCE/'FULL_COVERAGE.json');dump(O/'COVERAGE.json',full)
    S=np.array(full['sign_matrix']);best=js(SOURCE/'SUMMARY.json')['P0'];z=np.array(best['best_z']);t.warm('P0');res=[]
    try:
        r=evaluate(t,z,'baseline',S);res.append(r)
        old=np.load(Path(best['best']['folder'])/'BODY.npz');new=np.load(Path(r['folder'])/'BODY.npz')
        assert r['J']==best['best']['J'] and all(np.array_equal(old[k],new[k]) for k in old.files)
        for scale in [.25,.5,2.]:
            zz=z.copy();zz[66:]+=np.log(scale);res.append(evaluate(t,zz,'gain_'+str(scale),S))
        res.append(evaluate(t,z,'sensory_cut',S,sensory_cut=True))
        res.append(evaluate(t,z,'readout_cut',S,cut=True))
        man=js(b.B/'RUN/MAPPING.json')['rows'];poollegs={}
        for row in man:
            key=tuple(row['local_indices'])
            if key:
                if key in poollegs:assert poollegs[key]==row['leg']
                poollegs[key]=row['leg']
        for leg in b.LEGS:
            ss=S.copy();columns=[i for i,p in enumerate(t.pools) if poollegs[tuple(p)]==leg];ss[:,columns]=0
            res.append(evaluate(t,z,'cut_leg_'+leg,ss))
            dump(Path(res[-1]['folder'])/'CUT_POOL_INDICES.json',dict(leg=leg,pool_columns=columns))
        dump(O/'DIAGNOSIS.json',dict(results=res,baseline_exact=True,any_dynamic_pass=any(r['validation']['passed'] for r in res),escalation_needed=not any(r['validation']['passed'] for r in res)))
    finally:
        after={str(p):sha(p) for p in inputs()};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after
    print('DIAGNOSIS COMPLETE',flush=True)
if __name__=='__main__':
    if sys.argv[1]=='prereg':prereg()
    elif sys.argv[1]=='run':run()
