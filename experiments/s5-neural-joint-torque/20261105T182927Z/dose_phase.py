"""RH incoming excitation dose response and preregistered held-out release phases."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,importlib.util,inspect,textwrap,copy
from pathlib import Path
import numpy as np
P=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T181843Z/probe_neurons.py')
spec=importlib.util.spec_from_file_location('probe',P);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
e=p.e;b=e.b
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T182927Z')
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T181843Z')
js=b.js;dump=b.dump;sha=b.sha

def sources():return b.sources()+[OLD/'SUMMARY.json',OLD/'PREREGISTRATION.json',P,e.D,p.SRC_CODE,Path(__file__)]
def prereg():
    O.mkdir(parents=True,exist_ok=False)
    dump(O/'PREREGISTRATION.json',dict(stage='S5 RH dose response and held-out release phases',
        intervention='scale existing We incoming rows of 32 RH motor cells at release; full physical/neural/filter state preserved; no new connections/drive',
        doses=[1.,.25,.375,.5,.625,.75,.875],selection='minimum unchanged J at release1.3s among seven, including dose1 baseline; ties choose closest1',
        validation='repeat selected dose twice at1.3s; select once then freeze; held-out release1.225/1.25/1.35/1.375s each paired dose1 and frozen selected dose; no retuning by phase',
        controls='readout cut and sensory cut at1.3s selected dose',total_evaluations=19,
        duration='each release .2s; warmup PD gait unchanged except actual stop time; SNN starts0.2s as before; body never reset at release',
        scoring='same J/D1-D8; objective relative time origin adjusted to actual release time, no other metric changes. assisted reference actual final.2s of each warmup',
        timing=dict(body_dt_s=b.DT,neural_dt_s=b.NDT,command_dt_s=b.CDT),
        robustness='report paired J difference and each failed behavior gate at all held-out phases; favorable average is not support or walking',
        resource='one process, heavy jobs serialized',importance='核心必需',units='NOT_ESTABLISHED'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(f):sha(f) for f in sources()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(OLD/'PREREGISTRATION.json'),old_sha256=sha(OLD/'PREREGISTRATION.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Resolve RH excitation dose response and test frozen choice on unseen handover phases. Same body, readout, SNN and sensory settings.'))

def warm(t,duration,folder):
    b.O=folder;e.O=folder
    # Derive only this method; archived code is never modified.
    source=textwrap.dedent(inspect.getsource(b.Trial.warm))
    assert source.count('round(1.3/DT)')==1 and source.count('round(1.1/DT)')==1
    source=source.replace('round(1.3/DT)',f'round({duration!r}/DT)').replace('round(1.1/DT)',f'round({duration-.2!r}/DT)')
    namespace=dict(vars(b));exec(compile(source,str(Path(__file__)),'exec'),namespace)
    dump(folder/'WARMUP_ADAPTER.json',dict(original_file=b.__file__,original_sha256=sha(b.__file__),warmup_s=duration,changed_only=['warmup step count','assisted-reference last .2s start'],importance='核心必需'))
    namespace['warm'](t,'P0')

def run():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(f):sha(f) for f in sources()}
    b.O=O;t=e.dmod.DiagnosticTrial();(O/'COVERAGE.json').rename(O/'DOMINANT_COVERAGE.json');cover=js(p.SRC/'COVERAGE.json');dump(O/'COVERAGE.json',cover)
    old=js(OLD/'SUMMARY.json');cells=js(OLD/'PREREGISTRATION.json')['target_cells'];silent=js(OLD/'PREREGISTRATION.json')['silent_target_cells'];matrix=np.array(js(old['best']['matrix_file'])['matrix'])
    import train_dynamic_support as dst
    objective=dst.objective
    records=[];phases=[]
    def phase(duration):
        folder=O/f'PHASE_{duration:.3f}';folder.mkdir();dump(folder/'COVERAGE.json',cover);warm(t,duration,folder);return folder,copy.deepcopy(t.cache['P0'])
    def evaluate(folder,cache,duration,dose,label,cut=False,sens=False):
        b.O=folder;e.O=folder
        cfg=dict(label=label,exc_scale=dose);intervention=p.apply(t,cache,cfg,cells,silent)
        ip=folder/(label+'_INTERVENTION.json');dump(ip,intervention)
        def shifted(rows,assist):
            normalized=rows.copy();normalized[:,0]-=duration-1.3
            return objective(normalized,assist)
        dst.objective=shifted
        try:r=e.evaluate(t,'P0',matrix,label,cut=cut,sensory_cut=sens,previous=str(ip))
        finally:dst.objective=objective
        r.update(dose=dose,release_time_s=duration)
        dump(Path(r['folder'])/'PHASE_IDENTITY.json',dict(release_time_s=duration,dose=dose,intervention_file=str(ip),intervention_sha256=sha(ip),scoring_time_shift=duration-1.3))
        native=np.load(Path(r['folder'])/'NATIVE.npz')['rows'];neural=np.load(Path(r['folder'])/'NEURAL.npz')
        force=native[:,6];first=np.flatnonzero(force>1e-9)
        r['RH']=dict(spikes=int(neural['spikes'][:,cells].sum()),foot_load_mean=float(force.mean()/24.446354755878005),first_contact_s=None if len(first)==0 else float(native[first[0],0]-duration))
        records.append(r);dump(O/'PROGRESS.json',records);return r
    try:
        folder,cache=phase(1.3);dose_results=[]
        for dose in [1.,.25,.375,.5,.625,.75,.875]:
            r=evaluate(folder,cache,1.3,dose,'dose_'+str(dose));dose_results.append(r)
            if dose in [1.,.5]:
                ref=old['baseline'] if dose==1. else old['best'];assert r['J']==ref['J']
                aa=np.load(Path(r['folder'])/'BODY.npz');bb=np.load(Path(ref['folder'])/'BODY.npz');assert all(np.array_equal(aa[k],bb[k]) for k in aa.files)
        best=min(dose_results,key=lambda r:(r['J'],abs(r['dose']-1.)));chosen=best['dose']
        repeats=[evaluate(folder,cache,1.3,chosen,'best_repeat_'+str(i)) for i in range(2)]
        for r in repeats:
            assert r['J']==best['J'];aa=np.load(Path(r['folder'])/'BODY.npz');bb=np.load(Path(best['folder'])/'BODY.npz');assert all(np.array_equal(aa[k],bb[k]) for k in aa.files)
        cut=evaluate(folder,cache,1.3,chosen,'best_readout_cut',cut=True);sens=evaluate(folder,cache,1.3,chosen,'best_sensory_cut',sens=True)
        dump(folder/'SUMMARY.json',dict(baseline=dose_results[0],best=best,cut=cut,sensory_cut=sens,repeats=repeats))
        dump(O/'FROZEN_SELECTION.json',dict(dose=chosen,selected_from='only PHASE_1.300 seven doses',best=best))
        for duration in [1.225,1.25,1.35,1.375]:
            folder,cache=phase(duration);control=evaluate(folder,cache,duration,1.,'baseline');treated=evaluate(folder,cache,duration,chosen,'frozen_selected_dose')
            pair=dict(release_time_s=duration,baseline=control,treated=treated,delta_J=treated['J']-control['J']);phases.append(pair);dump(folder/'SUMMARY.json',pair)
        assert len(records)==19
        dump(O/'SUMMARY.json',dict(selected_dose=chosen,dose_results=dose_results,best=best,cut=cut,sensory_cut=sens,repeats=repeats,phase_pairs=phases,total_evaluations=len(records),all_dynamic_pass=all(r['validation']['passed'] for r in records),any_dynamic_pass=any(r['validation']['passed'] for r in records),walking_certified=False))
        print('DOSE AND PHASE COMPLETE',flush=True)
    finally:
        after={str(f):sha(f) for f in sources()};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after
if __name__=='__main__':
    if sys.argv[1]=='prereg':prereg()
    elif sys.argv[1]=='run':run()
