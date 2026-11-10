"""Finite actual-physics search: 1 center +6 batches x4 antithetic candidates.

Parent must explicitly launch --run. No physics occurs at import. Same original
plant/PD128/reference101114, six target-offset parameters only, no ANN claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types

import numpy as np

SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T112003Z/optimize_stance.py')
SOURCE_SHA256='7a4a4d57d26a2c4914a40751ac85fdab17cb473eddb6ab266e6eab5c39962801'
HERE=Path(__file__).resolve()
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T115537Z')
CENTER_EVIDENCE=OUT.parent/'20261107T113529Z/BEST_PARAMETERS.json'
CENTER=np.array([0.,.02,0.,0.,0.,.01])
SEED,SIGMA,BATCHES,TRIALS=115537,.025,6,25
STEP_BUDGET=25*240000
original=SOURCE.read_text()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256,'Frozen search implementation changed'
patches=[
    ("OUT=GAIT.parent.parent/'20261107T112003Z'","OUT=GAIT.parent.parent/'20261107T115537Z'"),
    ("tested_offset_rad=.02,maximum_trials=13","tested_offset_rad=.02,maximum_trials=25"),
    ("parameter_bounds_rad=[-.03,.03]","parameter_bounds_rad=[-.08,.08]"),
    ("Hard13-trial budget exhausted","Hard25-trial budget exhausted"),
    ("or abs(parameters).max()>.03:","or abs(parameters).max()>.08:"),
    ("Six finite parameters within+/-0.03rad required","Six finite parameters within+/-0.08rad required"),
    ("source_sha256=n.sha(__file__),seconds=SECONDS,maximum_steps=240000,",
     "source_sha256=n.sha(Path('"+str(HERE)+"')),seconds=SECONDS,maximum_steps=240000,"),
    ("                    mj.mj_step(self.m,d);self.physics_steps+=1;trialsteps+=1",
     "                    if self.physics_steps>=6000000:raise RuntimeError('Hard native-step budget exhausted')\n                    mj.mj_step(self.m,d);self.physics_steps+=1;trialsteps+=1"),
]
effective=original
for old,new in patches:
    assert effective.count(old)==1,('Expected one frozen source match',old)
    effective=effective.replace(old,new,1)
EFFECTIVE_SHA256=hashlib.sha256(effective.encode()).hexdigest()
runtime=types.ModuleType('adaptive_physical_search_runtime');runtime.__file__=str(SOURCE)
exec(compile(effective,str(SOURCE),'exec'),runtime.__dict__)
original_score=runtime.score_trial


def score_trial(data,reason):
    score=original_score(data,reason)
    if 'slip_to_progress_ratios' not in score:return score
    ratios=np.asarray(score['slip_to_progress_ratios'])
    original_mean=float(ratios.mean());penalty=float(score['objective']-original_mean)
    release=data[data[:,0]>=1.-1e-9]
    unloaded=release[:,1:7].sum(0)<=0
    score.update(objective=float(ratios.max()+.1*original_mean+penalty),
        maximum_leg_ratio=float(ratios.max()),mean_leg_ratio=original_mean,unchanged_penalty=penalty,
        never_loaded_leg_mask=unloaded.tolist())
    if unloaded.any():score.update(failed=True,selection_failure='NEVER_LOADED_LEG')
    if not np.isfinite(score['objective']):score.update(objective=1e6,failed=True,selection_failure='NONFINITE_OBJECTIVE')
    return score


runtime.score_trial=score_trial


def generate_batch(rng,center):
    directions=rng.normal(size=(2,6))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    perturbations=np.array([directions[0],-directions[0],directions[1],-directions[1]])*SIGMA
    candidates=np.clip(center[None,:]+perturbations,-.08,.08)
    return directions,perturbations,candidates


def eligible(row):
    return bool(not row['score']['failed'] and row['score'].get('completed_full_duration',False)
        and row['finite'] and row['warnings']==0 and np.isfinite(row['score']['objective']))


def measured_best(previous,rows):
    valid=([previous] if previous is not None and eligible(previous) else [])+[r for r in rows if eligible(r)]
    return min(valid,key=lambda r:r['score']['objective']) if valid else None


def sources():
    files={HERE,SOURCE,CENTER_EVIDENCE,runtime.GAIT,runtime.a_module.DEFAULT_MODEL_PATH}
    seen=set()
    def visit(module):
        if id(module) in seen:return
        seen.add(id(module));path=getattr(module,'__file__',None)
        if path and str(path).startswith(str(runtime.ROOT)) and '/site-packages/' not in str(path):
            files.add(Path(path))
            for value in vars(module).values():
                if isinstance(value,types.ModuleType):visit(value)
    visit(runtime.b);visit(runtime.a_module)
    return {str(f):runtime.n.sha(f) for f in sorted(files)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if not args.run:ap.error('Parent review required; launch --run for25 actual-physics trials')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():raise FileExistsError('Preserve prior adaptive search '+str(OUT))
    evidence=json.loads(CENTER_EVIDENCE.read_text())
    assert np.array_equal(evidence['selected']['parameters_rad'],CENTER),'Declared measured center mismatch'
    hashes=sources()
    (OUT/'EFFECTIVE_SEARCH.py').write_text(effective)
    runtime.n.dump(OUT/'PREREGISTRATION.json',dict(stage='S5 finite adaptive actualphysics reference-offset search',
        source_hashes=hashes,source_baseline_sha256=SOURCE_SHA256,effective_sha256=EFFECTIVE_SHA256,
        patches=[dict(old=old,new=new,required_matches=1) for old,new in patches],
        center=CENTER,center_evidence=str(CENTER_EVIDENCE),joint_order=runtime.JOINTS,
        budget=dict(new_trials=25,baseline_retest=1,batches=6,candidates_per_batch=4,
            seconds_per_trial=3.,warmup=1.,score=2.,max_native_steps=STEP_BUDGET,
            previously_declared_offset_trials=26,combined_declared_offset_trials=51,
            aggregate_note='26prior13+13offsetsearches only; unrelatedANN/strict/controlruns are notincludedandnotclaimedzero.'),
        sampling=dict(seed=SEED,normal_directions_per_batch=2,normalization='each6vectorunitEuclidean norm',sigma=SIGMA,
            order='center+.025d1,center-.025d1,center+.025d2,center-.025d2',global_parameter_bounds=[-.08,.08],
            update='After4actualtrials choosebestVALIDmeasuredcandidate amongpriorbest+4. CenterNEVERweighted/averaged; sigmafixed. Exactrngstate,directions,rawperturbations,clippingandrankingsrecordedbefore/aftereachbatch.'),
        objective='max6(slip/materialprogressratio)+.1*mean6ratio +UNCHANGED112003fall/nonfoot/support/height/no-progress/lateral/yaw/unloadedpenalties. Neverloadedlegorotherfailure noteligible.',
        physics=runtime.PARAM,reference='101114broadlift1Hz, original128finitePD,30rad/sincrementcap;offsetS5initialramp.5s;nativejointclipfractionsrecorded. No gain/force/friction/mass/rootchanges.',
        selection='Center113529latest selectedretestedfirst. Failedcandidates cannotreplace stablemeasuredbest; tieskeeppriorbest. All25actualtrialresults retained withcumulativestepcount. No heldoutselection.',
        integrity='Full71mesh49/48/42neq0/fullgravity[0,0,-9801]/dt12.5us; torqueandmj_forwardevery native step; contactmaterialJvvsxipos-twistcrosscheck5ms. Actualqpos/qvel writesfreshinitonly.',
        acceptance='5msscores/100msCOMendpointprogressareSEARCHONLY. Winner requiresseparatefullnativeD/clearance/realsteps/heading/materialslip<=.5 strictreplay. Algorithmteacherneverneuralgoal; no ANN deployedhere.'))
    search=runtime.Search();rng=np.random.default_rng(SEED);results=[];trace=[]
    center_trial=search.trial('measured_center',CENTER.copy());results.append(center_trial)
    best=measured_best(None,[center_trial]);center=CENTER.copy()
    def persist():
        runtime.n.dump(OUT/'SEARCH_RESULTS.json',dict(results=results,trials_started=search.trials_started,
            actual_physics_steps=search.physics_steps,maximum_trials=TRIALS,previous_offset_trials=26,combined_declared_budget=51))
        runtime.n.dump(OUT/'ADAPTATION_TRACE.json',trace)
    persist()
    for batch in range(BATCHES):
        prior=best;center=np.asarray(best['parameters_rad']).copy() if best is not None else center
        state_before=rng.bit_generator.state
        directions,perturbations,parameters=generate_batch(rng,center)
        record=dict(batch=batch,prior_best_tag=prior['tag'] if prior is not None else None,
            prior_best_objective=prior['score']['objective'] if prior is not None else None,
            center=center.copy(),rng_state_before=state_before,rng_state_after=rng.bit_generator.state,
            directions=directions,perturbations=perturbations,candidates=parameters,
            parameter_clip_mask=(center[None,:]+perturbations)!=parameters,status='PREREGISTERED_BEFORE_BATCH')
        trace.append(record);persist()
        evaluated=[]
        for i,parameter in enumerate(parameters):
            outcome=search.trial('batch'+str(batch+1)+'_candidate'+str(i+1),parameter.copy())
            evaluated.append(outcome);results.append(outcome);persist()
        best=measured_best(prior,evaluated)
        ranking=sorted(([prior] if prior is not None else [])+evaluated,key=lambda r:r['score']['objective'])
        record.update(status='BATCH_MEASURED',ranking=[dict(tag=r['tag'],objective=r['score']['objective'],eligible=eligible(r)) for r in ranking],
            selected_tag=best['tag'] if best else None,selected_parameters=best['parameters_rad'] if best else None,
            selected_objective=best['score']['objective'] if best else None,
            measured_selection_only=True)
        persist()
    assert search.trials_started==TRIALS and search.physics_steps<=STEP_BUDGET
    runtime.n.dump(OUT/'BEST_PARAMETERS.json',dict(selected=best,baseline_retest=center_trial,
        trials_attempted=search.trials_started,actual_physics_steps=search.physics_steps,
        previous_offset_trials=26,total_declared_offset_budget=51,source_hashes_unchanged={f:runtime.n.sha(f)==h for f,h in hashes.items()},
        status='PENDING_STRICT_HELDOUT_NATIVE_REPLAY' if best else 'NO_VALID_TRIAL',neural_success=False))


if __name__=='__main__':main()
