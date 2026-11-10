"""Pure pretrial checks; deliberately no MuJoCo, foundation stepping or optimizer launch."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from scoring import heading_from_points,cycle_score,score_episode,WEIGHT,NATIVE_DT
from search_rules import candidates,choose_update,SEED,SHAPE


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(records,name,ok,**details):
    records.append(dict(name=name,passed=bool(ok),**details))
    if not ok:raise AssertionError(name)

def main():
    records=[]
    prereg=json.loads((HERE/'PREREG.json').read_text())
    check(records,'prereg_parse',prereg['training_design']['seed']==20261107 and prereg['training_design']['total_trials']==53)
    check(records,'primary_newhistory_not_present',False if prereg.get('primary')=='newhistory' else True)
    check(records,'software_exception_abort_text','ABORT' in prereg['training_design']['incumbent_reevaluation'] or 'ABORT' in prereg['training_design'].get('early_stop',''))
    # Heading and cycle are the exact pure reward inputs.
    h=heading_from_points([1,0,1],[0,0,1]);check(records,'heading_unit',np.array_equal(h,[1.,0.]))
    try:heading_from_points([0,0,0],[0,0,0])
    except ValueError:check(records,'degenerate_heading_rejected',True)
    else:check(records,'degenerate_heading_rejected',False)
    loads=np.zeros((40,6));loads[:10]=WEIGHT/6;loads[20:30]=WEIGHT/6
    cycles=cycle_score(loads);check(records,'cycle_proxy_exact_one_leg',cycles['completed_legs']==6 and all(cycles['completed_by_leg']))
    ambiguous=np.full((40,6),.03*WEIGHT);check(records,'ambiguous_cycle_never_advances',cycle_score(ambiguous)['completed_legs']==0)
    # Pure synthetic native record: original columns t, foot6, nfpos, nfsigned, count, COMz...
    native=np.zeros((400,20));native[:,0]=np.arange(400)*NATIVE_DT;native[:,1:7]=WEIGHT/6;native[:,10]=1.2
    slip=np.zeros((40,13));slip[:,0]=np.arange(40)*.005;loads5=loads.copy();rates=np.zeros((40,42))
    result=score_episode(native,slip,loads5,rates,native[0,10]*np.array([0,0,1]),native[-1,10]*np.array([0,0,1]),h,dict(foot_support_mean=1.,com_z_mean=1.2),.005)
    check(records,'synthetic_score_finite',np.isfinite(result['reward']) and result['physical_gate']['passed'])
    check(records,'reward_cycle_term',result['reward_terms']['completed_cycle_legs']==6)
    # Antithetic population/order/clipping/update arithmetic.
    rng=np.random.default_rng(SEED);mean=np.zeros(SHAPE);std=np.full(SHAPE,.01);inc=np.zeros(SHAPE)
    noise,vals=candidates(rng,mean,std,inc)
    check(records,'candidate_shape_order',noise.shape==(6,)+SHAPE and vals.shape==(13,)+SHAPE and np.array_equal(vals[0]+vals[1],2*mean))
    check(records,'candidate_bounds',np.all(np.abs(vals)<=.25))
    results=[dict(valid=True,feasible=(i%2==0),reward=float(i)) for i in range(13)]
    update=choose_update(mean,std,vals,results,inc,-1.)
    check(records,'feasible_first_elites',[i for i in update['elites']]==[12,10,8,6])
    check(records,'incumbent_strict_feasible_best',update['incumbent_candidate']==12)
    results2=[dict(valid=True,feasible=True,reward=-float(i)) for i in range(13)]
    update2=choose_update(mean,std,vals,results2,inc,100.)
    check(records,'incumbent_preserved_when_no_improvement',update2['incumbent_candidate'] is None and update2['incumbent_reward']==100.)
    # Compile and statically inspect exception policy; no import of episode/optimizer.
    for name in ('scoring.py','search_rules.py','episode.py','optimize.py'):
        ast.parse((HERE/name).read_text(),filename=str(HERE/name));check(records,'syntax_'+name,True)
    episode_text=(HERE/'episode.py').read_text()
    check(records,'physical_invalid_only_candidate_exception','class PhysicalInvalid' in episode_text and 'software_exception=not isinstance(exc,PhysicalInvalid)' in episode_text and 'if software_exception:' in episode_text)
    check(records,'written_contact_rows_only','native_rows=native[:contact_rows].copy()' in episode_text)
    opt_text=(HERE/'optimize.py').read_text()
    check(records,'explicit_start_gate','--execute-approved-study' in opt_text and 'START_AUTHORIZATION.json' in opt_text)
    check(records,'no_automatic_trial','if not args.execute_approved_study' in opt_text)
    check(records,'openblas_gate',"os.environ.get(key)!='1'" in opt_text)
    check(records,'no_physics_modules_imported',not any(name in sys.modules for name in ('mujoco','episode','optimize')))
    hashes={name:sha(HERE/name) for name in ('PREREG.json','scoring.py','search_rules.py','episode.py','optimize.py','verify_pretrial.py')}
    output=dict(status='PASS_NO_PHYSICAL_TRIALS',checks_passed=len(records),checks=records,source_hashes=hashes,physical_steps=0,neural_steps=0,fit_solves=0,scope='Pure pretrial verification only; parent start authorization remains required')
    (HERE/'PRETRIAL_VERIFICATION.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('status','checks_passed','physical_steps','neural_steps')}))

if __name__=='__main__':main()
