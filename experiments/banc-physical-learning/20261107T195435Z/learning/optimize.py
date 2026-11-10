"""Execute exactly the preregistered checkpoint study after parent review gate."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import numpy as np
from search_rules import SEED,SHAPE,candidates,choose_update

HERE=Path(__file__).resolve().parent
OUTPUT=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning')
FOUNDATION=HERE.parent/'foundation'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


def confirm_determinism(old_summary,new_summary):
    oldfolder,newfolder=Path(old_summary['folder']),Path(new_summary['folder'])
    old=json.loads((oldfolder/'RESULT.json').read_text());new=json.loads((newfolder/'RESULT.json').read_text())
    if old['valid']!=new['valid'] or old['feasible']!=new['feasible']:raise AssertionError('Incumbent feasibility changed across full reset')
    if old['reward'] is None or new['reward'] is None:raise AssertionError('Canonical incumbent control invalid')
    errors={'reward':abs(old['reward']-new['reward'])}
    digests={}
    for filename in ('BODY.npz','NATIVE.npz','COMMANDS.npz','POLICY.npz','SLIP.npz','NEURAL_FINAL.npz'):
        a,b=load(oldfolder/filename),load(newfolder/filename)
        if a.keys()!=b.keys():raise AssertionError('Incumbent schema mismatch')
        digest_a=hashlib.sha256();digest_b=hashlib.sha256()
        for key in a:
            aa,bb=a[key],b[key]
            if aa.shape!=bb.shape or aa.dtype!=bb.dtype:raise AssertionError('Incumbent array shape/type differs')
            digest_a.update(key.encode()+aa.tobytes());digest_b.update(key.encode()+bb.tobytes())
            if aa.dtype.kind in 'USO':
                if not np.array_equal(aa,bb):raise AssertionError('Incumbent identity changed')
            else:
                if not np.isfinite(aa).all() or not np.isfinite(bb).all():raise AssertionError('Incumbent nonfinite')
                errors[filename+':'+key]=float(np.max(np.abs(aa.astype(float)-bb.astype(float)))) if aa.size else 0.
        digests[filename]=dict(old=digest_a.hexdigest(),new=digest_b.hexdigest(),exact=digest_a.digest()==digest_b.digest())
    if max(errors.values())>1e-9:raise AssertionError('Incumbent reset/crossworker determinism exceeds1e-9: '+str(errors))
    return dict(status='PASS_CROSS_WORKER_RESET_CONTROL',old_episode=old['episode_id'],new_episode=new['episode_id'],max_abs_error=max(errors.values()),errors=errors,array_digests=digests)


def task_for(index,theta,duration=1.,final=False,label=None):
    label=label or ('trial_'+str(index).zfill(3))
    return dict(episode_id=label,relative_path=('evaluation/'+label if final else 'training/'+label),theta=np.asarray(theta).tolist(),duration_s=duration,final_validation=final,condition=label,evaluation_seed=SEED+index)


def collect(pool,tasks,ledger):
    from episode import run_task
    futures={pool.submit(run_task,task):i for i,task in enumerate(tasks)}
    results=[None]*len(tasks);errors=[]
    for future in as_completed(futures):
        i=futures[future]
        try:
            result=future.result();results[i]=result;ledger.append(result)
            print(json.dumps(dict(event='episode_finished',episode=result['episode_id'],valid=result['valid'],feasible=result['feasible'],reward=result['reward'],native_steps=result['native_steps'],wall_s=result['wall_s'])),flush=True)
        except Exception:
            errors.append(dict(task=tasks[i],error=traceback.format_exc()))
            # Stop queued work; already-running tasks finish and their actual calls are collected.
            for pending in futures:
                if not pending.done():pending.cancel()
        dump(OUTPUT/'CALL_LEDGER.json',dict(episodes=ledger,actual_native_steps=sum(x['native_steps'] for x in ledger),actual_neural_steps=sum(x['neural_steps'] for x in ledger),actual_live_forward_calls=sum(x['live_forward_calls'] for x in ledger),software_errors=errors))
    if errors:
        dump(OUTPUT/'ABORTED_TASKS.json',errors)
        raise RuntimeError('Software task failure or cancellation: study aborted; inspect ABORTED_TASKS and pertrial RESULT')
    return results


def actual_ledger_from_records(ledger):
    # Includes failed software task partial counters omitted from Future return.
    by_id={x['episode_id']:x for x in ledger}
    for folder in ('training','evaluation'):
        root=OUTPUT/folder
        if root.exists():
            for path in root.glob('*/RESULT.json'):
                r=json.loads(path.read_text());by_id[r['episode_id']]=dict(episode_id=r['episode_id'],folder=str(path.parent),native_steps=r['native_steps'],neural_steps=r['neural_steps'],live_forward_calls=r['live_forward_calls'],valid=r['valid'],software_exception=r.get('software_exception',False))
    return dict(episodes=list(by_id.values()),actual_native_steps=sum(x['native_steps'] for x in by_id.values()),actual_neural_steps=sum(x['neural_steps'] for x in by_id.values()),actual_live_forward_calls=sum(x['live_forward_calls'] for x in by_id.values()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-approved-study',action='store_true');args=parser.parse_args()
    if not args.execute_approved_study:raise RuntimeError('Explicit parent-reviewed execute flag required')
    approval=json.loads((HERE/'START_AUTHORIZATION.json').read_text())
    if approval['status']!='PARENT_REVIEWED_START':raise RuntimeError('Parent authorization record missing')
    for path,expected in approval['reviewed_sha256'].items():
        if sha(path)!=expected:raise ValueError('Reviewed source changed: '+path)
    checks=json.loads((HERE/'PRETRIAL_VERIFICATION.json').read_text())
    if checks['status']!='PASS_NO_PHYSICAL_TRIALS':raise RuntimeError('Purearray checks not passed')
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
        if os.environ.get(key)!='1':raise RuntimeError(key+' must be1')
    if (OUTPUT/'training').exists() or (OUTPUT/'STUDY_RESULT.json').exists():raise FileExistsError('Never rerun/overwrite this study')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED);mean=np.zeros(SHAPE);std=np.full(SHAPE,.01);incumbent=np.zeros(SHAPE);ledger=[];history=[];started=time.monotonic()
    from episode import initialize_worker
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
            baseline=collect(pool,[task_for(0,incumbent)],ledger)[0]
            if not baseline['valid']:raise RuntimeError('Zero baseline invalid; abort rather than optimize from broken checkpoint')
            incumbent_reward=float(baseline['reward']);incumbent_summary=baseline
            for generation in range(4):
                folder=OUTPUT/('generation_'+str(generation));folder.mkdir()
                noise,values=candidates(rng,mean,std,incumbent)
                before=dict(generation=generation,seed=SEED,mean=mean,std=std,incumbent_theta=incumbent,incumbent_reward=incumbent_reward,incumbent_episode=incumbent_summary['episode_id'],rng_state_after_draw=rng.bit_generator.state,candidate_order='pair0plus,pair0minus,...pair5minus,incumbent',noise=noise,candidates=values)
                dump(folder/'BEFORE.json',before)
                tasks=[task_for(1+generation*13+i,values[i]) for i in range(13)]
                results=collect(pool,tasks,ledger)
                determinism=confirm_determinism(incumbent_summary,results[12]);dump(folder/'INCUMBENT_DETERMINISM.json',determinism)
                update=choose_update(mean,std,values,results,incumbent,incumbent_reward)
                dump(folder/'AFTER.json',dict(generation=generation,results=results,**update))
                if update['incumbent_candidate'] is not None:incumbent_summary=results[update['incumbent_candidate']]
                mean,std=update['next_mean'],update['next_std'];incumbent,incumbent_reward=update['incumbent_theta'],update['incumbent_reward']
                history.append(dict(generation=generation,incumbent_reward=incumbent_reward,incumbent_episode=incumbent_summary['episode_id'],elites=update['elites'],feasible=len(update['feasible_candidates'])))
                print(json.dumps(dict(event='generation_complete',**history[-1])),flush=True)
            np.savez_compressed(OUTPUT/'SELECTED.npz',theta=incumbent,mean=mean,std=std,training_reward=incumbent_reward,selected_episode=incumbent_summary['episode_id'])
            # Validation never changes selected theta or generation decisions.
            selected,zero=collect(pool,[task_for(53,incumbent,5.,True,'selected'),task_for(54,np.zeros(SHAPE),5.,True,'zero')],ledger)
        actual=actual_ledger_from_records(ledger)
        if actual['actual_native_steps']!=5040000 or actual['actual_neural_steps']!=126000:
            # Physical-invalid policies may have stopped early; retain real counts.
            expected_note='Some preregistered physical-invalid episodes stopped early; inspect records'
        else:expected_note='All53training+2validation episodes completed expectedcounts'
        result=dict(status='PREREGISTERED_PHYSICAL_LEARNING_COMPLETE_NOT_WALKING',seed=SEED,training_trials=53,validation_trials=2,generations=history,selected_training_episode=incumbent_summary['episode_id'],selected_training_reward=incumbent_reward,baseline_training_reward=baseline['reward'],selected_validation=selected,zero_validation=zero,actual_calls=actual,counts_note=expected_note,wall_s=time.monotonic()-started,source_hashes=approval['reviewed_sha256'],reward_is_acceptance=False)
        dump(OUTPUT/'STUDY_RESULT.json',result);dump(OUTPUT/'CALL_LEDGER.json',actual)
        print(json.dumps(dict(status=result['status'],actual_native_steps=actual['actual_native_steps'],actual_neural_steps=actual['actual_neural_steps'],wall_s=result['wall_s'])),flush=True)
    except Exception:
        actual=actual_ledger_from_records(ledger)
        dump(OUTPUT/'STUDY_ABORTED.json',dict(error=traceback.format_exc(),actual_calls=actual,wall_s=time.monotonic()-started))
        dump(OUTPUT/'CALL_LEDGER.json',actual)
        raise

if __name__=='__main__':main()
