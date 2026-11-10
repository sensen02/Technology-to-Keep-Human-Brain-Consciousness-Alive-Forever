"""Independent saved-study audit. No optimizer/runner/scoring imports or steps.

Reconstructs reward, 53 candidates, selection and fixed moment updates directly
from immutable saved arrays and RNG seed. Original dsm gates remain the gate.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
OUTPUT=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning')
CHECKPOINT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop')
FOUNDATION=HERE.parent/'foundation'
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype/tools')
import dynamic_support_metrics as dsm
W=24.446354755878005
DT=1.25e-5


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


class Audit:
    def __init__(self):self.checks=[];self.episode_metrics=[]
    def check(self,name,ok,**details):
        self.checks.append(dict(name=name,passed=bool(ok),**details))
        if not ok:raise AssertionError(name)
    def same(self,name,a,b,tol=0.):
        a,b=np.asarray(a),np.asarray(b)
        self.check(name+'_shape',a.shape==b.shape)
        if a.dtype.kind in 'USO' or b.dtype.kind in 'USO':self.check(name,np.array_equal(a,b));return
        self.check(name+'_finite',np.isfinite(a).all() and np.isfinite(b).all())
        error=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
        self.check(name,np.array_equal(a,b) if tol==0 else error<=tol,max_abs_error=error,tolerance=tol)


def independent_cycles(load):
    completed=[]
    for col in range(6):
        state=np.zeros(len(load),int)
        state[load[:,col]>.05*W]=1;state[load[:,col]<.005*W]=-1
        edges=np.r_[0,np.flatnonzero(np.diff(state)!=0)+1,len(state)]
        seq=[int(state[left]) for left,right in zip(edges[:-1],edges[1:]) if right-left>=4 and state[left]!=0]
        sought=[1,-1,1];stage=0
        for value in seq:
            if value==sought[stage]:stage+=1
            if stage==3:break
        completed.append(stage==3)
    return completed


def audit_episode(a,folder,assist,canonical):
    folder=Path(folder);result=read(folder/'RESULT.json');label=result['episode_id']
    for name,expected in result['files'].items():a.check(label+'_filehash_'+name,sha(folder/name)==expected)
    a.check(label+'_no_software_error',not result.get('software_exception',False))
    a.check(label+'_complete_valid',result['valid'] and result['failure'] is None)
    protocol=read(folder/'PROTOCOL.json')
    for path,expected in protocol['source_hashes'].items():a.check(label+'_sourcehash_'+path,sha(path)==expected)
    native=load(folder/'NATIVE.npz')['rows'];body=load(folder/'BODY.npz');commands=load(folder/'COMMANDS.npz')['rows'];policy=load(folder/'POLICY.npz');slip=load(folder/'SLIP.npz')['rows'];sc=load(folder/'SCORE_INPUTS.npz');actor=load(folder/'ACTOR.npz');initial=load(folder/'INITIAL.npz')
    a.same(label+'_canonical_physics',initial['state'],canonical['physics_state'])
    a.same(label+'_canonical_target',initial['target'],canonical['actuator_target'])
    neural=load(folder/'NEURAL_INITIAL.npz');frozen=load(FOUNDATION/'NEURAL_START.npz')
    for key in frozen:a.same(label+'_canonical_neural_'+key,neural[key],frozen[key])
    duration=float(protocol['duration_s']);steps=int(round(duration/DT));ticks=int(round(duration/.005))
    a.check(label+'_step_counts',result['native_steps']==steps==len(native) and result['neural_steps']==ticks*10 and result['contact_rows_saved']==steps)
    a.same(label+'_start_clock',body['time'][0],1.,1e-8)
    a.same(label+'_end_clock',body['time'][-1],1.+duration,1e-8)
    a.same(label+'_command_actual_starttime',commands[:,0],body['time'][:-1])
    a.same(label+'_policy_times',policy['time_s'],body['time'][np.arange(ticks)*5])
    a.same(label+'_score_rates',sc['rates5ms'],policy['predicted_rate'])
    a.same(label+'_commands_held_rates',commands[:,1:43],np.repeat(policy['predicted_rate'],5,axis=0))
    a.same(label+'_theta',actor['theta'],result['theta'])
    a.check(label+'_theta_bounds',np.all(np.abs(actor['theta'])<=.25))
    base=load(FOUNDATION/'BASE.npz')
    weights=np.zeros((323,42))
    for li,leg in enumerate(('LF','LH','LM','RF','RH','RM')):
        rows=np.flatnonzero(base['mn_leg']==leg.lower());cols=np.flatnonzero(base['joint_leg']==leg)
        weights[np.ix_(rows,cols)]=np.clip(base['basis'][rows,li*4:li*4+4]@actor['theta'][li],-1,1)
    a.same(label+'_independent_actor_weights',weights,actor['weights'])
    prediction=30*np.tanh((policy['mn_rates_hz']/100.)@weights)
    a.same(label+'_independent_MN_predictions',prediction,policy['predicted_rate'],1e-10)
    for key in ('mn_root_ids','mn_local','mn_leg','joint_names'):a.same(label+'_actor_identity_'+key,actor[key],base[key])
    q=body['qpos'][np.arange(ticks)*5][:,initial['qa']];vel=body['qvel'][np.arange(ticks)*5][:,initial['va']]
    a.same(label+'_localq',q,policy['local_physical_input'][:,:42]);a.same(label+'_localvel',vel,policy['local_physical_input'][:,42:84])
    a.same(label+'_localfoot_previousnative',native[np.arange(1,ticks)*400-1,1:7],policy['local_physical_input'][1:,84:])
    a.same(label+'_foot_endpoint_score',sc['loads5ms'],native[np.arange(1,ticks+1)*400-1,1:7])
    a.same(label+'_score_startCOM',sc['start_com'],body['com'][0]);a.same(label+'_score_endCOM',sc['end_com'],body['com'][-1])
    heading=np.asarray(protocol['initial_heading_xy']);a.same(label+'_score_heading',sc['heading'],heading)
    a.same(label+'_unit_heading',np.linalg.norm(heading),1.,1e-12)
    disp=body['com'][-1]-body['com'][0]
    cycles=independent_cycles(sc['loads5ms'])
    slip_dt=float(protocol['slip_dt'])
    terms=dict(forward_displacement=float(np.dot(disp[:2],heading)),weighted_slip_distance=float(slip[:,7:13].sum()*slip_dt/W),lateral_displacement=float(np.dot(disp[:2],[-heading[1],heading[0]])),height_drop_excess=max(0.,float(body['com'][0,2])-min(float(body['com'][0,2]),float(native[:,10].min()))-.05),positive_nonfoot_impulse_overW=float(np.sum(native[:,7])*DT/W),support_deficit_integral=float(np.sum(np.maximum(.6-np.sum(native[:,1:7],axis=1)/W,0))*DT),rate_squared_time=float(np.mean(policy['predicted_rate']**2)*duration),completed_cycle_legs=int(sum(cycles)))
    reward=terms['forward_displacement']-2*terms['weighted_slip_distance']-.5*abs(terms['lateral_displacement'])-5*terms['height_drop_excess']-5*terms['positive_nonfoot_impulse_overW']-5*terms['support_deficit_integral']-.0005*terms['rate_squared_time']+.03*terms['completed_cycle_legs']
    for key,value in terms.items():a.same(label+'_rewardterm_'+key,value,result['reward_terms'][key],1e-10)
    a.same(label+'_reward',reward,result['reward'],1e-10);a.same(label+'_cycleflags',cycles,result['cycles']['completed_by_leg'])
    gate=dsm.evaluate_native(assist,native);gate['checks']['D1_no_nonfoot_ground_force']=bool(native[:,7].max()==0);gate['passed']=all(gate['checks'].values())
    a.check(label+'_original_gate',gate==result['physical_gate'] and gate['passed']==result['feasible'])
    end=load(folder/'NEURAL_FINAL.npz');a.same(label+'_end_neural_clock',end['t_ms'],1000.+duration*1000)
    a.same(label+'_end_neural_steps',end['steps'],2000+ticks*10)
    summary=dict(episode_id=label,reward=float(reward),feasible=bool(gate['passed']),valid=True,native_steps=steps,neural_steps=ticks*10,folder=str(folder),theta=actor['theta'].tolist())
    a.episode_metrics.append(summary)
    return summary


def main():
    out=HERE/'INDEPENDENT_STUDY_VERIFICATION.json'
    if out.exists():raise FileExistsError(str(out))
    started=time.monotonic();a=Audit()
    try:
        study=read(OUTPUT/'STUDY_RESULT.json');authorization=read(HERE/'START_AUTHORIZATION.json')
        for path,expected in authorization['reviewed_sha256'].items():a.check('reviewed_hash_'+path,sha(path)==expected)
        a.check('fixed_seed_budget',study['seed']==20261107 and study['training_trials']==53 and study['validation_trials']==2)
        source=load(CHECKPOINT/'NATIVE.npz')['rows'];assistrows=source[(source[:,0]>=.8)&(source[:,0]<1.)]
        assist=dict(foot_support_mean=float(assistrows[:,1:7].sum(axis=1).mean()/W),com_z_mean=float(assistrows[:,10].mean()))
        canonical=load(CHECKPOINT/'HANDOVER.npz')
        baseline=audit_episode(a,OUTPUT/'training/trial_000',assist,canonical)
        mean=np.zeros((6,4,7));std=np.full((6,4,7),.01);inc=np.zeros((6,4,7));inc_reward=baseline['reward'];inc_id='trial_000';rng=np.random.default_rng(20261107)
        controls=[]
        for gen in range(4):
            before=read(OUTPUT/f'generation_{gen}/BEFORE.json');after=read(OUTPUT/f'generation_{gen}/AFTER.json')
            a.same(f'g{gen}_mean_before',before['mean'],mean);a.same(f'g{gen}_std_before',before['std'],std);a.same(f'g{gen}_inc_before',before['incumbent_theta'],inc)
            a.same(f'g{gen}_reward_before',before['incumbent_reward'],inc_reward,1e-10)
            draw=rng.standard_normal((6,6,4,7));values=[]
            for pair in draw:values.extend([np.clip(mean+std*pair,-.25,.25),np.clip(mean-std*pair,-.25,.25)])
            values.append(inc.copy());values=np.asarray(values)
            a.same(f'g{gen}_rng_draw',before['noise'],draw);a.same(f'g{gen}_candidates',before['candidates'],values)
            results=[]
            for i in range(13):
                index=1+gen*13+i;episode=audit_episode(a,OUTPUT/f'training/trial_{index:03d}',assist,canonical)
                a.same(f'g{gen}_candidate{i}_theta',episode['theta'],values[i]);results.append(episode)
            # Verify archived incumbent repeated under another worker/reset.
            prior=next(item for item in a.episode_metrics if item['episode_id']==inc_id)
            a.same(f'g{gen}_incumbent_control_score',results[12]['reward'],prior['reward'],1e-9)
            control=read(OUTPUT/f'generation_{gen}/INCUMBENT_DETERMINISM.json')
            a.check(f'g{gen}_incumbent_control_report',control['status']=='PASS_CROSS_WORKER_RESET_CONTROL' and control['max_abs_error']<=1e-9)
            # Independently inspect arrays rather than trusting report hashes.
            maxdiff=0.
            for name in ('BODY.npz','NATIVE.npz','COMMANDS.npz','POLICY.npz','SLIP.npz','NEURAL_FINAL.npz'):
                x,y=load(Path(prior['folder'])/name),load(Path(results[12]['folder'])/name)
                for key in x:
                    if x[key].dtype.kind in 'USO':a.check(f'g{gen}_control_identity_'+key,np.array_equal(x[key],y[key]));continue
                    err=float(np.max(np.abs(x[key].astype(float)-y[key].astype(float)))) if x[key].size else 0.;maxdiff=max(maxdiff,err)
            a.check(f'g{gen}_control_allarrays',maxdiff<=1e-9,max_abs_error=maxdiff);controls.append(maxdiff)
            feasible=[i for i,r in enumerate(results) if r['feasible']]
            pool=feasible or list(range(13));elites=sorted(pool,key=lambda i:(-results[i]['reward'],i))[:4]
            nextmean=.5*mean+.5*np.mean(values[elites],axis=0);nextstd=np.clip(.7*std+.3*np.std(values[elites],axis=0,ddof=0),.004,.05)
            improvements=[i for i in feasible if results[i]['reward']>inc_reward]
            winner=min(improvements,key=lambda i:(-results[i]['reward'],i)) if improvements else None
            if winner is not None:inc=values[winner].copy();inc_reward=results[winner]['reward'];inc_id=results[winner]['episode_id']
            a.same(f'g{gen}_elites',after['elites'],elites);a.same(f'g{gen}_mean_update',after['next_mean'],nextmean);a.same(f'g{gen}_std_update',after['next_std'],nextstd)
            a.check(f'g{gen}_incumbent_selection',after['incumbent_candidate']==winner);a.same(f'g{gen}_incumbent_theta',after['incumbent_theta'],inc);a.same(f'g{gen}_incumbent_reward',after['incumbent_reward'],inc_reward,1e-10)
            mean,std=nextmean,nextstd
            print(json.dumps(dict(event='verified_generation',generation=gen,episodes=len(a.episode_metrics),checks=len(a.checks))),flush=True)
        selected=load(OUTPUT/'SELECTED.npz');a.same('selected_theta',selected['theta'],inc);a.same('selected_training_reward',selected['training_reward'],inc_reward,1e-10);a.check('selected_episode',str(selected['selected_episode'])==inc_id)
        final_selected=audit_episode(a,OUTPUT/'evaluation/selected',assist,canonical);final_zero=audit_episode(a,OUTPUT/'evaluation/zero',assist,canonical)
        a.same('validation_selected_theta',final_selected['theta'],inc);a.same('validation_zero_theta',final_zero['theta'],np.zeros((6,4,7)))
        physical=sum(x['native_steps'] for x in a.episode_metrics);neural=sum(x['neural_steps'] for x in a.episode_metrics)
        a.check('all55_episodes',len(a.episode_metrics)==55);a.check('actualcalls',physical==5040000==study['actual_calls']['actual_native_steps'] and neural==126000==study['actual_calls']['actual_neural_steps'])
        a.check('no_training_or_physics_import',not any(x in sys.modules for x in ('mujoco','episode','optimize','scoring','search_rules')))
        report=dict(status='PASS_INDEPENDENT_STUDY_RECONSTRUCTION_NOT_WALKING',checks_passed=len(a.checks),checks=a.checks,verified_episodes=55,verified_original_physical_steps=physical,verified_original_neural_steps=neural,new_physical_steps=0,new_neural_steps=0,new_optimization_updates=0,selected_training_episode=inc_id,selected_training_reward=inc_reward,baseline_training_reward=baseline['reward'],validation_selected=final_selected,validation_zero=final_zero,incumbent_control_max_errors=controls,wall_s=time.monotonic()-started,verifier_sha256=sha(__file__),limitations='Pure saved-data reconstruction; no independent native dynamics replay here, parent owns final physical replay. Global score is training-only, cyclecoverage is proxy, no walking/biology acceptance.')
        out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','checks_passed','verified_episodes','wall_s')}),flush=True)
    except Exception:
        import traceback
        (HERE/'FAILED_INDEPENDENT_STUDY_VERIFICATION.json').write_text(json.dumps(dict(error=traceback.format_exc(),checks=a.checks,wall_s=time.monotonic()-started),indent=2)+'\n')
        raise

if __name__=='__main__':main()
