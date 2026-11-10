"""Synthetic checks for actual prefix PPO loss; no physical integration or training.

Core construction/capture verifies deployment identity without mj_step. Optimizer
steps below consume synthetic Gaussian data only. Temporary checkpoints cleaned.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
import prefix_ppo as pp
from models import ActorCritic, gaussian_kl, latent_log_prob
from recurrent import RecurrentPPOConfig, RecurrentPPOTrainer, replay_mean
from ppo import gae, save_checkpoint, load_checkpoint
from identity import HERE, sha256, atomic_json, capture

CHECKS=[]

def check(name,condition,**evidence):
    if not bool(condition): raise AssertionError(name+' '+str(evidence))
    CHECKS.append(dict(name=name,passed=True,**evidence))


def rejects(name,call,exception=(ValueError,FloatingPointError)):
    try: call()
    except exception: check(name,True); return
    raise AssertionError(name+' accepted invalid input')


def exact(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    return a==b


def new_model(halfspan=None):
    return ActorCritic('gru238',critic_obs_dim=255,dtype=torch.float64,
        actor_halfspan=np.ones(42) if halfspan is None else halfspan,
        logstd_bounds=(-5.,.3),initial_logstd=math.log(.02))


def synthetic(model,n=16,prefix=7):
    rng=np.random.default_rng(144)
    x=rng.normal(0,.03,(n,238)); cv=rng.normal(0,.03,(n,255)); cv[:,:238]=x
    cv[:,-1]=np.maximum(0,(n-np.arange(n))*.005)
    state=model.actor.initial_state(1); outputs=[]
    for t in range(n):
        out=model.act(x[t],cv[t],deterministic=t<prefix,state=state)
        state=out['state']; outputs.append(out)
    e=dict(obs=x,critic_obs=cv,reward=rng.normal(0,.1,n),next_value=np.zeros(n),
        terminated=np.zeros(n),truncated=np.r_[np.zeros(n-1),1.],learn_mask=np.arange(n)>=prefix,
        starts_at_reset=True,prefix_length=prefix)
    for key in ('mean','latent','logstd'): e[key]=np.stack([o[key][0].numpy() for o in outputs])
    for key in ('value','log_prob'): e[key]=np.array([float(o[key][0]) for o in outputs])
    e['next_value'][:-1]=e['value'][1:]
    e['next_value'][-1]=.25
    return e


def finite_differences(model,trainer,e):
    tensors=trainer.tensor_episodes([e])
    # Nonuniform signed advantages exercise actual PPO unclipped and clipped branches.
    tensors[0]['advantages'][7:]=torch.linspace(-1.,1.3,9)
    tensors[0]['log_prob'][7:]+=torch.tensor([-.3,.3,0,0,0,0,0,0,0.],dtype=torch.float64)
    loss,metrics=pp.masked_recurrent_loss(model,tensors,trainer.config)
    model.zero_grad(set_to_none=True);loss.backward()
    check('actual_PPO_clip_branches',float(metrics['clip_fraction'])>0)
    params=[('actor_head',model.actor.head.weight),('actor_encoder',model.actor.encoder.weight),
        ('actor_gain',model.actor.raw_gain),('logstd',model.raw_logstd),('critic',model.critic.layers[-1].weight)]
    for name,p in params:
        flat=p.detach().reshape(-1); grads=p.grad.reshape(-1)
        index=int(grads.abs().argmax()); analytic=float(grads[index]); original=float(flat[index]); eps=1e-6
        with torch.no_grad():
            flat[index]=original+eps; plus=float(pp.masked_recurrent_loss(model,tensors,trainer.config)[0])
            flat[index]=original-eps; minus=float(pp.masked_recurrent_loss(model,tensors,trainer.config)[0])
            flat[index]=original
        numeric=(plus-minus)/(2*eps)
        check('FD_'+name,abs(analytic-numeric)<2e-5*(1+abs(numeric)),analytic=analytic,numeric=numeric)


def loss_mask_checks(model,trainer,e):
    base=trainer.tensor_episodes([e])[0]
    changed=deepcopy(e)
    for key in ('reward','value','next_value'): changed[key][:7]=1e6
    tensor=trainer.tensor_episodes([changed])[0]
    check('prefix_excluded_from_GAE',torch.equal(base['advantages'],tensor['advantages']) and torch.equal(base['returns'],tensor['returns']))
    baseline=pp.masked_recurrent_loss(model,[base],trainer.config)[0]
    mutated=deepcopy(base)
    for key in ('latent','log_prob','mean','logstd','critic_obs','advantages','returns'):
        mutated[key][:7]=1e6
    actual=pp.masked_recurrent_loss(model,[mutated],trainer.config)[0]
    check('prefix_counterfactual_density_targets_not_objective',torch.equal(baseline,actual))
    pp.MaskedPrefixTrainer.normalize_advantages([base])
    a=base['advantages'][7:]
    check('normalize_auto_only',abs(float(a.mean()))<1e-12 and abs(float(a.std(unbiased=False))-1)<1e-6 and not base['advantages'][:7].any())
    # Full-history gradients reach prefix observations through neural memory.
    differentiable=trainer.tensor_episodes([e])[0]
    differentiable['obs'].requires_grad_(True)
    loss=pp.masked_recurrent_loss(model,[differentiable],trainer.config)[0]
    grad=torch.autograd.grad(loss,differentiable['obs'])[0]
    check('full_BPTT_prefix_gradient_nonzero',float(grad[:7].abs().sum())>1e-10,prefix_gradient=float(grad[:7].abs().sum()))
    # Even a history only change modifies the actual autonomous output.
    altered=deepcopy(differentiable); altered['obs']=altered['obs'].detach().clone(); altered['obs'][:7,0]+=.5
    check('prefix_observation_history_affects_auto',not torch.equal(replay_mean(model,differentiable)[7:],replay_mean(model,altered)[7:]))
    check('replay_from_zero_exact',max(trainer.check_behavior_replay(trainer.tensor_episodes([e])).values())<1e-10)
    check('replay_prefix_count',len(e['obs'])==16 and int(e['learn_mask'].sum())==9)
    for part in ('prefix','auto'):
        stale=deepcopy(e);stale['mean'][0 if part=='prefix' else 9,0]+=.1
        rejects('stale_'+part+'_rejected',lambda:trainer.update([stale]))
    for name,mutate in (
        ('nonmonotonic',lambda a:a['learn_mask'].__setitem__(0,True)),
        ('no_auto',lambda a:a.update(prefix_length=16,learn_mask=np.zeros(16))),
        ('no_prefix',lambda a:a.update(prefix_length=0,learn_mask=np.ones(16))),
        ('boundary_inside',lambda a:a['terminated'].__setitem__(3,1)),
        ('missing_reset',lambda a:a.update(starts_at_reset=False)),
        ('nan',lambda a:a['obs'].__setitem__((4,0),np.nan))):
        bad=deepcopy(e);mutate(bad)
        rejects('reject_'+name,lambda:trainer.update([bad]))
    # Hand joint analytic KL and masks, with arbitrary excluded prefix statistics.
    b=trainer.tensor_episodes([e])[0]
    with torch.no_grad(): model.actor.head.bias[0]+=.001
    new=replay_mean(model,b).detach();ls=model.logstd().detach()
    hand=(ls-b['logstd'][7:]+(b['logstd'][7:].exp().square()+(b['mean'][7:]-new[7:]).square())/(2*ls.exp().square())-.5).sum(-1).mean()
    check('masked_joint_KL_hand_formula',abs(trainer.rollout_kl([b])-float(hand))<1e-12)
    before=trainer.rollout_kl([b]);b['mean'][:7]+=100;b['logstd'][:7]+=100
    check('prefix_KL_excluded',trainer.rollout_kl([b])==before)
    with torch.no_grad(): model.actor.head.bias[0]-=.001


def transactional_checks():
    model=new_model();tr=pp.MaskedPrefixTrainer(model,replace(pp.config(),epochs=1))
    initial=synthetic(model)
    report=tr.update([initial])
    check('synthetic_actual_update_accepted',report['accepted_steps']>0 and report['samples']==9 and report['replay_rows']==16)
    check('both_Adam_states_initialized',len(tr.actor_optimizer.state)>0 and len(tr.critic_optimizer.state)>0)
    snap=tr.snapshot(); e=synthetic(model)
    original=tr.rollout_kl; calls=[0]
    def reject(b):
        calls[0]+=1
        return original(b) if calls[0]==1 else 1.
    tr.rollout_kl=reject
    out=tr.update([e])
    check('forced_joint_KL_rejection',out['stopped'] and out['steps'][0]['attempts']==5)
    check('transaction_restores_model_both_Adam_counters_LRs',exact(snap,tr.snapshot()))
    tr.rollout_kl=original
    # Explicit nonfinite candidate is rejected by same transactional path.
    calls[0]=0
    def nan_reject(b):
        calls[0]+=1
        return original(b) if calls[0]==1 else float('nan')
    tr.rollout_kl=nan_reject
    tr.update([e])
    check('nonfinite_candidate_transaction',exact(snap,tr.snapshot()))
    tr.rollout_kl=original
    # Force one rejection, then accept and retain exactly half both LRs.
    calls[0]=0
    def once(b):
        calls[0]+=1
        return 1. if calls[0]==2 else original(b)
    tr.rollout_kl=once
    out=tr.update([e])
    check('retry_half_LRs_persist',out['steps'][0]['attempts']==2 and out['steps'][0]['accepted']
        and tr.actor_optimizer.param_groups[0]['lr']==snap['actor']['param_groups'][0]['lr']/2
        and tr.critic_optimizer.param_groups[0]['lr']==snap['critic']['param_groups'][0]['lr']/2)


class MockCore:
    def __init__(self):
        self.th=0;self.initial_heading=np.array([1.,0.]);self.n=0
        self.d=SimpleNamespace(time=1.,qpos=np.zeros(3),qvel=np.zeros(3),subtree_com=np.array([[0.,0.,1.3]]))
    def obs238(self,previous,target_speed):
        x=np.zeros(238);x[0]=self.n*.001;x[195:237]=previous;x[-1]=target_speed;return x
    def critic_extras(self): return np.zeros(16)
    def com_velocity(self): return np.array([.075,0.,0.])
    def measure(self):
        return dict(loads=np.full(6,pp.W/6),speed=np.zeros(6),nonfoot=0.,padz=np.zeros(6),foot=np.zeros((6,3)))


class MockTeacher:
    def __init__(self): self.calls=[]
    def action(self,c,time):
        if len(self.calls)>=2: raise AssertionError('teacher queried after prefix')
        self.calls.append(time);return np.full(42,.01*(len(self.calls))),{}


class MockTrace:
    def __init__(self): self.calls=[]
    def tick(self,c,a,source):
        self.calls.append((a.copy(),source,c.d.time));c.n+=1;c.d.time+=.001;c.d.qpos[0]+=.000075
        return dict(loads=np.full(6,pp.W/6),speed=np.zeros(6),nonfoot=0.)


def mock_collection_checks():
    model=new_model();c=MockCore();t=MockTeacher();trace=MockTrace()
    act_calls=[];original_act=model.act
    def counted_act(*args,**kwargs):
        act_calls.append(kwargs['deterministic'])
        return original_act(*args,**kwargs)
    model.act=counted_act
    observer=pp.sw.tracker(c)
    ep,report=pp.collect_episode(model,c,t,trace,autonomous_ms=10,prefix_ms=10,observer=observer)
    check('mock_cycle_observer_starts_at_handover_1ms',abs(observer.time-c.d.time)<1e-12 and report['cycles'] is not None)
    check('mock_one_5ms_decision',len(trace.calls)==20 and np.array_equal(ep['held_ms'],[5,5,5,5]))
    check('mock_no_teacher_after_prefix',len(t.calls)==2 and report['autonomous_teacher_queries']==0)
    check('mock_actual_previous_action',np.array_equal(ep['obs'][1:,195:237],ep['action'][:-1]))
    check('mock_source_flags',np.array_equal(ep['control_source'],[2,2,3,3]))
    check('mock_remaining_autonomous_time',np.allclose(ep['critic_obs'][:,-1],[.01,.01,.01,.005]) and ep['next_critic_obs'][-1,-1]==0)
    check('mock_no_actor_bootstrap_step',ep['truncated'][-1] and not ep['terminated'].any() and act_calls==[True,True,False,False])
    with torch.no_grad():
        actual_value=float(model.value(torch.as_tensor(ep['next_critic_obs'][-1],dtype=torch.float64)))
    check('mock_bootstrap_critic_actual_post_state',ep['next_value'][-1]==actual_value)
    x=ep['obs'][-1];v=ep['critic_obs'][-1].copy();state=model.actor.initial_state(1)
    first=original_act(x,v,deterministic=True,state=state)['mean'];v[-1]+=10
    second=original_act(x,v,deterministic=True,state=state)['mean']
    check('actor_excludes_critic_remaining_time',torch.equal(first,second))
    check('mock_physical_state_copies',not np.shares_memory(ep['qpos_after'],c.d.qpos) and abs(ep['time_after'][-1]-1.02)<1e-12)
    trainer=pp.MaskedPrefixTrainer(model)
    check('mock_full_history_replay',max(trainer.check_behavior_replay(trainer.tensor_episodes([ep])).values())<1e-10)
    check('mock_hidden_evidence_only_zero_start',not ep['behavior_hidden_before'][0].any() and
        np.array_equal(ep['behavior_hidden_before'][1:],ep['behavior_hidden_after'][:-1]))
    check('mock_prefix_counterfactual_not_executed',not np.array_equal(ep['action'][:2],np.tanh(ep['latent'][:2])))
    for i in range(4):
        chunk=trace.calls[i*5:(i+1)*5]
        check('mock_action_held_'+str(i),all(np.array_equal(a,ep['action'][i]) for a,_,_ in chunk))
    # Hard fail during prefix retained, with no fabricated autonomous samples.
    class FallTrace(MockTrace):
        def tick(self,c,a,source):
            m=super().tick(c,a,source);c.d.subtree_com[0,2]=1.;return m
    bad,rep=pp.collect_episode(model,MockCore(),MockTeacher(),FallTrace(),10,10)
    check('failed_prefix_retained_no_NN_samples',rep['prefix_failed'] and len(bad['obs'])==1 and not bad['learn_mask'].any() and bad['terminated'][-1])


def reward_and_gae_checks():
    z=np.zeros(42);args=(np.array([.075,0.,0.]),np.array([1.,0.]),np.full(6,pp.W/6),np.zeros(6),1.3,0.,pp.W,z,z)
    rate,parts=pp.reward_rate(*args)
    check('reward_exact_target_progress',rate==3. and sum(parts.values())==rate)
    check('reward_dt_integral',abs(sum(.001*rate for _ in range(5))-.005*rate)<1e-15)
    args=list(args);args[3]=np.ones(6)*1.5
    floor,parts=pp.reward_rate(*args)
    check('raw_slip_unclipped_total_floor_explicit',parts['slip']<-100 and floor==-100 and parts['floor_adjustment']>0)
    check('slip_no_achieved_speed_denominator',abs(parts['slip']+800.)<1e-10)
    args[0]=np.zeros(3);_,stationary=pp.reward_rate(*args)
    check('slip_at_zero_forward_same_physical_cost',stationary['slip']==parts['slip'])
    args[0]=np.array([np.nan,0.,0.]);rejects('reward_nan_rejected',lambda:pp.reward_rate(*args))
    g=pp.config().gamma
    check('absorbing_longer_early_failure_worse',pp.absorbing_penalty(100,g)<pp.absorbing_penalty(10,g)<0)
    check('absorbing_discount_exact',abs(pp.absorbing_penalty(8,g)+.5*sum(g**j for j in range(1,9)))<1e-14)
    check('absorbing_zero_remaining',pp.absorbing_penalty(0,g)==0)
    # At bounded -100 rate, omitted continuation has exactly the absorbing value.
    check('absorption_equals_worst_bounded_continuation',abs(pp.absorbing_penalty(10,g)-sum(g**j*(-100*.005) for j in range(1,11)))<1e-14)
    a,r=gae([1.],[2.],[7.],[0],[1],.9,.8)
    check('GAE_truncation_actual_successor',np.allclose(a,[5.3]) and np.allclose(r,[7.3]))
    a,r=gae([1.],[2.],[7.],[1],[0],.9,.8)
    check('GAE_terminal_zero_bootstrap',np.allclose(a,[-1.]) and np.allclose(r,[1.]))
    a,r=gae([1.,2.],[.5,.7],[.7,4.],[0,0],[0,1],.9,.8)
    hand_last=2+.9*4-.7
    check('GAE_two_row_recursion',np.allclose(a,[1+.9*.7-.5+.9*.8*hand_last,hand_last]))
    model=new_model()
    with torch.no_grad(): model.raw_logstd[0]=-6.;model.raw_logstd[1]=1.
    check('pure_hard_logstd_bounds',model.logstd()[0]==-5 and model.logstd()[1]==.3)
    check('standard_config_only_fields',RecurrentPPOConfig(**asdict(pp.config()))==pp.config())
    check('trainer_default_5ms_discount',pp.MaskedPrefixTrainer(new_model()).config==pp.config())
    check('CLI_train_contract',pp.parser().parse_args(['train','--checkpoint','/abs/input.pt','--out','/abs/new','--updates','3','--autonomous-ms','6000','--seed','20261401']).seed==20261401)
    check('CLI_eval_contract',pp.parser().parse_args(['eval','--checkpoint','/abs/input.pt','--out','/abs/new']).mode=='eval')


def full_length_checks():
    model=new_model();trainer=pp.MaskedPrefixTrainer(model,pp.config())
    episode=synthetic(model,n=2160,prefix=960)
    tensors=trainer.tensor_episodes([episode])
    errors=trainer.check_behavior_replay(tensors)
    check('full_960_prefix_1200_autonomous_replay',max(errors.values())<1e-10,
        rows=2160,prefix=960,autonomous=1200)
    loss,metrics=pp.masked_recurrent_loss(model,tensors,trainer.config)
    loss.backward()
    check('full_length_loss_counts_only_1200',metrics['samples']==1200 and torch.isfinite(loss))
    check('full_length_BPTT_finite',all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


def compatibility_checks():
    # No physical tick is executed: construct compiled actual H and provenance only.
    core=pp.Core(adhesion=False)
    model=new_model(pp.sw.physical_halfspan(core));tr=pp.MaskedPrefixTrainer(model,pp.config())
    ident=capture(core,dict(mode='synthetic_prefix_checkpoint_compatibility',physical_integration_ms=0),extra_paths=pp.SOURCE_FILES)
    with tempfile.TemporaryDirectory(prefix='prefix-ppo-check-') as directory:
        path=Path(directory)/'synthetic.pt'
        save_checkpoint(path,tr,ident,dict(kind='prefix_ppo',prefix_ppo=dict(test_only=True)))
        original=RecurrentPPOTrainer(new_model(pp.sw.physical_halfspan(core)),RecurrentPPOConfig(**asdict(pp.config())))
        load_checkpoint(path,original,ident,restore_rng=False)
        check('original_trainer_checkpoint_load_exact',exact(tr.snapshot(),original.snapshot()))
        loaded,payload=pp.sw.load_student(path,core)
        check('original_sw_load_student_exact',exact(model.state_dict(),loaded.state_dict()) and payload['model_identity']['logstd_bounds']==(-5.,.3))
        fresh,parent=pp.fresh_from_bc(path,core)
        check('BC_mean_gain_preserved_model_only',exact(model.actor.state_dict(),fresh.model.actor.state_dict()))
        check('fresh_both_Adam_zero',not fresh.actor_optimizer.state and not fresh.critic_optimizer.state and fresh.updates==0 and fresh.accepted_steps==0)
        check('BC_std_reset_metadata',torch.allclose(fresh.model.logstd().exp(),torch.full((42,),.02,dtype=torch.float64)) and fresh.model.identity()['logstd_bounds']==(-5.,.3))
        check('actual_physical_H_constructor',np.array_equal(fresh.model.actor.initial_halfspan.numpy(),pp.sw.physical_halfspan(core)))


def main():
    torch.set_num_threads(1);torch.manual_seed(417);np.random.seed(417)
    model=new_model();trainer=pp.MaskedPrefixTrainer(model,pp.config());e=synthetic(model)
    finite_differences(model,trainer,e)
    # Rebuild exact behavior after finite-difference writes to avoid roundoff.
    e=synthetic(model)
    loss_mask_checks(model,trainer,e)
    transactional_checks();mock_collection_checks();reward_and_gae_checks();full_length_checks();compatibility_checks()
    result=dict(all_passed=True,checks=CHECKS,check_count=len(CHECKS),
        source_hashes={str(p.resolve()):sha256(p) for p in pp.SOURCE_FILES},
        environment=dict(executable=__import__('sys').executable,torch=torch.__version__,threads=torch.get_num_threads()),
        physical_integration_ms=0,actual_training=False,synthetic_optimizer_steps=True,
        interpretation='numerical/software validation only; no neural walking or physical gate claim')
    atomic_json(HERE/'PREFIX_PPO_CHECKS.json',result)
    print(json.dumps(dict(all_passed=True,check_count=len(CHECKS),physical_integration_ms=0)),flush=True)


if __name__=='__main__':main()
