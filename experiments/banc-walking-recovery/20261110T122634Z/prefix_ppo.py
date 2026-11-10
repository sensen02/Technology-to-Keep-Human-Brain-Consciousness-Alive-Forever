"""S2/S5 conditional PPO refinement: physical teacher history, autonomous-only loss.

No execution on import. Full BPTT starts from zero at physical reset, after a
1s unobserved zero-command settle. Teacher counterfactual policy densities are
replay evidence ONLY. Native units; absolute SI NOT_ESTABLISHED. No gait claim.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import traceback
import numpy as np
import torch
from models import ActorCritic, latent_log_prob, gaussian_kl
from recurrent import RecurrentPPOConfig, RecurrentPPOTrainer, replay_mean
from ppo import gae, save_checkpoint
import student_wave as sw
from identity import HERE, sha256, atomic_json, capture, verify
from physics import Core, W
from static_teacher import StaticTeacher

OWN_FILES = [HERE/'prefix_ppo.py', HERE/'check_prefix_ppo.py']
SOURCE_FILES = sw.SOURCE_FILES + OWN_FILES
REWARD_CONFIG = dict(target_speed=.075, forward=3., speed_error=10., slip=2.,
    com_floor=1.2, com_scale=.1, com_cost=10., nonfoot=10., support_floor=.9,
    support_scale=.1, support_cost=10., action=.002, slew=.01, slew_scale=.1,
    rate_dt_s=.001, decision_dt_s=.005, absorbing_rate=100., rate_floor=-100., cycle_bonus=0.)


def config():
    return RecurrentPPOConfig(gamma=math.exp(-.005/1.), gae_lambda=math.exp(-.005/.2),
        epochs=4, actor_lr=3e-5, critic_lr=1e-4, clip=.1, entropy_coef=0.,
        max_grad_norm=.5, target_kl=.03, backtracks=4, episodes_per_minibatch=1)


def mask_contract(episode, n):
    mask = np.asarray(episode['learn_mask'])
    prefix = episode['prefix_length']
    if isinstance(prefix, bool) or int(prefix) != prefix or not 0 < int(prefix) < n:
        raise ValueError('positive teacher prefix and positive autonomous suffix required')
    prefix = int(prefix)
    expected = np.arange(n) >= prefix
    if mask.shape != (n,) or not np.array_equal(mask, expected):
        raise ValueError('learn_mask must be exactly monotonic 0-prefix / 1-autonomous')
    return prefix


def masked_recurrent_loss(model, episodes, cfg):
    count = sum(int(e['learn_mask'].sum()) for e in episodes)
    if count < 1:
        raise ValueError('empty autonomous minibatch')
    actor_sum = model.raw_logstd.new_zeros(())
    critic_sum = actor_sum.clone(); clip_sum = actor_sum.clone(); ratio_sum = actor_sum.clone()
    ls = model.logstd()
    entropy = (ls + .5*math.log(2*math.pi*math.e)).sum()
    for e in episodes:
        # Replay ALL observations with current weights; slice BEFORE densities or
        # critic/objective calculation, so excluded extreme prefix values cannot leak.
        mean = replay_mean(model, e)
        keep = e['learn_mask'].bool()
        ratio = (latent_log_prob(e['latent'][keep], mean[keep], ls)-e['log_prob'][keep]).exp()
        adv = e['advantages'][keep]
        objective = torch.minimum(ratio*adv, ratio.clamp(1-cfg.clip, 1+cfg.clip)*adv)
        actor_sum = actor_sum - objective.sum()
        critic_sum = critic_sum + (model.value(e['critic_obs'][keep])-e['returns'][keep]).square().sum()
        clip_sum = clip_sum + ((ratio-1).abs() > cfg.clip).to(ratio.dtype).sum()
        ratio_sum = ratio_sum + ratio.sum()
    actor = actor_sum/count-cfg.entropy_coef*entropy
    critic = cfg.value_coef*critic_sum/count
    return actor+critic, dict(actor_loss=actor, critic_loss=critic, entropy=entropy,
        clip_fraction=clip_sum/count, ratio_mean=ratio_sum/count, samples=count)


class MaskedPrefixTrainer(RecurrentPPOTrainer):
    """Standard snapshot/config API; own loss, mask validation and replay semantics."""
    def __init__(self, model, config=None):
        super().__init__(model, config if config is not None else globals()['config']())

    def tensor_episodes(self, episodes):
        if not episodes:
            raise ValueError('empty recurrent rollout')
        result = []
        p = self.model.raw_logstd
        vectors = dict(obs=238, critic_obs=self.model.critic_obs_dim, latent=42, mean=42, logstd=42)
        scalars = ('log_prob','value','reward','terminated','truncated','next_value','learn_mask')
        for episode in episodes:
            if episode.get('starts_at_reset') is not True:
                raise ValueError('starts_at_reset=True required')
            e = {k:torch.as_tensor(episode[k],dtype=p.dtype,device=p.device).detach().clone()
                 for k in (*vectors,*scalars)}
            n = len(e['obs'])
            prefix = mask_contract(episode,n)
            for key,width in vectors.items():
                if e[key].shape != (n,width): raise ValueError('invalid shape '+key)
            for key in scalars:
                if e[key].shape != (n,): raise ValueError('invalid shape '+key)
            if not all(torch.isfinite(v).all() for v in e.values()):
                raise ValueError('nonfinite episode')
            for key in ('terminated','truncated'):
                if not ((e[key]==0)|(e[key]==1)).all(): raise ValueError('nonboolean boundary')
            boundary = e['terminated'].bool()|e['truncated'].bool()
            if not boundary[-1] or boundary[:-1].any() or (e['terminated']*e['truncated']).any():
                raise ValueError('flags only at final physical boundary')
            # Prefix rewards/values never enter GAE. This is a fresh on-policy
            # autonomous suffix with exact actual reset observation history.
            adv,ret = gae(*(e[k][prefix:].cpu().numpy() for k in
                ('reward','value','next_value','terminated','truncated')),self.config.gamma,self.config.gae_lambda)
            e['advantages'] = torch.zeros(n,dtype=p.dtype,device=p.device)
            e['returns'] = torch.zeros_like(e['advantages'])
            e['advantages'][prefix:] = torch.as_tensor(adv,dtype=p.dtype,device=p.device)
            e['returns'][prefix:] = torch.as_tensor(ret,dtype=p.dtype,device=p.device)
            # Optional supplied targets are useful for independent numerical tests;
            # collector deliberately supplies none. Prefix supplied entries ignored.
            if 'advantages' in episode or 'returns' in episode:
                for key in ('advantages','returns'):
                    a = torch.as_tensor(episode[key],dtype=p.dtype,device=p.device)
                    if a.shape != (n,) or not torch.isfinite(a).all(): raise ValueError('invalid '+key)
                    e[key][prefix:] = a[prefix:]
            e['prefix_length'] = prefix
            result.append(e)
        return result

    @torch.no_grad()
    def rollout_kl(self, episodes):
        numerator = self.model.raw_logstd.new_zeros(())
        count = 0
        for e in episodes:
            keep = e['learn_mask'].bool()
            means = replay_mean(self.model,e)
            numerator += gaussian_kl(e['mean'][keep],e['logstd'][keep],means[keep],self.model.logstd()).sum()
            count += int(keep.sum())
        if not count: raise ValueError('empty autonomous rollout')
        return float(numerator/count)

    @torch.no_grad()
    def check_behavior_replay(self, episodes):
        errors = dict(mean=0.,logstd=0.,log_prob=0.)
        for e in episodes:
            mean = replay_mean(self.model,e)
            ls = self.model.logstd().expand_as(mean)
            logp = latent_log_prob(e['latent'],mean,ls)
            for key,current in (('mean',mean),('logstd',ls),('log_prob',logp)):
                difference = (e[key]-current).abs()
                if not torch.isfinite(difference).all(): raise ValueError('nonfinite behavior replay')
                errors[key] = max(errors[key],float(difference.max()))
        if any(v > self.config.replay_tolerance for v in errors.values()):
            raise ValueError(f'stale policy or invalid zero-state history: {errors}')
        return errors

    @staticmethod
    def normalize_advantages(episodes):
        all_adv = torch.cat([e['advantages'][e['learn_mask'].bool()] for e in episodes])
        mean,scale = all_adv.mean(),all_adv.std(unbiased=False)+1e-8
        for e in episodes:
            keep = e['learn_mask'].bool()
            e['advantages'][keep] = (e['advantages'][keep]-mean)/scale
            e['advantages'][~keep] = 0

    def update(self, episodes):
        c = self.config
        episodes = self.tensor_episodes(episodes)
        replay_errors = self.check_behavior_replay(episodes)
        initial_kl = self.rollout_kl(episodes)
        if not math.isfinite(initial_kl) or initial_kl > c.target_kl:
            raise ValueError('initial recurrent KL exceeds budget')
        if c.normalize_advantage: self.normalize_advantages(episodes)
        total_samples = sum(int(e['learn_mask'].sum()) for e in episodes)
        rows=[]; stopped=False
        for epoch in range(c.epochs):
            order = torch.randperm(len(episodes)).tolist()
            for start in range(0,len(order),c.episodes_per_minibatch):
                minibatch = [episodes[i] for i in order[start:start+c.episodes_per_minibatch]]
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss,metrics = masked_recurrent_loss(self.model,minibatch,c)
                sample_weight_scale = (len(episodes)/len(minibatch))*(metrics['samples']/total_samples)
                loss = loss*sample_weight_scale
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite recurrent loss')
                loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(self.model.actor_parameters(),c.max_grad_norm)
                critic_grad = torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(),c.max_grad_norm)
                if not torch.isfinite(actor_grad) or not torch.isfinite(critic_grad):
                    raise FloatingPointError('nonfinite recurrent gradient')
                snapshot = self.snapshot()
                accepted=False
                # Exact original transaction: restore BOTH Adam states/counters,
                # apply the SAME clipped gradients, halve both LR each retry.
                for attempt in range(c.backtracks+1):
                    self.restore(snapshot)
                    for optimizer in (self.actor_optimizer,self.critic_optimizer):
                        for group in optimizer.param_groups: group['lr'] *= .5**attempt
                        optimizer.step()
                    candidate_kl=self.rollout_kl(episodes)
                    if math.isfinite(candidate_kl) and candidate_kl <= c.target_kl:
                        accepted=True; self.accepted_steps+=1; break
                if not accepted: self.restore(snapshot); stopped=True
                row={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in metrics.items()}
                row.update(epoch=epoch,episodes=len(minibatch),sample_weight_scale=sample_weight_scale,
                    accepted=accepted,attempts=attempt+1,candidate_kl=candidate_kl if math.isfinite(candidate_kl) else None,kl=self.rollout_kl(episodes),
                    actor_grad_before=float(actor_grad),critic_grad_before=float(critic_grad),
                    actor_grad_after=min(float(actor_grad),c.max_grad_norm),critic_grad_after=min(float(critic_grad),c.max_grad_norm))
                rows.append(row)
                if stopped: break
            if stopped: break
        if any(row['accepted'] for row in rows): self.updates+=1
        with torch.no_grad():
            returns=torch.cat([e['returns'][e['learn_mask'].bool()] for e in episodes])
            predicted=torch.cat([self.model.value(e['critic_obs'][e['learn_mask'].bool()]) for e in episodes])
            variance=returns.var(unbiased=False)
            ev=float(1-(returns-predicted).var(unbiased=False)/variance) if variance>0 else None
            saturation=torch.cat([e['latent'][e['learn_mask'].bool()] for e in episodes]).tanh().abs()>.99
        kl=self.rollout_kl(episodes)
        return dict(steps=rows,stopped=stopped,kl=kl,kl_per_dim=kl/42,explained_variance=ev,
            std_min=float(self.model.logstd().detach().exp().min()),std_max=float(self.model.logstd().detach().exp().max()),
            saturation_fraction=float(saturation.to(returns.dtype).mean()),
            actor_lr=self.actor_optimizer.param_groups[0]['lr'],critic_lr=self.critic_optimizer.param_groups[0]['lr'],
            accepted_steps=self.accepted_steps,updates=self.updates,replay_errors=replay_errors,
            replay_rows=sum(len(e['obs']) for e in episodes),episodes=len(episodes),samples=total_samples,
            prefix_rows=sum(e['prefix_length'] for e in episodes),bptt='full_reset_prefix_and_autonomous_current_weights')


def reward_rate(velocity, heading, loads, speeds, comz, nonfoot, weight, action, previous_policy_action):
    """Pure actual physical 1ms rate; no phase, teacher, or achieved-speed denominator."""
    arrays=[np.asarray(x,dtype=float) for x in (velocity,heading,loads,speeds,action,previous_policy_action)]
    if not all(np.isfinite(x).all() for x in arrays) or not np.isfinite([comz,nonfoot,weight]).all() or weight<=0:
        raise FloatingPointError('nonfinite reward measurement')
    vel,heading,loads,speeds,action,previous=arrays
    v=float(vel[:2]@heading)
    parts=dict(forward=3*v/.075,speed_error=-10*((v-.075)/.075)**2,
        slip=-2*float(np.sum((loads/weight)*(speeds/.075)**2)),
        height=-10*(max(0.,1.2-comz)/.1)**2,nonfoot=-10*nonfoot/weight,
        support=-10*(max(0.,.9-float(loads.sum()/weight))/.1)**2,
        action=-.002*float(np.mean(action**2)),slew=-.01*float(np.mean(((action-previous)/.1)**2)))
    if not np.isfinite(list(parts.values())).all(): raise FloatingPointError('nonfinite reward rate')
    raw = sum(parts.values())
    # Explicit TOTAL rate floor makes the remaining-budget absorbing penalty a
    # conservative continuation. Raw physical slip term remains unmodified/logged.
    parts['floor_adjustment'] = max(-100.,raw)-raw
    return max(-100.,raw),parts


def absorbing_penalty(remaining_steps,gamma):
    if int(remaining_steps)!=remaining_steps or remaining_steps<0: raise ValueError('invalid remaining budget')
    return -100*.005*sum(gamma**j for j in range(1,int(remaining_steps)+1))


def physical_failure(c,m):
    values=[c.d.qpos,c.d.qvel,c.d.subtree_com[c.th],m['loads'],m['speed'],np.asarray(m['nonfoot'])]
    if not all(np.isfinite(v).all() for v in values): return 'nonfinite physical state'
    if float(c.d.subtree_com[c.th][2])<1.1: return 'COM below 1.1'
    if float(m['nonfoot'])>.2*W: return 'nonfoot above 0.2W'
    return None


def collect_episode(model,core,teacher,trace,autonomous_ms=6000,prefix_ms=4800,observer=None):
    """Core is at actual reset; initial zero hidden precedes unobserved settle.

    Mock tests may pass tiny durations and a Core-compatible object. Production
    caller fixes 4800ms prefix and 1000ms settle. No teacher call in auto branch.
    """
    if autonomous_ms<5 or prefix_ms<5 or autonomous_ms%5 or prefix_ms%5:
        raise ValueError('durations require positive multiples of 5ms')
    previous=np.zeros(42); state=model.actor.initial_state(1)
    rows=[]; termination=None; teacher_queries=0; auto_ticks=0
    events=[]; components={}; penalty=0.; floored_ticks=0; raw_rate_sum=0.
    for k in range((prefix_ms+autonomous_ms)//5):
        prefix=k<prefix_ms//5
        auto_index=k-prefix_ms//5
        if not prefix and auto_index==0 and observer is not None:
            # The autonomous diagnostic requires consecutive 1ms timestamps.
            # Prefix was not part of this observer; start at actual handover.
            observer.reset(start_time=float(core.d.time))
        remaining=autonomous_ms*.001 if prefix else (autonomous_ms-auto_index*5)*.001
        x,v=sw.observations(core,previous,remaining)
        before=dict(qpos_before=core.d.qpos.copy(),qvel_before=core.d.qvel.copy(),time_before=float(core.d.time),
            behavior_hidden_before=state.cpu().numpy()[0].copy())
        output=model.act(x,v,deterministic=prefix,state=state)
        state=output['state']
        if prefix:
            action,_=teacher.action(core,k*.005); teacher_queries+=1; source=2
        else:
            action=output['action'][0].cpu().numpy(); source=3
        action=sw.bounded_action(action)
        r=0.; parts_sum={}; held_ms=0
        # Previous DECISION command stays fixed for all five physical milliseconds.
        for _ in range(5):
            m=trace.tick(core,action,source); held_ms+=1
            termination=physical_failure(core,m)
            if not prefix:
                auto_ticks+=1
                if termination != 'nonfinite physical state':
                    rate,parts=reward_rate(core.com_velocity(),core.initial_heading,m['loads'],m['speed'],
                        float(core.d.subtree_com[core.th][2]),float(m['nonfoot']),W,action,previous)
                    r+=.001*rate
                    raw_rate_sum += .001*(rate-parts['floor_adjustment'])
                    floored_ticks += int(parts['floor_adjustment']>0)
                    for key,value in parts.items():
                        parts_sum[key]=parts_sum.get(key,0.)+.001*value
                        components[key]=components.get(key,0.)+.001*value
                if observer is not None: events.extend(observer.measure(core)['events'])
            if termination: break
        last=k==(prefix_ms+autonomous_ms)//5-1
        if termination and not prefix:
            penalty=absorbing_penalty(autonomous_ms//5-auto_index-1,config().gamma)
            r+=penalty
        if termination:
            successor_v=np.zeros(255); next_value=0.
        else:
            _,successor_v=sw.observations(core,action,max(0.,remaining-held_ms*.001) if not prefix else autonomous_ms*.001)
            with torch.no_grad(): next_value=float(model.value(torch.as_tensor(successor_v,dtype=model.raw_logstd.dtype)))
        row=dict(obs=x.copy(),critic_obs=v.copy(),action=action.copy(),previous_actual_action=previous.copy(),
            next_critic_obs=successor_v.copy(),reward=r,terminated=bool(termination),truncated=bool(last and not termination),
            next_value=next_value,learn_mask=int(not prefix),control_source=source,held_ms=held_ms,
            qpos_after=core.d.qpos.copy(),qvel_after=core.d.qvel.copy(),time_after=float(core.d.time),
            behavior_hidden_after=state.cpu().numpy()[0].copy(),**before)
        for key in ('latent','mean','logstd'): row[key]=output[key][0].cpu().numpy().copy()
        for key in ('log_prob','value'): row[key]=float(output[key][0])
        rows.append(row); previous=action.copy()
        if termination: break
    episode={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    episode.update(starts_at_reset=True,prefix_length=min(prefix_ms//5,len(rows)))
    actual_auto=int(episode['learn_mask'].sum())
    if teacher_queries!=min(len(rows),prefix_ms//5) or np.any(episode['control_source'][episode['learn_mask'].astype(bool)]!=3):
        raise RuntimeError('teacher/source accounting mismatch')
    cycles=sw.cycle_report(observer,events) if observer is not None else None
    report=dict(prefix_rows=min(len(rows),prefix_ms//5),autonomous_rows=actual_auto,teacher_queries=teacher_queries,
        autonomous_teacher_queries=0,autonomous_1ms_ticks=auto_ticks,termination=termination,
        reward=float(episode['reward'][episode['learn_mask'].astype(bool)].sum()),reward_components=components,
        absorbing_penalty=penalty,raw_reward_integral=raw_rate_sum,floor_ticks=floored_ticks,
        floor_fraction=floored_ticks/max(1,auto_ticks),cycles=cycles,starts_at_reset=True,initial_actor_hidden_zero=True,
        settle_actor_forwards=0,prefix_density='deterministic mean counterfactual; not executed; replay only',
        prefix_failed=actual_auto==0)
    return episode,report


def fresh_from_bc(parent,core):
    """MODEL ONLY: preserve all actor mean/gain parameters, explicitly reset std."""
    old,payload=sw.load_student(parent,core)
    if old.architecture!='gru238': raise ValueError('requires SlowGRUv1 BC')
    model=ActorCritic('gru238',critic_obs_dim=255,dtype=torch.float64,
        actor_halfspan=sw.physical_halfspan(core),initial_logstd=math.log(.02),logstd_bounds=(-5.,.3))
    model.load_state_dict(old.state_dict())
    with torch.no_grad(): model.raw_logstd.fill_(math.log(.02))
    if not all(torch.equal(value,model.actor.state_dict()[key]) for key,value in old.actor.state_dict().items()):
        raise RuntimeError('parent mean/gain changed at initialization')
    trainer=MaskedPrefixTrainer(model,config())
    return trainer,payload


def preflight():
    checks=json.loads((HERE/'PREFIX_PPO_CHECKS.json').read_text())
    if not checks['all_passed']: raise ValueError('prefix PPO checks failed')
    for path,digest in checks['source_hashes'].items():
        if not Path(path).is_absolute() or sha256(path)!=digest: raise ValueError('prefix PPO source drift: '+path)
    if not (HERE/'PREREG.json').is_file(): raise FileNotFoundError('parent must write '+str(HERE/'PREREG.json'))
    sw.student_preflight(); sw.teacher_preflight()


def train(args):
    if not Path(args.checkpoint).is_absolute(): raise ValueError('--checkpoint must be absolute')
    if args.updates<1 or args.autonomous_ms!=6000: raise ValueError('controlled initial curriculum requires 6000ms and positive updates')
    preflight()
    out=sw.exclusive_out(args.out)
    sw.seed_all(args.seed)
    reports=[]
    try:
        core=Core(adhesion=False)
        trainer,parent=fresh_from_bc(args.checkpoint,core)
        ident=capture(core,dict(mode='prefix_ppo',cli=vars(args),reward=REWARD_CONFIG,
            settle_s=1.,prefix_s=4.8,autonomous_s=6.,teacher=sw.teacher_config(3),
            initialization='BC model only; actor mean/gains preserved; std=.02 bounds(-5,.3); fresh PPO optimizers; BC critic preserved'),
            extra_paths=SOURCE_FILES+[HERE/'PREFIX_PPO_CHECKS.json',Path(args.checkpoint).resolve()])
        atomic_json(out/'IDENTITY.json',ident)
        for update in range(1,args.updates+1):
            if update>1: core=Core(adhesion=False)
            trace=sw.Trace(full=True)
            # Actor has never run on this Core; settle has NO neural forwards.
            sw.settle(core,trace,1000)
            teacher=StaticTeacher(core,sw.teacher_config(3))
            episode,report=collect_episode(trainer.model,core,teacher,trace,args.autonomous_ms,4800,sw.tracker(core))
            ep_path=out/f'episode_{update:03d}.npz'
            np.savez_compressed(ep_path,**episode)
            report['episode']=dict(path=str(ep_path),sha256=sha256(ep_path))
            report['trace']=trace.save(out/f'episode_{update:03d}_1ms.npz')
            report['autonomous_gates']=trace.summary(core,3)
            if report['prefix_failed']:
                report['optimization']=dict(skipped=True,reason='physical teacher prefix failed; retained complete failure')
            else:
                report['optimization']=trainer.update([episode])
                tensor=trainer.tensor_episodes([episode])[0]
                report['gae']=dict(autonomous_adv_mean=float(tensor['advantages'][tensor['learn_mask'].bool()].mean()),
                    autonomous_return_sum=float(tensor['returns'][tensor['learn_mask'].bool()].sum()))
            report['update_index']=update
            reports.append(report)
            verify(ident)
            extra=dict(kind='prefix_ppo',prefix_ppo=dict(reward_config=REWARD_CONFIG,seed=args.seed,
                parent_checkpoint=str(Path(args.checkpoint).resolve()),parent_sha256=sha256(args.checkpoint),
                parent_model_identity=parent['model_identity'],initial_std=.02,logstd_bounds=(-5.,.3),
                config=asdict(trainer.config),curriculum=dict(settle_ms=1000,prefix_ms=4800,autonomous_ms=args.autonomous_ms),
                source_hashes={str(p.resolve()):sha256(p) for p in SOURCE_FILES},
                update=update,reports=reports,final_selection='last update; no score selection',
                evaluation='separate original student_wave.py evaluate; no automatic simulation'),
                **{k:v for k,v in (parent.get('extra') or {}).items() if k in
                   ('dataset_path','dataset_sha256','dataset_metadata','collection_identities')})
            save_checkpoint(out/f'update_{update:03d}.pt',trainer,ident,extra)
            atomic_json(out/'RESULT.json',dict(status='running',reports=reports,closed_loop_acceptance='NOT_ESTABLISHED'))
            opt=report['optimization']
            print(json.dumps(dict(update=update,kl=opt.get('kl'),accepted_steps=trainer.accepted_steps,
                reward=report['reward'],termination=report['termination'],autonomous_rows=report['autonomous_rows'],
                gates=report['autonomous_gates'].get('gates'),all_gates=report['autonomous_gates'].get('all_gates'),
                cycles=(report['cycles'] or {}).get('complete_per_leg')),allow_nan=False),flush=True)
        save_checkpoint(out/'final.pt',trainer,ident,extra)
        # Independent standard loader, standard config and immutable original trainer.
        check,_=sw.load_student(out/'final.pt',core)
        if not all(torch.equal(v,check.state_dict()[k]) for k,v in trainer.model.state_dict().items()):
            raise RuntimeError('original student loader checkpoint mismatch')
        atomic_json(out/'RESULT.json',dict(status='complete',reports=reports,checkpoint=str(out/'final.pt'),
            checkpoint_sha256=sha256(out/'final.pt'),standard_loader_exact=True,
            evaluation='not run; use separate deterministic original evaluate CLI',closed_loop_acceptance='NOT_ESTABLISHED'))
        return reports
    except Exception as error:
        atomic_json(out/'TECHNICAL_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),reports=reports))
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    tr=sub.add_parser('train');tr.add_argument('--checkpoint',required=True);tr.add_argument('--out',required=True)
    tr.add_argument('--updates',type=int,default=3);tr.add_argument('--autonomous-ms',type=int,default=6000)
    tr.add_argument('--seed',type=int,default=20261401)
    ev=sub.add_parser('eval',help='original compatible deterministic student evaluation, 4.8s prefix + 9.6s autonomous')
    ev.add_argument('--checkpoint',required=True);ev.add_argument('--out',required=True)
    ev.add_argument('--seed',type=int,default=20261401)
    return p


def main():
    args=parser().parse_args()
    if args.mode=='train': return train(args)
    preflight()
    return sw.evaluate(argparse.Namespace(mode='evaluate',checkpoint=args.checkpoint,out=args.out,
        seed=args.seed,cycles=2,control='student',sampled=False,native=False))


if __name__=='__main__': main()
