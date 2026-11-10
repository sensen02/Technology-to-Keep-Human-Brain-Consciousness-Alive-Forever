"""Focused S2/S5 auxiliary training checks. Constructor-only physics, no ticks."""
from __future__ import annotations
import argparse
import copy
import json
import random
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
import role_aux_bc as aux
import student_wave as sw
from identity import HERE, atomic_json, capture, verify, sha256
from ppo import save_checkpoint, load_checkpoint


def exact(a,b):
    if isinstance(a, torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def run():
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:
            raise AssertionError(name)
    with patch.object(sw.Core,'tick',side_effect=AssertionError('physics tick forbidden in auxiliary checks')):
        meta, provenance=aux.validate_role_dataset(aux.DATASET)
        check('exact_verified_dataset_train01_validation2',[(e['episode'],e['split'],e['samples']) for e in meta['episodes']]==[(0,'train',1920),(1,'train',1920),(2,'validation',1920)])
        check('teacher_mapping_actual_passed_cfg_order',provenance['class_to_teacher_leg']==['RM','LH','RF','LM','RH','LF'])
        check('teacher_exact_condition_and_no_physics',provenance['teacher_config']==sw.teacher_config(2) and provenance['constructor_only_physics_steps']==0)
        check('label_boundaries',aux.role_labels(torch.tensor([0,159,160,959,960])).tolist()==[0,0,1,5,0])
        labels=aux.role_labels(torch.arange(1920))
        check('all_six_classes_twice_integer_finite',labels.dtype==torch.int64 and torch.equal(torch.bincount(labels),torch.full((6,),320)))
        for bad in (torch.tensor([.1]),torch.tensor([-1]),torch.tensor([[0]])):
            try:
                aux.role_labels(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid labels accepted')
        check('invalid_target_indices_rejected',True)
        episodes=aux.load_episodes(meta)
        check('raw_X238_not_clock_augmented',all(x.shape==(1920,238) and y.shape==(1920,42) for split in episodes.values() for x,y,l in split))
        sw.seed_all(20261110)
        baseline=sw.new_model('gru238')
        baseline_rng=torch.get_rng_state().clone()
        sw.seed_all(20261110)
        model=sw.new_model('gru238')
        head=aux.new_aux_head(20261110)
        check('new_model_weights_critic_std_bitexact_baseline',exact(baseline.state_dict(),model.state_dict()))
        check('aux_head_fork_rng_baseline_exact',torch.equal(baseline_rng,torch.get_rng_state()))
        check('model_identity_exact_slow_GRU',exact(baseline.identity(),model.identity()))
        check('aux_head_not_registered_inside_model',not any('aux' in k or 'role' in k for k in model.state_dict()) and all(p is not q for p in model.parameters() for q in head.parameters()))
        actual_loss,actual_mse,actual_ce=aux.objective(model,head,episodes['train'],.001)
        check('full_actual_train_combined_loss_finite',all(torch.isfinite(v) for v in (actual_loss,actual_mse,actual_ce)))
        actual_loss.backward()
        check('full_actual_train_BPTT_gradients_finite',all(p.grad is None or torch.isfinite(p.grad).all() for group in (model.parameters(),head.parameters()) for p in group))
        check('full_actual_train_role_GRU_gradient_nonzero',model.actor.gru.weight_hh.grad is not None and model.actor.gru.weight_hh.grad.abs().sum()>0)
        check('full_actual_heldout_metrics_finite',all(np.isfinite(v) for v in aux.metrics(model,head,episodes['validation']).values()))
        model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        del actual_loss,actual_mse,actual_ce
        x=torch.randn(19,238,dtype=torch.float64)*.03
        y=torch.randn(19,42,dtype=torch.float64).tanh()*.1
        l=aux.role_labels(torch.arange(19)*159)
        syn=[(x[:11],y[:11],l[:11]),(x[11:],y[11:],l[11:])]
        mean,hs=aux.sequence_with_hidden(model.actor,x)
        ref,last=model.actor.sequence(x[:,None],model.actor.initial_state(1))
        check('full_sequence_means_hidden_bitexact',torch.equal(mean,ref[:,0]) and torch.equal(hs[-1:],last))
        grad1=torch.autograd.grad(mean.square().sum(),tuple(model.actor.parameters()),retain_graph=True)
        grad2=torch.autograd.grad(ref.square().sum(),tuple(model.actor.parameters()))
        check('sequence_gradient_bitexact_original',all(torch.equal(a,b) for a,b in zip(grad1,grad2)))
        # Explicit step state carries all history and restarts only per episode.
        prefix,state=model.actor.sequence(x[:7,None],model.actor.initial_state(1))
        suffix,final=model.actor.sequence(x[7:,None],state)
        check('continuous_hidden_prefix_no_reset',torch.equal(torch.cat((prefix,suffix))[:,0],mean) and torch.equal(final,hs[-1:]))
        reset=torch.zeros((19,1),dtype=torch.bool);reset[7]=True
        resetmean,_=model.actor.sequence(x[:,None],model.actor.initial_state(1),reset=reset)
        restarted,_=aux.sequence_with_hidden(model.actor,x[7:])
        check('original_reset_semantics_preserved',torch.equal(resetmean[7:,0],restarted))
        before=copy.deepcopy(model.state_dict())
        aux.objective(model,head,syn,.001)
        aux.objective(model,head,[(a,b,(c+3)%6) for a,b,c in syn],.001)
        check('altered_targets_no_policy_state_effect_before_optimization',exact(before,model.state_dict()))
        model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        _,hidden=aux.sequence_with_hidden(model.actor,x)
        F.cross_entropy(head(hidden),l).backward()
        check('role_gradient_reaches_encoder_GRU',all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for name,p in model.actor.named_parameters() if name.startswith(('encoder.','gru.'))))
        check('role_only_loss_does_not_touch_action_decoder_gain_critic_std',all(p.grad is None for name,p in model.named_parameters() if not name.startswith(('actor.encoder.','actor.gru.'))))
        # Weight-zero must match baseline Adam actor updates bit for bit.
        zero=copy.deepcopy(baseline);zerohead=aux.new_aux_head(20261110)
        ordinary=copy.deepcopy(baseline)
        zeroopt=aux.new_optimizer(zero,zerohead,.001)
        ordinaryopt=torch.optim.Adam(ordinary.actor.parameters(),lr=.001)
        for _ in range(2):
            zeroopt.zero_grad(set_to_none=True);ordinaryopt.zero_grad(set_to_none=True)
            zloss,_,_=aux.objective(zero,zerohead,syn,0.)
            bloss=sw.bounded_bc_loss(ordinary,[(a,b) for a,b,c in syn])
            check('zero_weight_loss_exact',torch.equal(zloss,bloss))
            zloss.backward();bloss.backward()
            zeroopt.step();ordinaryopt.step()
            check('zero_weight_two_updates_actor_critic_std_bitexact',exact(zero.state_dict(),ordinary.state_dict()))
        check('zero_head_no_grads_no_adam_state',all(p.grad is None and p not in zeroopt.state for p in zerohead.parameters()))
        check('zero_weight_BCAdam_actor_state_exact',exact(zeroopt.state_dict()['state'],ordinaryopt.state_dict()['state']))
        # Positive weight: two synthetic updates, heldout tensors never enter backward.
        model=copy.deepcopy(baseline);head=aux.new_aux_head(20261110)
        optimizer=aux.new_optimizer(model,head,.001)
        unchanged={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('actor.')}
        hx=episodes['validation'][0][0][:9].clone().requires_grad_()
        hy=episodes['validation'][0][1][:9].clone().requires_grad_()
        heldout=[(hx,hy,aux.role_labels(torch.arange(9)))]
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss,mse,ce=aux.objective(model,head,syn,.001)
            check('combined_loss_finite',all(torch.isfinite(v) for v in (loss,mse,ce)))
            loss.backward()
            check('combined_actor_head_gradients_finite',all(p.grad is None or torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params']))
            optimizer.step()
            aux.metrics(model,head,heldout)
        check('heldout_no_gradients',hx.grad is None and hy.grad is None)
        check('critic_std_unchanged_after_two_updates',all(torch.equal(v,model.state_dict()[k]) for k,v in unchanged.items()))
        # Sample weighting versus concatenated per-episode sums (separate zero states).
        loss,mse,ce=aux.objective(model,head,syn,.001)
        parts=[aux.objective(model,head,[e],.001) for e in syn]
        expected=sum(part[0]*len(e[0]) for part,e in zip(parts,syn))/len(x)
        check('sample_weighted_episode_objective',torch.allclose(loss,expected,rtol=0,atol=1e-16))
        trained_head=copy.deepcopy(head.state_dict())
        with torch.no_grad():
            before_action=model.act(x[0],torch.zeros(255,dtype=torch.float64),deterministic=True,state=model.actor.initial_state(1))
            for p in head.parameters():
                p.fill_(float('nan'))
            after_action=model.act(x[0],torch.zeros(255,dtype=torch.float64),deterministic=True,state=model.actor.initial_state(1))
        check('arbitrary_head_mutation_action_hidden_inference_exact',exact(before_action,after_action))
        head.load_state_dict(trained_head)
        # Complete checkpoint uses real capture, real fresh PPO, real BC Adam.
        core=sw.Core(adhesion=False)
        check_paths=[p for p in aux.identity_paths(meta,provenance) if p.name != 'ROLE_AUX_CHECKS.json']
        ident=capture(core,dict(mode='role_aux_synthetic_checks',teacher_reference_provenance=provenance),extra_paths=check_paths)
        verify(ident)
        check('identity_closure_own_sources_dataset_original_teacher',all(str(p.resolve()) in ident['files'] for p in check_paths))
        args=argparse.Namespace(mode='train',dataset=str(aux.DATASET),out='synthetic_check_only',steps=300,lr=.001,seed=20261110,role_weight=.001)
        # Do not include check report itself in synthetic identity to avoid circular hash.
        trainer=sw.make_trainer(model)
        with tempfile.TemporaryDirectory(prefix='role_aux_check_') as td:
            path=Path(td)/'final.pt'
            extra=aux.checkpoint_extra(model,head,optimizer,args,meta,provenance,[])
            save_checkpoint(path,trainer,ident,extra=extra)
            payload=torch.load(path,map_location='cpu',weights_only=False)
            expected_numpy=np.random.random(5);expected_python=[random.random() for _ in range(5)];expected_torch=torch.rand(5)
            restored=sw.new_model('gru238');fresh=sw.make_trainer(restored)
            loaded_extra=load_checkpoint(path,fresh,ident,restore_rng=True)
            check('checkpoint_full_model_and_PPO_exact',exact(trainer.snapshot(),fresh.snapshot()))
            check('checkpoint_rng_numpy_python_torch_exact',np.array_equal(expected_numpy,np.random.random(5)) and expected_python==[random.random() for _ in range(5)] and torch.equal(expected_torch,torch.rand(5)))
            restoredhead=aux.new_aux_head(20261110);restoredhead.load_state_dict(loaded_extra['aux_head_state_dict'])
            restoredopt=aux.new_optimizer(restored,restoredhead,.001);restoredopt.load_state_dict(loaded_extra['aux_bc_optimizer'])
            check('checkpoint_BCAdam_and_aux_head_exact',exact(optimizer.state_dict(),restoredopt.state_dict()) and exact(head.state_dict(),restoredhead.state_dict()))
            check('fresh_PPO_states_empty',len(payload['state']['actor']['state'])==0 and len(payload['state']['critic']['state'])==0)
            deployed,_=sw.load_student(path,core)
            check('unchanged_load_student_exact_no_aux_keys',exact(deployed.state_dict(),model.state_dict()) and not any('aux' in k or 'role' in k for k in deployed.state_dict()))
            with torch.no_grad():
                first=deployed.act(x[0],torch.zeros(255,dtype=torch.float64),deterministic=True,state=deployed.actor.initial_state(1))
            original_sha=sha256(path)
            payload['extra']['aux_head_state_dict']={k:torch.full_like(v,float('nan')) for k,v in head.state_dict().items()}
            payload['extra']['teacher_reference_provenance']['class_to_teacher_leg']=['TAMPER']*6
            torch.save(payload,path)
            check('full_checkpoint_hash_detects_extra_tamper',sha256(path)!=original_sha)
            other,_=sw.load_student(path,core)
            second=other.act(x[0],torch.zeros(255,dtype=torch.float64),deterministic=True,state=other.actor.initial_state(1))
            check('deployment_ignores_head_and_role_provenance_tamper',exact(first,second))
            check('deployment_neural_step_mean_tanh',torch.equal(first['action'],first['mean'].tanh()))
            # Authoritative source/data drift remains enforced by unmodified loader.
            payload['extra']['dataset_sha256']='0'*64
            torch.save(payload,path)
            try:
                sw.load_student(path,core)
            except ValueError as e:
                check('deployment_rejects_dataset_hash_tamper','dataset metadata changed' in str(e))
            else:
                raise AssertionError('dataset tamper not rejected')
        check('no_physics_steps_no_gait_no_long_training',float(core.d.time)==0.)
    report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=aux.source_hashes(),
        teacher_reference_provenance=provenance,synthetic_updates=2,zero_control_updates=2,
        dataset_metadata_sha256=sha256(aux.DATASET/'DATASET.json'),physics_steps=0,
        long_training='NOT_EXECUTED; conditional parent only',closed_loop='NOT_EVALUATED',
        checkpoint_integrity='full checkpoint SHA256 must be checked externally; unchanged load_student deliberately ignores training-only auxiliary extra')
    atomic_json(HERE/'ROLE_AUX_CHECKS.json',report)
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physics_steps=0)),flush=True)
    return report


if __name__=='__main__':
    run()
