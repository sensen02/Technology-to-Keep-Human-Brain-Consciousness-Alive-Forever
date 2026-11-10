"""Bounded role continuation regression: tiny 2->3 resume and real300 zero updates."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from torch.nn import functional as F
import continue_role_bc as c
import role_aux_bc as aux
import student_wave as sw
from identity import HERE, atomic_json, capture, sha256
from ppo import save_checkpoint


def run():
    torch.set_num_threads(1)
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:
            raise AssertionError(name)
    with patch.object(sw.Core,'tick',side_effect=AssertionError('physical tick forbidden')):
        aux.preflight()
        real=c.build_resume(c.PARENT)
        check('real_parent_all_actor_and_head_Adam_counters_300',c.optimizer_steps(real['optimizer'],real['model'],real['head'])==[300]*13)
        check('real_parent_model_head_Adam_RNG_exact',c.exact(real['model'].state_dict(),real['payload']['state']['model']) and c.exact(real['head'].state_dict(),real['payload']['extra']['aux_head_state_dict']) and c.exact(real['optimizer'].state_dict(),real['payload']['extra']['aux_bc_optimizer']) and c.exact(c.rng_state(),real['payload']['rng']))
        initial=c.measure(real,300)
        result=json.loads(c.PARENT_RESULT.read_text())
        check('real_parent_zero_update_full_train_heldout_metrics_exact',initial==result['final']==real['curve'][-1])
        check('real_parent_PPO_fresh_zero_physical_steps',not real['trainer'].actor_optimizer.state and not real['trainer'].critic_optimizer.state and float(real['core'].d.time)==0.)
        check('real_parent_train01_validation2_only',len(real['episodes']['train'])==2 and len(real['episodes']['validation'])==1 and all(len(x)==1920 for es in real['episodes'].values() for x,y,l in es))
        # Negative tests enter the production builder; model loader must not run.
        extra=real['payload']['extra']; expected=extra['aux_bc_config']
        cases={}
        for name,mutate in (
            ('wrong_config',lambda p:p['extra']['aux_bc_config'].__setitem__('lr',.002)),
            ('wrong_role_weight',lambda p:p['extra']['aux_bc_config'].__setitem__('role_weight',.002)),
            ('new_data',lambda p:p['extra']['aux_bc_config'].__setitem__('dataset','/new/data')),
            ('head_tamper',lambda p:p['extra']['aux_head_state_dict']['weight'].add_(1)),
            ('head_Adam_counter',lambda p:p['extra']['aux_bc_optimizer']['state'][12].__setitem__('step',torch.tensor(299.))),
        ):
            bad=deepcopy(real['payload']); mutate(bad); cases[name]=bad
        for name,bad in cases.items():
            with patch.object(c,'validate_parent',side_effect=lambda path,b=bad:(c.validate_aux_payload(b,expected,300),real['episodes'])),patch.object(sw,'load_student',side_effect=AssertionError('mutation reached')) as loader:
                try:
                    c.build_resume(c.PARENT)
                except ValueError:
                    check('reject_'+name+'_before_model_load',not loader.called)
                else:
                    raise AssertionError('accepted '+name)
        with patch.object(sw,'load_student',side_effect=AssertionError('mutation reached')) as loader:
            try:
                c.build_resume(str(HERE/'unrelated.pt'))
            except ValueError:
                check('reject_wrong_ancestry_before_model_load',not loader.called)
            else:
                raise AssertionError('wrong ancestry accepted')
        with patch.object(c,'sha256',return_value='0'*64),patch.object(sw,'load_student',side_effect=AssertionError('mutation reached')) as loader:
            try:
                c.build_resume(c.PARENT)
            except ValueError:
                check('reject_external_checkpoint_hash_before_model_load',not loader.called)
            else:
                raise AssertionError('external digest accepted')
        sw.seed_all(20261110)
        model=sw.new_model('gru238'); head=aux.new_aux_head(20261110); opt=aux.new_optimizer(model,head,.001)
        x=torch.randn(13,238,dtype=torch.float64)*.03
        y=torch.randn(13,42,dtype=torch.float64).tanh()*.1
        labels=aux.role_labels(torch.arange(13)*159)
        episodes=dict(train=[(x[:8],y[:8],labels[:8]),(x[8:],y[8:],labels[8:])],validation=[(x,y,labels)])
        untouched=c.frozen(model)
        curve=[]
        tiny=dict(model=model,head=head,episodes=episodes)
        for step in range(3):
            if step:
                c.update(model,head,opt,episodes['train'])
            curve.append(c.measure(tiny,step))
        # Real identity and real checkpoint, with original provenance and full optimizer.
        ident=capture(real['core'],dict(mode='tiny_role_resume_check'),extra_paths=aux.identity_paths(extra['dataset_metadata'],extra['teacher_reference_provenance'])+[HERE/'continue_role_bc.py',HERE/'check_continue_role.py'])
        args=argparse.Namespace(mode='train',dataset=str(aux.DATASET),out='tiny_check_only',steps=2,lr=.001,seed=20261110,role_weight=.001)
        tiny_extra=aux.checkpoint_extra(model,head,opt,args,extra['dataset_metadata'],extra['teacher_reference_provenance'],curve)
        with tempfile.TemporaryDirectory(prefix='role_continuation_check_') as td:
            path=Path(td)/'final.pt'
            save_checkpoint(path,sw.make_trainer(model),ident,extra=tiny_extra)
            payload=torch.load(path,map_location='cpu',weights_only=False)
            c.validate_aux_payload(payload,tiny_extra['aux_bc_config'],2)
            c.update(model,head,opt,episodes['train'])
            direct=dict(model=deepcopy(model.state_dict()),head=deepcopy(head.state_dict()),optimizer=deepcopy(opt.state_dict()),rng=deepcopy(c.rng_state()))
            # Only the fixed-real-parent validator is replaced for the synthetic fixture;
            # actual build_resume constructors/head/Adam/load_checkpoint/RNG path is used.
            with patch.object(c,'validate_parent',return_value=(payload,episodes)):
                resumed=c.build_resume(path,real['core'])
            check('tiny_saved2_full_model_head_Adam_RNG_restored',c.exact(resumed['model'].state_dict(),payload['state']['model']) and c.exact(resumed['head'].state_dict(),payload['extra']['aux_head_state_dict']) and c.exact(resumed['optimizer'].state_dict(),payload['extra']['aux_bc_optimizer']) and c.exact(c.rng_state(),payload['rng']))
            c.update(resumed['model'],resumed['head'],resumed['optimizer'],episodes['train'])
            check('tiny_direct3_vs_build_resume3_model_bitexact',c.exact(direct['model'],resumed['model'].state_dict()))
            check('tiny_direct3_vs_build_resume3_head_bitexact',c.exact(direct['head'],resumed['head'].state_dict()))
            check('tiny_direct3_vs_build_resume3_Adam_all_moments_counters_bitexact',c.exact(direct['optimizer'],resumed['optimizer'].state_dict()) and c.optimizer_steps(resumed['optimizer'],resumed['model'],resumed['head'])==[3]*13)
            check('tiny_direct3_vs_build_resume3_RNG_bitexact',c.exact(direct['rng'],c.rng_state()))
            check('tiny_critic_logstd_unchanged',c.exact(untouched,c.frozen(resumed['model'])))
            means,hidden=aux.sequence_with_hidden(resumed['model'].actor,x)
            seq,last=resumed['model'].actor.sequence(x[:,None],resumed['model'].actor.initial_state(1))
            prefix,state=resumed['model'].actor.sequence(x[:5,None],resumed['model'].actor.initial_state(1))
            suffix,final=resumed['model'].actor.sequence(x[5:,None],state)
            check('restored_full_sequence_and_stateful_prefix_exact',torch.equal(means,seq[:,0]) and torch.equal(hidden[-1:],last) and torch.equal(torch.cat((prefix,suffix))[:,0],means) and torch.equal(final,last))
            q=x.clone().requires_grad_(); _,hs=aux.sequence_with_hidden(resumed['model'].actor,q)
            F.cross_entropy(resumed['head'](hs[-1:]),labels[-1:]).backward()
            check('restored_role_graph_reaches_prefix_and_GRU',q.grad[:5].abs().sum()>0 and resumed['model'].actor.gru.weight_hh.grad.abs().sum()>0)
            before=means.detach().clone()
            with torch.no_grad():
                for p in resumed['head'].parameters():
                    p.fill_(float('nan'))
                after,_=aux.sequence_with_hidden(resumed['model'].actor,x)
            check('head_never_in_policy_inference',torch.equal(before,after))
        check('no_physics_no_real_training',float(real['core'].d.time)==0.)
    report=dict(all_passed=all(checks.values()),checks=checks,source_hashes=c.source_hashes(),parent_checkpoint=str(c.PARENT),
        parent_checkpoint_sha256=sha256(c.PARENT),parent_result=str(c.PARENT_RESULT),parent_result_sha256=sha256(c.PARENT_RESULT),
        parent_aux_config=expected,parent_initial_metrics=initial,real_parent_updates=0,synthetic_direct_updates=3,synthetic_resumed_updates=1,
        physics_steps=0,closed_loop='NOT_EVALUATED',dataset_path=extra['dataset_path'],dataset_sha256=extra['dataset_sha256'],
        role_head='training auxiliary only, excluded from deployment',fixed_budget=dict(parent_steps=300,total_steps=1000,additional_steps=700))
    atomic_json(c.CHECKS,report)
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physical_steps=0,real_parent_updates=0)),flush=True)
    return report


if __name__=='__main__':
    run()
