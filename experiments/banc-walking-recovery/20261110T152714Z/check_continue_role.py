"""Bounded role continuation regression: tiny 2->3 resume and real2000 zero updates."""
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
        check('real_parent_all_actor_and_head_Adam_counters_2000',c.optimizer_steps(real['optimizer'],real['model'],real['head'])==[2000]*13)
        check('real_parent_model_head_Adam_RNG_exact',c.exact(real['model'].state_dict(),real['payload']['state']['model']) and c.exact(real['head'].state_dict(),real['payload']['extra']['aux_head_state_dict']) and c.exact(real['optimizer'].state_dict(),real['payload']['extra']['aux_bc_optimizer']) and c.exact(c.rng_state(),real['payload']['rng']))
        initial=c.measure(real,2000)
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
            ('changed_loss',lambda p:p['extra']['aux_bc_config'].__setitem__('loss','changed')),
            ('changed_original_ancestry',lambda p:p['extra']['original_aux_bc_config'].__setitem__('seed',0)),
            ('changed_parent_chain',lambda p:p['extra']['parent'].__setitem__('step',999)),
            ('changed_identity_profile',lambda p:p['identity']['config']['cli'].__setitem__('total_steps',3000)),
            ('changed_parent_aux_config',lambda p:p['extra']['parent_aux_bc_config'].__setitem__('steps',999)),
            ('changed_original_curve',lambda p:p['extra']['curve'][300]['train'].__setitem__('MSE',-1.)),
            ('changed_1000_curve',lambda p:p['extra']['curve'][1000]['train'].__setitem__('MSE',-1.)),
            ('changed_global_step',lambda p:p['extra'].__setitem__('global_step',999)),
            ('changed_critic',lambda p:next(v for k,v in p['state']['model'].items() if k.startswith('critic.') and v.is_floating_point()).add_(1)),
            ('changed_PPO_state',lambda p:p['state'].__setitem__('updates',1)),
            ('head_digest_tamper',lambda p:p['extra'].__setitem__('aux_head_sha256','0'*64)),
            ('config_digest_tamper',lambda p:p['extra'].__setitem__('aux_config_sha256','0'*64)),
            ('head_tamper',lambda p:p['extra']['aux_head_state_dict']['weight'].add_(1)),
            ('head_Adam_counter',lambda p:p['extra']['aux_bc_optimizer']['state'][12].__setitem__('step',torch.tensor(999.))),
        ):
            bad=deepcopy(real['payload']); mutate(bad); cases[name]=bad
        original_load=torch.load
        for name,bad in cases.items():
            def tampered_load(path,*args,b=bad,**kwargs):
                return b if Path(path)==c.PARENT else original_load(path,*args,**kwargs)
            with patch.object(torch,'load',side_effect=tampered_load),patch.object(sw,'load_student',side_effect=AssertionError('mutation reached')) as loader:
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
        valid_cli=argparse.Namespace(checkpoint=str(c.PARENT),out=str(c.OUT),total_steps=3000,checkpoint_every=250)
        c.validate_cli(valid_cli)
        for key,value in [('total_steps',2000),('total_steps',3001),('checkpoint_every',500),('out',str(HERE/'wrong'))]:
            args=deepcopy(valid_cli); setattr(args,key,value)
            try:
                c.validate_cli(args)
            except ValueError:
                check('reject_fixed_profile_'+key+'_'+str(value),True)
            else:
                raise AssertionError('accepted changed profile')
        original_read=Path.read_text
        def bad_result_read(path,*args,**kwargs):
            if path==c.PARENT_RESULT:
                bad=deepcopy(result); bad['steps']=999
                return json.dumps(bad)
            return original_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',bad_result_read),patch.object(sw,'load_student',side_effect=AssertionError('mutation reached')) as loader:
            try:
                c.build_resume(c.PARENT)
            except ValueError:
                check('reject_external_parent_report_before_model_load',not loader.called)
            else:
                raise AssertionError('accepted changed parent report')
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
        parent_identity=str(c.PARENT_IDENTITY),parent_identity_sha256=sha256(c.PARENT_IDENTITY),
        parent_aux_config=expected,parent_initial_metrics=initial,real_parent_updates=0,synthetic_direct_updates=3,synthetic_resumed_updates=1,
        physics_steps=0,closed_loop='NOT_EVALUATED',dataset_path=extra['dataset_path'],dataset_sha256=extra['dataset_sha256'],
        role_head='training auxiliary only, excluded from deployment',fixed_budget=dict(parent_steps=2000,total_steps=3000,additional_steps=1000))
    original_read=Path.read_text
    def checked_report_read(path,*args,**kwargs):
        return json.dumps(report) if path==c.CHECKS else original_read(path,*args,**kwargs)
    original_sha=c.sha256
    with patch.object(Path,'read_text',checked_report_read):
        c.preflight()
        check('production_preflight_complete_source_and_parent_closure',True)
        for path in (c.PARENT,c.PARENT_RESULT,c.PARENT_IDENTITY):
            with patch.object(c,'sha256',side_effect=lambda p,target=path:'0'*64 if Path(p)==target else original_sha(p)):
                try:
                    c.preflight()
                except ValueError:
                    check('preflight_pins_external_'+path.name,True)
                else:
                    raise AssertionError('preflight accepted substituted '+str(path))
    report['all_passed']=all(checks.values())
    atomic_json(c.CHECKS,report)
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),physical_steps=0,real_parent_updates=0)),flush=True)
    return report


if __name__=='__main__':
    run()
