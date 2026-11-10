"""Bounded S2/S5 replacement-ANN checks; no full training or walking evaluation."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

import models
import student_wave as s
from identity import HERE, atomic_json, sha256
from ppo import save_checkpoint, load_checkpoint

FF_PARENT = HERE.parent/'20261110T111240Z/models.py'


def main():
    s.seed_all(7731)
    rows = []
    def check(name, condition, **details):
        rows.append(dict(name=name, passed=bool(condition), **details))
        if not condition:
            raise AssertionError(name)
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError, KeyError):
            check(name, True)
        else:
            check(name, False)
    c = s.Core(adhesion=False)
    hspan = s.physical_halfspan(c)
    model = s.new_model('gru238')
    actor = model.actor
    check('declared slow neural integration uses stable expm1', actor.ALPHA == -math.expm1(-.005/.1))
    check('explicit physical H compiled for GRU', np.array_equal(actor.initial_halfspan.numpy(), hspan))
    rejects('GRU constructor requires explicit physical H', lambda: models.ActorCritic('gru238'))
    check('all neural weights and all 42 gains learnable', all(p.requires_grad for p in actor.parameters()) and actor.raw_gain.shape == (42,))
    check('initial gain exactly native H x10 within roundoff', torch.allclose(actor.gain(), torch.as_tensor(10*hspan), atol=1e-14, rtol=1e-15))
    check('initial state zero with model dtype and no hidden cache', torch.equal(actor.initial_state(2), torch.zeros(2,128,dtype=torch.float64)) and not any('state' in k for k in actor.__dict__ if not k.startswith('_')))
    check('critic remains raw255', model.critic.layers[0].in_features == 255)
    mi = model.identity()
    check('GRU identity explicit transform tau policy mask H gain', mi['actor_input_transform'] == 'physical_gru_slow_plus_trainable_inhibitory_target_v1' and mi['neural_integration_tau_s'] == .1 and mi['policy_dt_s'] == .005 and mi['actor_mask_ranges'] == [[42,84],[126,168],[195,237]] and mi['gain_init_H'] == hspan.tolist())

    # Frozen FFv2 source is evidence only; nothing in the historical directory is edited.
    spec = importlib.util.spec_from_file_location('frozen_ffv2_models', FF_PARENT)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    torch.manual_seed(710); before = old.ActorCritic('ff238', critic_obs_dim=255, initial_logstd=math.log(.05), actor_halfspan=hspan)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(710); after = models.ActorCritic('ff238', critic_obs_dim=255, initial_logstd=math.log(.05), actor_halfspan=hspan)
    check('FFv2 constructor and RNG bitexact unchanged', torch.equal(old_rng,torch.get_rng_state()) and all(torch.equal(v,after.state_dict()[k]) for k,v in before.state_dict().items()))
    obs = torch.randn(20,238,dtype=torch.float64)*.05
    check('FFv2 identity and forward bitexact unchanged', before.identity() == after.identity() and torch.equal(before.actor(obs),after.actor(obs)))
    # Historical GRU linear/head initialization preserved including scaled head.
    torch.manual_seed(710); raw = old.ActorCritic('gru238',critic_obs_dim=255)
    torch.manual_seed(710); slow = models.ActorCritic('gru238',critic_obs_dim=255,actor_halfspan=hspan)
    check('GRU original encoder cell decoder head initialization preserved', all(torch.equal(v,slow.actor.state_dict()[k]) for k,v in raw.actor.state_dict().items()))

    state = torch.randn(1,128,dtype=torch.float64)*.1
    x = obs[:1].clone()
    mean, newstate = actor.step(x,state)
    candidate = actor.gru(torch.tanh(actor.encoder(x*actor.physical_mask)),state)
    check('neural update exact declared alpha formula', torch.equal(newstate,(1-actor.ALPHA)*state+actor.ALPHA*candidate))
    zero = deepcopy(actor)
    with torch.no_grad():
        for p in zero.encoder.parameters(): p.zero_()
        for p in zero.head.parameters(): p.zero_()
        enc = torch.tanh(zero.encoder(x*zero.physical_mask))
        cand = zero.gru(enc,state)
        zm,zh = zero.step(x,state)
    check('zero encoder/head still evolves exact neural alpha formula', torch.equal(zh,(1-zero.ALPHA)*state+zero.ALPHA*cand) and torch.equal(zm,-zero.gain()*x[:,126:168]))
    changed = x.clone(); changed[:,42:84] += 3; changed[:,195:237] -= 2
    mm,hh = actor.step(changed,state)
    check('target error and previous action influence exactly zero', torch.equal(mean,mm) and torch.equal(newstate,hh))
    target_changed = x.clone(); target_changed[:,126:168] += .01
    tm,th = actor.step(target_changed,state)
    check('target position excluded from neural candidate', torch.equal(newstate,th))
    # Coherent physical T perturbation also changes target error; fixed hidden.
    def coherent_target(t):
        z=x.clone(); z[:,126:168]=t; z[:,42:84] += t-x[:,126:168]
        return actor.step(z,state)[0][0]
    jac = torch.autograd.functional.jacobian(coherent_target,x[:,126:168])[...,0,:]
    check('coherent T feedthrough KT negative diagonal', torch.equal(jac,-torch.diag(actor.gain())))
    q_only = torch.autograd.functional.jacobian(lambda q: actor.step(torch.cat((q,x[:,42:]),-1),state)[0][0],x[:,:42])[...,0,:]
    physical_q = torch.autograd.functional.jacobian(lambda q: actor.head(torch.tanh(actor.hidden((1-actor.ALPHA)*state+actor.ALPHA*actor.gru(torch.tanh(actor.encoder(torch.cat((q,x[:,42:]),-1)*actor.physical_mask)),state))))[0],x[:,:42])[...,0,:]
    check('q derivative has no explicit gain term', torch.equal(q_only,physical_q))

    seq = torch.randn(1920,1,238,dtype=torch.float64)*.03
    reset = torch.zeros(1920,1,dtype=torch.bool); reset[0]=True; reset[990]=True
    valid = torch.ones(1920,1,dtype=torch.bool); valid[500:510]=False; valid[-5:]=False
    with torch.no_grad():
        full,end = actor.sequence(seq,actor.initial_state(1),reset,valid)
        st=actor.initial_state(1); pieces=[]
        for t in range(len(seq)):
            out,candidate=actor.step(seq[t],st,reset[t])
            st=torch.where(valid[t,:,None],candidate,st); pieces.append(out)
        check('full1920 sequence reset/padding vs step bitexact', torch.equal(full,torch.stack(pieces)) and torch.equal(end,st))
        _,a=actor.step(seq[990],torch.randn_like(st),reset[990]); _,b=actor.step(seq[990],actor.initial_state(1))
        check('reset happens before step exactly',torch.equal(a,b))
        initial=torch.randn_like(st)
        padding,_end=actor.sequence(seq[:3],initial,torch.ones(3,1,dtype=torch.bool),torch.zeros(3,1,dtype=torch.bool))
        check('invalid padding freezes original state even with reset',torch.equal(initial,_end) and torch.isfinite(padding).all())
    actor.zero_grad(set_to_none=True)
    start=time.perf_counter()
    full,_=actor.sequence(seq,actor.initial_state(1))
    loss=(full.tanh()-.02).square().mean();loss.backward()
    full_seconds=time.perf_counter()-start
    check('full1920 contiguous finite gradients all actor parameters',torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()),seconds=full_seconds)
    gains_before=actor.raw_gain.detach().clone()
    optimizer=torch.optim.Adam(actor.parameters(),lr=.001);optimizer.step()
    check('all42 inhibitory gains updated by full sequence BC loss',torch.all(actor.raw_gain.detach()!=gains_before))
    # Only 20 samples for timing extrapolation; no 300-step training.
    actor.zero_grad(set_to_none=True)
    start=time.perf_counter();m,_=actor.sequence(seq[:20],actor.initial_state(1));m.tanh().square().mean().backward();timing20=time.perf_counter()-start
    check('20sample scaling measurement finite',math.isfinite(timing20) and timing20>0, samples=20,seconds=timing20,linear_1920_forward_backward_estimate_s=timing20*96, actual_single_1920_forward_backward_s=full_seconds)
    actor.zero_grad(set_to_none=True)

    with tempfile.TemporaryDirectory(prefix='slow-gru-check-') as tmp:
        path=Path(tmp)/'roundtrip.pt'
        trainer=s.make_trainer(model)
        ident={'purpose':'synthetic exact roundtrip, no walking evidence'}
        save_checkpoint(path,trainer,ident,extra={'bc_optimizer':optimizer.state_dict()})
        expected_rng=(random.random(),np.random.random(),torch.rand(5))
        loaded=s.new_model('gru238'); lt=s.make_trainer(loaded)
        extra=load_checkpoint(path,lt,ident,restore_rng=True)
        check('checkpoint weights exact',all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items()))
        actual_rng=(random.random(),np.random.random(),torch.rand(5))
        check('checkpoint RNG python numpy torch exact',actual_rng[0]==expected_rng[0] and actual_rng[1]==expected_rng[1] and torch.equal(actual_rng[2],expected_rng[2]))
        check('PPO optimizers fresh separate from BC Adam',not lt.actor_optimizer.state and not lt.critic_optimizer.state)
        restored=torch.optim.Adam(loaded.actor.parameters(),lr=.001);restored.load_state_dict(extra['bc_optimizer'])
        check('BC Adam exact full state roundtrip',len(restored.state)==len(optimizer.state) and all(torch.equal(a[k],b[k]) if isinstance(a[k],torch.Tensor) else a[k]==b[k] for a,b in zip(optimizer.state.values(),restored.state.values()) for k in a))
        for am,op in ((model,optimizer),(loaded,restored)):
            op.zero_grad(set_to_none=True);s.bc_mean(am,seq[:20,0]).tanh().square().mean().backward();op.step()
        check('restored BC Adam next update weights bitexact',all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items()))
        payload=torch.load(path,weights_only=False);payload['model_identity']={k:v for k,v in mi.items() if k in ('architecture','obs_dim','critic_obs_dim','logstd_bounds','dtype')}
        bad=Path(tmp)/'raw.pt';torch.save(payload,bad)
        snap={k:v.clone() for k,v in loaded.state_dict().items()};rng=torch.get_rng_state().clone()
        rejects('old raw GRU rejected before checkpoint parameter mutation',lambda:load_checkpoint(bad,lt,ident))
        check('rejected raw checkpoint preserves weights and RNG',torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,loaded.state_dict()[k]) for k,v in snap.items()))
        # Actual deployment validation is exercised before any actor construction.
        with patch.object(s,'verify_runtime',lambda *a:None),patch.object(s,'ActorCritic',side_effect=AssertionError('must reject before actor creation')):
            rejects('deployment raw GRU rejected before construction',lambda:s.load_student(bad,c))
            for key,value in (('neural_integration_tau_s',.2),('policy_dt_s',.01),('neural_integration_alpha',.5),('actor_mask_ranges',[]),('gain_init_H',[1.]*42)):
                payload['model_identity']=deepcopy(mi);payload['model_identity'][key]=value;torch.save(payload,bad)
                rejects('deployment rejects changed '+key,lambda:s.load_student(bad,c))
    sources=s.SOURCE_FILES+[Path(__file__).resolve(),HERE/'check_student_wave.py',FF_PARENT]
    output=dict(status='PASS',all_passed=all(r['passed'] for r in rows),passed=len(rows),total=len(rows),checks=rows,
        command=[sys.executable,'-B',str(Path(__file__).resolve())],source_hashes={str(p.resolve()):sha256(p) for p in sources},
        model_identity=mi,padding_contract='valid freezes hidden; invalid candidate output is computed, caller masks loss',
        not_executed=['full BC training','full walking evaluation'],interpretation='replacement ANN engineering checks; no claim slow tau improves walking')
    atomic_json(HERE/'SLOW_GRU_CHECKS.json',output)
    print(json.dumps(dict(all_passed=True,passed=len(rows),report=str(HERE/'SLOW_GRU_CHECKS.json'))),flush=True)


if __name__=='__main__':
    main()
