"""S2/S5 inhibitory ANN wiring only; no gait or long training.

Frozen-body derivatives are with respect to unsaturated raw sensory coordinates.
The actual encoder retains its preexisting float32 quantization and clipping.
"""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import random
import sys
import tempfile

import numpy as np
import torch

import student_wave as s
from models import ActorCritic, MLP, NeuralFeedbackActorFF
from identity import HERE, atomic_json, capture, sha256


def main():
    s.seed_all(472)
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
    physical_before = capture(c, dict(purpose='inhibitory synthetic wiring'))
    h = s.physical_halfspan(c)
    model = s.new_model('ff238')
    actor = model.actor
    check('exact target-position v2 identity and branch range',
          model.identity()['actor_input_transform'] == 'physical_mlp_plus_trainable_inhibitory_target_v2' and
          model.identity()['inhibitory_input_range'] == [126,168] and
          '-Tposition/H' in model.identity()['initialization_prior'])
    check('FF declared neural class, two 128 tanh layers', isinstance(actor, NeuralFeedbackActorFF) and
          [tuple(layer.weight.shape) for layer in actor.layers] == [(128,238),(128,128),(42,128)])
    check('physical H exact compiled actuator-order sensory scale', np.array_equal(h, actor.initial_halfspan.numpy()) and
          np.array_equal(h, np.maximum(np.ptp(c.plant.jnt_range[c.shim.jids],axis=1)/2,.1)))
    check('initial gain equals 10H', torch.allclose(actor.gain(), torch.as_tensor(10*h), rtol=1e-14, atol=1e-14))
    check('all actor parameters including 42 gains trainable', all(p.requires_grad for p in actor.parameters()) and actor.raw_gain.shape == (42,))
    check('PPO actor parameter collection includes gain', any(p is actor.raw_gain for p in model.actor_parameters()))
    x, v = s.observations(c, np.zeros(42), .02)
    xx = torch.as_tensor(x).clone()
    rng = np.random.default_rng(472)
    max_off = 0.; max_prev = 0.; min_q = float('inf'); rho = []
    for k in range(4):
        probe = xx.clone() if k == 0 else torch.as_tensor(rng.normal(0,.1,238))
        def action_target(delta):
            changed = probe.clone()
            changed[42:84] = probe[42:84] + delta / actor.initial_halfspan
            changed[126:168] = probe[126:168] + delta / actor.initial_halfspan
            return actor(changed).tanh()
        zero = torch.zeros(42,dtype=torch.float64)
        jt = torch.autograd.functional.jacobian(action_target, zero).detach()
        def fixed_target_q(delta):
            changed = probe.clone()
            changed[:42] = probe[:42] + delta / actor.initial_halfspan
            changed[42:84] = probe[42:84] - delta / actor.initial_halfspan
            return actor(changed).tanh()
        jq = torch.autograd.functional.jacobian(fixed_target_q, zero).detach()
        residual_jac = torch.autograd.functional.jacobian(
            lambda z: MLP.forward(actor,z*actor.physical_mask),probe).detach()
        expected_q = ((1-actor(probe).tanh().square())[:,None] *
                      residual_jac[:,:42] / actor.initial_halfspan)
        check(f'probe{k} coherent fixed-T q derivative is only physical MLP',
              torch.allclose(jq,expected_q,rtol=1e-13,atol=1e-15),
              maximum_error=float((jq-expected_q).abs().max()))
        expected_t = -(1-actor(probe).tanh().square())*actor.gain()/actor.initial_halfspan
        check(f'probe{k} coherent target derivative equals inhibitory diagonal',
              torch.allclose(jt,torch.diag(expected_t),rtol=1e-13,atol=1e-15))
        jm = torch.autograd.functional.jacobian(lambda z: actor(z).tanh(), probe).detach()
        off = jt - torch.diag(jt.diag())
        max_off = max(max_off,float(off.abs().max()))
        max_prev = max(max_prev,float(jm[:,195:237].abs().max()))
        min_q = min(min_q,float(jm[:,:42].abs().max()))
        check(f'probe{k} target action diagonal negative', bool((jt.diag() < 0).all()), min=float(jt.diag().min()), max=float(jt.diag().max()))
        # Memory state [target radians, previous action]. Frozen q, 5ms decisions,
        # 2rad/s actuator; no joint-limit clipping here. Prev action has zero drive.
        memory = torch.zeros((84,84),dtype=torch.float64)
        memory[:42,:42] = torch.eye(42,dtype=torch.float64) + .01*jt
        memory[42:,:42] = jt
        rho.append(float(torch.linalg.eigvals(memory).abs().max()))
    check('all target offdiagonals exactly zero', max_off == 0., maximum=max_off)
    check('previous-action derivatives exactly zero', max_prev == 0., maximum=max_prev)
    check('actual q retained with nonzero residual gradients', min_q > 0., minimum_max_abs=min_q)
    check('initial frozen-body memory spectral radius <=1', max(rho) <= 1+1e-14, values=rho,
          limitation='initial model only; softplus has no upper bound; no full-physics stability guarantee')
    zero_residual = deepcopy(actor)
    with torch.no_grad():
        for layer in zero_residual.layers:
            layer.weight.zero_(); layer.bias.zero_()
    def zero_residual_q(delta):
        changed = xx.clone()
        changed[:42] += delta / actor.initial_halfspan
        changed[42:84] -= delta / actor.initial_halfspan
        return zero_residual(changed).tanh()
    def zero_residual_target(delta):
        changed = xx.clone()
        changed[42:84] += delta / actor.initial_halfspan
        changed[126:168] += delta / actor.initial_halfspan
        return zero_residual(changed).tanh()
    zq = torch.autograd.functional.jacobian(zero_residual_q,torch.zeros(42,dtype=torch.float64))
    zt = torch.autograd.functional.jacobian(zero_residual_target,torch.zeros(42,dtype=torch.float64))
    check('zero MLP gives exactly zero coherent fixed-T q gain for all 42 outputs',
          bool((zq == 0).all()), maximum_abs=float(zq.abs().max()))
    check('zero MLP retains strictly negative diagonal target drive and zero offdiagonals',
          bool((zt.diag()<0).all() and (zt-torch.diag(zt.diag()) == 0).all()))
    masked = actor.physical_mask
    check('only declared memory channels masked', torch.equal(torch.nonzero(masked == 0).flatten(),
          torch.tensor(list(range(42,84))+list(range(126,168))+list(range(195,237)))))
    direct = actor(xx).detach()
    check('supervised and act deterministic mean/action identical', torch.equal(s.bc_mean(model,xx[None])[0],direct) and
          torch.equal(model.act(x,v,deterministic=True)['mean'][0],direct) and
          torch.equal(model.act(x,v,deterministic=True)['action'][0],direct.tanh()))
    check('critic raw inputs unchanged', torch.equal(model.act(x,v,deterministic=True)['value'], model.value(torch.as_tensor(v)[None])))
    opt = torch.optim.Adam(actor.parameters(),lr=.001)
    bx = torch.as_tensor(rng.normal(0,.1,(6,238)))
    by = torch.as_tensor(rng.uniform(-.5,.5,(6,42)))
    gain_before = actor.raw_gain.detach().clone()
    for step in range(2):
        opt.zero_grad(); loss=s.bounded_bc_loss(model,[(bx,by)]);loss.backward()
        check(f'update{step} every gain finite nonzero gradient', bool(torch.isfinite(actor.raw_gain.grad).all() and (actor.raw_gain.grad != 0).all()))
        opt.step()
    check('two bounded MSE updates change all 42 gain parameters', bool((actor.raw_gain.detach()!=gain_before).all()))
    stressed = deepcopy(actor)
    stress_opt = torch.optim.SGD(stressed.parameters(),lr=1e6)
    stressed.raw_gain.grad = torch.linspace(-1,1,42,dtype=torch.float64)
    stress_opt.step()
    check('large SGD retains finite nonnegative softplus gains', bool(torch.isfinite(stressed.gain()).all() and (stressed.gain() >= 0).all()))
    trainer = s.make_trainer(model)
    ident = capture(c,dict(purpose='synthetic checkpoint only'),extra_paths=s.SOURCE_FILES)
    with tempfile.TemporaryDirectory(prefix='inhibitory-check-') as tmp:
        path=Path(tmp)/'model.pt'
        s.save_checkpoint(path,trainer,ident,extra=dict(bc_optimizer=opt.state_dict()))
        expected_rng=(random.random(),np.random.random(),torch.rand(3))
        reloaded=s.new_model('ff238'); reload_trainer=s.make_trainer(reloaded)
        extra=s.load_checkpoint(path,reload_trainer,ident)
        actual_rng=(random.random(),np.random.random(),torch.rand(3))
        check('checkpoint every weight/buffer and model identity exact', model.identity()==reloaded.identity() and
              all(torch.equal(value,reloaded.state_dict()[key]) for key,value in model.state_dict().items()))
        check('checkpoint full RNG restoration exact',expected_rng[0]==actual_rng[0] and expected_rng[1]==actual_rng[1] and torch.equal(expected_rng[2],actual_rng[2]))
        restored_opt=torch.optim.Adam(reloaded.actor.parameters(),lr=.001);restored_opt.load_state_dict(extra['bc_optimizer'])
        check('checkpoint BC Adam gain moments and step exact', all(torch.equal(opt.state[actor.raw_gain][key],restored_opt.state[reloaded.actor.raw_gain][key]) for key in ('step','exp_avg','exp_avg_sq')))
        check('checkpoint PPO optimizer states remain fresh',not reload_trainer.actor_optimizer.state and not reload_trainer.critic_optimizer.state)
        deployed,_=s.load_student(path,c)
        check('deployment uses identity H before exact state load',deployed.identity()==model.identity())
        payload=torch.load(path,weights_only=False)
        old_error=deepcopy(payload)
        old_error['model_identity']['actor_input_transform']='physical_mlp_plus_trainable_inhibitory_error_v1'
        old_error['model_identity']['initialization_prior']='declared teacher error relaxation; not anatomical'
        for key in ('inhibitory_input_range','inhibitory_input','residual_prior'):
            old_error['model_identity'].pop(key)
        # Same class and parameter shapes: identity must reject before any restore.
        old_error['state']['model']['actor.raw_gain'].add_(123.)
        old_error_path=Path(tmp)/'old_error.pt';torch.save(old_error,old_error_path)
        before_reject={key:value.clone() for key,value in reloaded.state_dict().items()}
        with patch.object(reload_trainer,'restore',side_effect=AssertionError('old v1 reached parameter restore')):
            rejects('PPO rejects old error v1 before parameter restore',
                    lambda:s.load_checkpoint(old_error_path,reload_trainer,ident))
        check('old error v1 rejection leaves all model parameters and buffers exact',
              all(torch.equal(value,reloaded.state_dict()[key]) for key,value in before_reject.items()))
        with patch.object(s,'ActorCritic',side_effect=AssertionError('old v1 reached model construction')):
            rejects('deployment rejects old error v1 before model construction',
                    lambda:s.load_student(old_error_path,c))
        raw=ActorCritic('ff238',critic_obs_dim=255);raw.actor=MLP(238,128,42)
        raw_identity={key:value for key,value in payload['model_identity'].items() if key in ('architecture','obs_dim','critic_obs_dim','logstd_bounds','dtype')}
        old=deepcopy(payload);old['model_identity']=raw_identity;old['state']['model']=raw.state_dict()
        raw_path=Path(tmp)/'raw.pt';torch.save(old,raw_path)
        rejects('deployment rejects old raw FF identity',lambda:s.load_student(raw_path,c))
        rejects('PPO rejects old raw FF identity',lambda:s.load_checkpoint(raw_path,reload_trainer,ident))
        rejects('direct strict state load rejects raw FF weights',lambda:reloaded.load_state_dict(raw.state_dict()))
        bad=deepcopy(payload);bad['model_identity']['gain_init_H'][0]+=.01
        bad_path=Path(tmp)/'badH.pt';torch.save(bad,bad_path)
        rejects('deployment rejects physical H mismatch',lambda:s.load_student(bad_path,c))
    check('legacy GRU remains original raw actor and identity',s.new_model('gru238').identity()==ActorCritic('gru238',critic_obs_dim=255).identity())
    check('raw495 remains original MLP',type(ActorCritic('ff495').actor) is MLP)
    check('all compiled physical arrays unchanged',all(sha256_bytes(np.asarray(getattr(c.plant,key)))==old['sha256'] for key,old in physical_before['compiled_arrays'].items()))
    outputs=dict(status='PASS',all_passed=all(r['passed'] for r in rows),passed=len(rows),total=len(rows),checks=rows,
        model_identity=model.identity(),command=[sys.executable,'-B',str(Path(__file__).resolve())],
        source_hashes={str(p.resolve()):sha256(p) for p in s.SOURCE_FILES+[Path(__file__).resolve()]},
        physical_steps=0,interpretation='replacement ANN architecture prior, not anatomical or walking evidence; initial frozen-body stability only')
    atomic_json(HERE/'INHIBITORY_CHECKS.json',outputs)
    print(json.dumps(dict(all_passed=True,passed=len(rows),report=str(HERE/'INHIBITORY_CHECKS.json'))))


def sha256_bytes(value):
    import hashlib
    return hashlib.sha256(value.tobytes()).hexdigest()


if __name__=='__main__':
    main()
