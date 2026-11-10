"""S2/S5 frozen TRAIN-only normalization checks. No simulation or long training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

import student_wave as s
from models import MLP, ActorCritic
from identity import HERE, atomic_json, sha256


def main():
    s.seed_all(763)
    rows = []
    def check(name, condition, **details):
        passed = bool(condition)
        rows.append(dict(name=name, passed=passed, **details))
        if not passed:
            raise AssertionError(name)
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, RuntimeError, KeyError):
            check(name, True)
        else:
            check(name, False)
    def equal_state(a, b):
        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal_state(x,y) for x,y in zip(a,b))
        return a == b

    with tempfile.TemporaryDirectory(prefix='frozen-normalization-check-') as tmp:
        root = Path(tmp)
        arrays = [np.full((2,238), 1.), np.full((6,238), 5.), np.full((3,238), 900.)]
        # Known two-point distribution gives population mean4/std sqrt3;
        # the target channel stays constant, including its floor behavior.
        for x in arrays:
            x[:,-1] = .075
        entries = []
        for i,x in enumerate(arrays):
            path = root/f'episode_{i}.npz'
            np.savez_compressed(path, X238=x)
            entries.append(dict(path=str(path),sha256=sha256(path),episode=i,
                                split='validation' if i == 2 else 'train'))
        meta = dict(episodes=entries)
        dataset = root/'DATASET.json'; atomic_json(dataset, meta)
        stats = s.fit_actor_normalization(meta, dataset)
        check('all TRAIN frames equally weighted across unequal episode lengths',
              stats['count']==8 and stats['mean'][0]==4. and np.isclose(stats['std'][0],np.sqrt(3.)))
        check('population variance and explicit raw-unit floor',
              stats['variance_ddof']==0 and stats['std_floor']==.05 and
              stats['variance_floor']==.05**2 and stats['scale'][-1]==.05)
        check('training provenance excludes validation files',
              len(stats['train_files'])==2 and all(e['episode']<2 for e in stats['train_files']))
        validation = Path(entries[-1]['path'])
        np.savez_compressed(validation, X238=np.full((73,238),-1e12))
        # The metadata file is deliberately held constant: only the forbidden
        # holdout payload changes, isolating whether fit reads that payload.
        real_load = np.load
        def train_only_load(path, *args, **kwargs):
            if Path(path)==validation:
                raise AssertionError('fit opened validation')
            return real_load(path,*args,**kwargs)
        with patch.object(np,'load',train_only_load):
            again = s.fit_actor_normalization(meta,dataset)
        check('arbitrary holdout perturbation cannot change fitted statistics',again==stats)
        rejects('reject zero floor',lambda:s.fit_actor_normalization(meta,dataset,0))
        rejects('reject nonfinite floor',lambda:s.fit_actor_normalization(meta,dataset,float('nan')))
        x = torch.as_tensor(np.concatenate(arrays[:2]),dtype=torch.float64)
        x = x + torch.randn_like(x)*.02
        v = torch.randn(len(x),255,dtype=torch.float64)
        for architecture in ('ff238','gru238'):
            model = s.new_model(architecture)
            before_identity = model.identity()
            model.actor.set_input_stats(stats['mean'],stats['scale'])
            check(architecture+' model identity invariant under fitting',before_identity==model.identity())
            check(architecture+' persistent statistics excluded from optimizer parameters',
                  {'actor.center','actor.scale'}<=model.state_dict().keys() and
                  not any(k in ('center','scale') for k,_ in model.actor.named_parameters()))
            check(architecture+' fitted buffers match provenance',
                  s.verify_actor_normalization(model,stats,meta,dataset))
            bad = deepcopy(stats); bad['count']+=1
            rejects(architecture+' refuse altered normalization provenance',
                    lambda:s.verify_actor_normalization(model,bad,meta,dataset))
            with torch.no_grad():
                model.actor.center[0]+=1
            rejects(architecture+' refuse altered normalization buffers',
                    lambda:s.verify_actor_normalization(model,stats,meta,dataset))
            model.actor.set_input_stats(stats['mean'],stats['scale'])
            normalized = (x-torch.tensor(stats['mean'],dtype=torch.float64))/torch.tensor(stats['scale'],dtype=torch.float64)
            check(architecture+' affine inverse recovers raw observation',
                  torch.allclose(model.actor.normalize(x)*model.actor.scale+model.actor.center,x,atol=1e-15,rtol=1e-15))
            extreme = model.actor.center+1000*model.actor.scale
            check(architecture+' no standardized clipping or previous-action mask',
                  torch.allclose(model.actor.normalize(extreme),torch.full((238,),1000.,dtype=torch.float64)))
            seen=[]
            hook=model.critic.layers[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
            model.value(v); hook.remove()
            check(architecture+' critic receives all255 raw features',torch.equal(seen[0],v))
            if architecture=='ff238':
                manual = MLP(238,128,42).double()
                manual.layers.load_state_dict(model.actor.layers.state_dict())
                check('FF output matches manual normalized raw MLP',torch.equal(model.actor(x),manual(normalized)))
                # A scalar linear response at the first layer exposes scaling,
                # centering, and retention of previous-action channels directly.
                captured=[]
                hook=model.actor.layers[0].register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
                model.actor(x); hook.remove()
                check('FF first linear layer sees manual standardized inputs',torch.equal(captured[0],normalized))
            else:
                reset=torch.zeros(len(x),1,dtype=torch.bool);reset[3]=True
                valid=torch.ones(len(x),1,dtype=torch.bool);valid[5]=False
                seq,final=model.actor.sequence(x[:,None],model.actor.initial_state(1),reset,valid)
                state=model.actor.initial_state(1);outputs=[]
                encoded=[]
                hook=model.actor.encoder.register_forward_pre_hook(lambda module,args:encoded.append(args[0].detach().clone()))
                for i,xx in enumerate(x):
                    mean,candidate=model.actor.step(xx[None],state,reset[i])
                    state=torch.where(valid[i,:,None],candidate,state);outputs.append(mean)
                hook.remove()
                check('GRU step and sequence exact with reset and padding',torch.equal(seq,torch.stack(outputs)) and torch.equal(final,state))
                check('GRU encoder sees normalized inputs each step',torch.equal(torch.cat(encoded),normalized))
            buffers={k:t.clone() for k,t in model.actor.named_buffers()}
            model.eval()
            for _ in range(3): s.bc_mean(model,x*2)
            check(architecture+' eval never adapts statistics',all(torch.equal(t,dict(model.actor.named_buffers())[k]) for k,t in buffers.items()))
            trainer=s.make_trainer(model)
            trainer.actor_optimizer.zero_grad();trainer.critic_optimizer.zero_grad()
            loss=s.bc_mean(model,x).square().mean()+model.value(v).square().mean()+model.logstd().square().mean()
            loss.backward()
            check(architecture+' finite gradients under normalized actor and raw critic',
                  all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            trainer.actor_optimizer.step();trainer.critic_optimizer.step()
            checkpoint=root/f'{architecture}.pt';identity=dict(test='synthetic normalization only')
            s.save_checkpoint(checkpoint,trainer,identity,extra=dict(actor_normalization=stats))
            expected_random=(random.random(),np.random.random(),torch.rand(5))
            clone=s.new_model(architecture);fresh=s.make_trainer(clone)
            s.load_checkpoint(checkpoint,fresh,identity,restore_rng=True)
            got_random=(random.random(),np.random.random(),torch.rand(5))
            check(architecture+' RNG checkpoint roundtrip',expected_random[:2]==got_random[:2] and torch.equal(expected_random[2],got_random[2]))
            check(architecture+' model buffers and warm optimizer checkpoint roundtrip',
                  equal_state(trainer.snapshot(),fresh.snapshot()))
            check(architecture+' predictions exact after reload',torch.equal(s.bc_mean(model,x),s.bc_mean(clone,x)))
            payload=torch.load(checkpoint,weights_only=False)
            payload['model_identity'].pop('actor_input_transform')
            old=root/f'old_{architecture}.pt';torch.save(payload,old)
            before={k:t.clone() for k,t in clone.state_dict().items()}
            rejects(architecture+' raw checkpoint identity rejected before mutation',lambda:s.load_checkpoint(old,fresh,identity))
            check(architecture+' rejected old checkpoint leaves model unchanged',equal_state(before,clone.state_dict()))
        legacy=ActorCritic('ff495')
        check('legacy ff495 actor remains raw without normalization buffers',legacy.identity()['actor_input_transform']=='identity' and not list(legacy.actor.named_buffers()))
    # Read actual existing dataset only; never train or simulate it.
    actual_path=HERE.parent/'20261110T094051Z/SMOOTH_DATASET/DATASET.json'
    actual_meta=s.validate_dataset(actual_path.parent)
    actual=s.fit_actor_normalization(actual_meta,actual_path)
    check('real smooth dataset statistics include episodes0and1 only',actual['count']==3840 and [e['episode'] for e in actual['train_files']]==[0,1])
    check('real smooth dataset finite scales at least floor',np.isfinite(actual['scale']).all() and min(actual['scale'])>=.05)
    sources=s.SOURCE_FILES+[Path(__file__).resolve()]
    report=dict(all_passed=all(r['passed'] for r in rows),passed=len(rows),total=len(rows),checks=rows,
        source_hashes={str(p.resolve()):sha256(p) for p in sources},
        command=[sys.executable,'-B',str(Path(__file__).resolve())],actual_train_statistics=actual,
        not_executed=['long BC','collection','simulation','walking evaluation'],
        interpretation='engineering correctness only; normalization does not fix distribution coverage or prove physical success')
    atomic_json(HERE/'NORMALIZATION_CHECKS.json',report)
    print(json.dumps(dict(all_passed=report['all_passed'],passed=len(rows),report=str(HERE/'NORMALIZATION_CHECKS.json'))))


if __name__=='__main__':
    main()
