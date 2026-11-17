"""Train and archive a feed-forward FlyGym motor student from same-state examples."""
import argparse
from hashlib import file_digest
import json
from pathlib import Path
import os

import numpy as np
import torch

from cases import TRAIN, DEVELOP, CORRECT
from model import BodyPolicy

HERE = Path(__file__).resolve().parent
WIDTH = 192
EPOCHS = 40
BATCH = 512


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def collect(paths, teacher):
    x, y, w, identities = [], [], [], []
    for path in paths:
        if not (path/'RESULT.json').exists():
            raise FileNotFoundError(path)
        result = read(path/'RESULT.json')
        if teacher and not result['pass_']:
            print(f'Exclude unqualified teacher {path.name}: {result["gates"]}', flush=True)
            continue
        trace_path = path/'TRACE.npz'
        assert sha(trace_path) == result['trace_sha256']
        with np.load(trace_path, allow_pickle=False) as z:
            features, labels = z['x'], z['y']
        assert features.shape[0] == labels.shape[0] and labels.shape[1] == 42
        assert np.isfinite(features).all() and np.isfinite(labels).all()
        x.append(features)
        y.append(labels)
        # Static windows must retain substantial weight despite their low joint variance.
        sample_weight = np.where(np.abs(features[:, :2]).max(1) > .04, 1.0, 1.6)
        w.append(sample_weight.astype(np.float32))
        identities.append(str(trace_path.resolve())+':'+result['trace_sha256'])
    return (*[torch.from_numpy(np.concatenate(a)) for a in (x,y,w)], identities)


def save(out, name, model, opt, generator, epoch, data_sources):
    path = out / name
    temporary = path.with_suffix('.partial')
    torch.save({'state': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch,
                'rng': generator.get_state(), 'n_input': model.center.numel(), 'width': WIDTH,
                'envelope': model.envelope, 'no_control': model.no_control,
                'data_source': data_sources}, temporary)
    os.replace(temporary, path)
    return path, sha(path)


def decompose(model, batch):
    x,y,_=batch[:3]
    predictions = model(x)
    dynamic = x[:, :2].abs().amax(1) > .04
    error = ((predictions-y).square().mean(1)).sqrt()
    return {'rmse': float((error.square().mean()).sqrt()),
            'move_rmse': float(error[dynamic].square().mean().sqrt()),
            'quiet_rmse': float(error[~dynamic].square().mean().sqrt())}


def fit(out, sources, continuation, extras, envelope, no_control):
    torch.manual_seed(3217)
    generator = torch.Generator().manual_seed(3458)
    x,y,w,roots = collect([sources/c.name for c in TRAIN], True)
    dev = collect([sources/c.name for c in DEVELOP], True)
    out.mkdir()
    model=BodyPolicy(x.shape[1],WIDTH,envelope,no_control)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    prior_epochs=0
    if continuation:
        prior=torch.load(continuation,weights_only=True,map_location='cpu')
        missing, unexpected = model.load_state_dict(prior['state'],strict=False)
        assert not unexpected and all(k == 'target_bound' for k in missing)
        same_objective = (prior.get('envelope',False) == envelope and prior.get('no_control',False) == no_control)
        if same_objective:
            optimizer.load_state_dict(prior['optimizer'])
        generator.set_state(prior['rng'])
        prior_epochs=prior['epoch']
        for path in extras:
            extra=collect([path/c.name for c in CORRECT],False)
            x,y,w=(torch.cat((a,b)) for a,b in zip((x,y,w),extra[:3],strict=True))
            roots.extend(extra[3])
    else:
        model.center.copy_(x.mean(0))
        model.scale.copy_(x.std(0).clamp_min(.015))
        model.target_scale.copy_(y.std(0).clamp_min(.045))
    if envelope:
        model.target_bound.copy_((1.15*y.abs().amax(0)).clamp_min(.06))
    ident={'train':roots,'dev':dev[3],'training_size':len(x),'dev_size':len(dev[0]),
           'epochs':EPOCHS,'previous':sha(continuation) if continuation else None,
           'envelope':envelope,'no_control':no_control,
           'optimizer_restored':bool(continuation and same_objective)}
    (out/'INPUTS.json').write_text(json.dumps(ident,indent=2)+'\n')
    path,digest=save(out,'start.pt',model,optimizer,generator,prior_epochs,ident)
    best=None
    with (out/'JOURNAL.jsonl').open('x',buffering=1) as log:
        for epoch in range(1,EPOCHS+1):
            loss_sum=0.
            for idx in torch.randperm(len(x),generator=generator).split(BATCH):
                inp=x[idx].clone()
                # The current actuator setpoint is observable at deployment. Add a small
                # tolerance so the one-step loss does not learn a perfect action echo.
                inp[:,86:128]+=torch.randn((len(idx),42),generator=generator)*(.012 if envelope else .003)
                output=model(inp)
                error=(output-y[idx])/model.target_scale
                loss=(error.square().mean(1)*w[idx]).mean()
                if not bool(torch.isfinite(loss)):
                    raise ValueError('nonfinite supervised loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.)
                if not bool(torch.isfinite(grad)):
                    raise ValueError('nonfinite gradient')
                optimizer.step()
                loss_sum+=float(loss.detach())*len(idx)
            with torch.no_grad():
                scores=decompose(model,dev)
            ep=prior_epochs+epoch
            path,digest=save(out,f'epoch{ep:03d}.pt',model,optimizer,generator,ep,ident)
            log.write(json.dumps(dict(epoch=ep,train_loss=loss_sum/len(x), dev=scores,
                                      sha256=digest,path=path.name,optimizer_has_state=bool(optimizer.state)))+'\n')
            metric=scores['move_rmse']+.5*scores['quiet_rmse']
            if best is None or metric<best['score']:
                best=dict(path=path.name,sha256=digest,score=metric,epoch=ep,metrics=scores)
            if epoch%10==0:
                print('epoch',ep,best,flush=True)
    (out/'BEST.json').write_text(json.dumps(best,indent=2)+'\n')
    ckpt=torch.load(out/best['path'],weights_only=True,map_location='cpu')
    assert ckpt['optimizer']['state'] and sha(out/best['path'])==best['sha256']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    parser.add_argument('--teachers',type=Path,default=HERE/'TEACHER')
    parser.add_argument('--continue-from',type=Path)
    parser.add_argument('--extra',type=Path,action='append',default=[])
    parser.add_argument('--envelope',action='store_true')
    parser.add_argument('--no-control',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    fit(args.out,args.teachers,args.continue_from,args.extra,args.envelope,args.no_control)
