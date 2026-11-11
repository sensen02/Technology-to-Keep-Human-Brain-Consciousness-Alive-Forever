"""Fresh-process saved artifact validation, no optimizer steps/physical execution."""
import argparse
import json
import numpy as np
import torch
from residual_motor import load_candidate,sha256


def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact(v,w) for v,w in zip(a,b))
    return type(a)==type(b) and a==b


def verify(path,digest):
    torch.set_num_threads(1)
    if sha256(path)!=digest:raise ValueError('checkpoint digest differs')
    model,payload=load_candidate(path);cfg=payload['config']
    from cached_training import verify_hashes,neutral_check
    verify_hashes(cfg['source_hashes']);verify_hashes(cfg['data_hashes'])
    curve=cfg['curve'];step=curve[-1]['step']
    if step not in (100,200,300,400,500) or [r['step'] for r in curve]!=list(range(step+1)):raise ValueError('fixed contiguous curve mismatch')
    if cfg['steps']!=500 or cfg['eval_only_step']!=500 or cfg['lr']!=.001:raise ValueError('fixed budget changed')
    if not exact(model.state_dict(),payload['state_dict']):raise ValueError('weight reload differs')
    opt=torch.optim.Adam(model.adapter_parameters(),lr=.001);opt.load_state_dict(payload['optimizer'])
    if not exact(opt.state_dict(),payload['optimizer']):raise ValueError('optimizer reload differs')
    for s in opt.state.values():
        if float(s['step'])!=step or not torch.isfinite(s['exp_avg']).all() or not torch.isfinite(s['exp_avg_sq']).all():raise ValueError('Adam counters/moments invalid')
    torch.set_rng_state(cfg['rng']['torch']);np.random.set_state(cfg['rng']['numpy'])
    if not exact(torch.get_rng_state(),cfg['rng']['torch']) or not exact(np.random.get_state(),cfg['rng']['numpy']):raise ValueError('RNG reload differs')
    neutral=neutral_check(model)
    print(json.dumps(dict(checkpoint=str(path),step=step,weights_optimizer_rng_curve_reload_exact=True,neutral=neutral)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--expected-sha256',required=True);a=p.parse_args();verify(a.checkpoint,a.expected_sha256)
