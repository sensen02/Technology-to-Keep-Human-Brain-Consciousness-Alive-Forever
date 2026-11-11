"""Code-only fixed500 adapter trainer. No execution on import; no physics imports."""
from pathlib import Path
import json
import time
import subprocess
import sys
import numpy as np
import torch
from residual_motor import HERE,BASE,CHECKPOINT,ResidualMotor,load_baseline_actor,save_candidate,load_candidate,sha256

REFERENCE=BASE.parent/'20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
SOURCE_NAMES=('residual_motor.py','PREREG.json','check_residual.py','CHECK_RESULTS.json','cached_training.py','check_cache.py','TRAIN_PREREG.json','verify_saved.py','evaluate_candidate.py')


def verify_hashes(hashes):
    for p,h in hashes.items():
        if not Path(p).is_absolute() or sha256(p)!=h:raise ValueError('identity drift '+p)


def prepare(model):
    """Frozen baseline cache once per actual full teacher episode, no residual detach."""
    episodes=[];hashes={str(REFERENCE):sha256(REFERENCE),str(CHECKPOINT):sha256(CHECKPOINT),str(BASE/'models.py'):sha256(BASE/'models.py')}
    for name in ('left_full','right_full'):
        result_path=HERE.parent/'teacher'/name/'RESULT.json'
        r=json.loads(result_path.read_text())
        eligible=(r['early_stop'] is None and r['samples']['decisions']==2880 and r['strict_nonfoot_zero'] and r['foot_support_mean']>=.9 and r['com_height_min']>=1.2 and r['slip_ratio_max']<=.5 and r['cycles']['all_legs_completed'] and r['requested_sign_agreement'] and r['forward_signed_native_per_s']>=.05 and r['condition']['speed_scale']==1)
        if not eligible:raise ValueError('teacher admission failure '+name)
        path=Path(r['samples']['path'])
        if sha256(path)!=r['samples']['sha256']:raise ValueError('teacher sample hash mismatch')
        for p in (result_path,path):hashes[str(p)]=sha256(p)
        with np.load(path,allow_pickle=False) as z:
            x=torch.tensor(z['X238'],dtype=torch.float64)[:,None]
            y=torch.tensor(z['Y42'],dtype=torch.float64)[:,None]
            u=torch.tensor(z['command2'],dtype=torch.float64)[:,None]
        if x.shape!=(2880,1,238) or y.shape!=(2880,1,42) or u.shape!=(2880,1,2) or not torch.equal(u[:960],torch.zeros_like(u[:960])):raise ValueError('teacher shape/neutral prefix mismatch')
        state=model.base.initial_state(1);means=[];hidden=[]
        with torch.no_grad():
            for row in x:
                mean,state=model.base.step(row,state)
                means.append(mean);hidden.append(state)
            means=torch.stack(means);hidden=torch.stack(hidden)
            # Same concatenation arithmetic/order as production; no new weights.
            features=torch.cat((x[...,model.base.physical_mask.bool()],hidden,u),dim=-1)
            gate=torch.tanh(u.square().sum(dim=-1,keepdim=True))
        episodes.append(dict(name=name,x=x,y=y,u=u,means=means,hidden=hidden,features=features,gate=gate))
    return episodes,hashes


def cached_sequence(model,episode):
    rh=model.initial_state(1)[1];outputs=[];states=[]
    for mean,features,gate in zip(episode['means'],episode['features'],episode['gate']):
        candidate=model.gru(torch.tanh(model.encoder(features)),rh)
        rh=(1-model.ALPHA)*rh+model.ALPHA*candidate
        correction=gate*model.RESIDUAL_BOUND*torch.tanh(model.head(rh))
        outputs.append(mean+correction);states.append(rh)
    return torch.stack(outputs),torch.stack(states)


def neutral_check(model):
    with np.load(REFERENCE,allow_pickle=False) as z,torch.no_grad():
        x=torch.tensor(z['X238'],dtype=torch.float64)[:,None]
        m,h=model.sequence(x,x.new_zeros((2880,1,2)))
        checks=dict(mean=torch.equal(m[:,0],torch.tensor(z['mean'])),action=torch.equal(m[:,0].tanh(),torch.tensor(z['policy_action'])),hidden=torch.equal(h[0][:,0],torch.tensor(z['hidden_after'])))
    if not all(checks.values()):raise ValueError('neutral retention failed')
    return checks


def metrics(model,episodes):
    with torch.no_grad():
        return {e['name']:float((cached_sequence(model,e)[0][960:].tanh()-e['y'][960:]).square().mean()) for e in episodes}


def train(out):
    torch.set_num_threads(1);torch.manual_seed(20261110);np.random.seed(20261110)
    report=json.loads((HERE/'CACHE_CHECKS.json').read_text())
    if not report['all_passed']:raise ValueError('cache checks required')
    verify_hashes(report['source_hashes']);verify_hashes(report['data_hashes'])
    sources={str(HERE/n):sha256(HERE/n) for n in SOURCE_NAMES}
    sources[str(HERE/'CACHE_CHECKS.json')]=sha256(HERE/'CACHE_CHECKS.json')
    out=Path(out).resolve()
    if not out.is_relative_to(HERE):raise ValueError('output must remain inside candidate')
    out.mkdir(exist_ok=False)
    started=time.monotonic();model=ResidualMotor(load_baseline_actor())
    episodes,datahashes=prepare(model);opt=torch.optim.Adam(model.adapter_parameters(),lr=.001)
    config=dict(kind='alternate_frozen_base_residual_fixed500',steps=500,lr=.001,seed=20261110,source_hashes=sources,data_hashes=datahashes,original_model_sha256=model.identity()['baseline_sha256'],prereg=json.loads((HERE/'TRAIN_PREREG.json').read_text()),physical_evaluation='NOT_RUN',eval_only_step=500)
    (out/'IDENTITY.json').write_text(json.dumps(config,indent=2)+'\n')
    curve=[dict(step=0,metrics=metrics(model,episodes),neutral=neutral_check(model),wall_s=time.monotonic()-started)]
    log=out/'TRAINING.jsonl'
    with log.open('x') as stream:
        stream.write(json.dumps(curve[0])+'\n');stream.flush();print(json.dumps(curve[0]),flush=True)
        for step in range(1,501):
            if time.monotonic()-started>14400:raise TimeoutError('fixed4h wall budget exhausted; no continuation or selection')
            opt.zero_grad(set_to_none=True);losses={}
            for e in episodes:
                prediction,_=cached_sequence(model,e)
                loss=(prediction[960:].tanh()-e['y'][960:]).square().mean()/2
                loss.backward();losses[e['name']]=float(loss.detach())*2
                del prediction,loss
            grad=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.)
            if not torch.isfinite(grad):raise ValueError('nonfinite adapter gradient')
            if any(p.grad is not None or p.requires_grad for p in model.base.parameters()):raise ValueError('base freeze violation')
            opt.step()
            row=dict(step=step,postprime_MSE=losses,gradient_norm=float(grad),wall_s=time.monotonic()-started)
            if step%100==0:
                row['neutral']=neutral_check(model);row['metrics']=metrics(model,episodes)
            curve.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if step%25==0:print(json.dumps(row),flush=True)
            if step%100==0:
                verify_hashes(sources);verify_hashes(datahashes)
                # save_candidate includes weights/optimizer/curve through config and both RNG states.
                saved_config=dict(config,curve=curve,rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state()))
                ck=out/f'checkpoint_{step:06d}.pt'
                save_candidate(ck,model,opt,saved_config)
                from verify_saved import exact
                restored,payload=load_candidate(ck)
                if not exact(model.state_dict(),restored.state_dict()) or not exact(opt.state_dict(),payload['optimizer']) or not exact(saved_config,payload['config']):raise ValueError('live-to-saved state mismatch')
                subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py'),'--checkpoint',str(ck),'--expected-sha256',sha256(ck)],check=True)
        result=dict(status='TRAJECTORY_FIT_ONLY_NOT_PHYSICAL_TURNING',steps=500,checkpoint=str(ck),checkpoint_sha256=sha256(ck),wall_s=time.monotonic()-started,evaluate_only_step=500)
        (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();train(args.out)
