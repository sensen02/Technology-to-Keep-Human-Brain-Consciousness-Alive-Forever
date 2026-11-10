"""Latent loss wiring preflight: synthetic fixtures, Core compile, ZERO physics steps.
No real-data fitting, collection, handover, or long training. Temporary files removed.
"""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import latent_bc as l
import student_wave as sw
from identity import HERE, atomic_json, capture, verify, sha256
from ppo import load_checkpoint


def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    rows=[]
    def check(name,condition,**details):
        rows.append(dict(name=name,passed=bool(condition),**details))
        if not condition: raise AssertionError(name)
    def rejects(name,fn):
        try: fn()
        except (ValueError,RuntimeError,FileExistsError): check(name,True)
        else: check(name,False)
    sw.seed_all(20261110)
    defaults=l.parser().parse_args(['train','--dataset','/unused','--out','/unused'])
    old=sw.parser().parse_args(['bc','--dataset','/unused','--out','/unused'])
    check('same old actual default seed; fixed comparison defaults',defaults.seed==old.seed==20261110 and defaults.lr==.001 and defaults.steps==1000 and defaults.action_epsilon==.001)
    evidence=sw.teacher_preflight()
    check('parent physical teacher gates and identity pass',evidence['all_gates'])
    y=torch.tensor([-1.,-.999,-.998,0.,.998,.999,1.],dtype=torch.float64)
    z=l.latent_targets(y)
    check('finite atanh exact endpoints and clip boundary',torch.isfinite(z).all() and torch.equal(z[:2],z[0].expand(2)) and torch.equal(z[-2:],z[-1].expand(2)) and torch.allclose(z.tanh(),y.clamp(-.999,.999),atol=1e-15,rtol=0))
    rejects('epsilon prereg fixed',lambda:l.latent_targets(y,.002))
    rejects('reject nonfinite raw labels',lambda:l.latent_targets(torch.tensor([float('nan')],dtype=torch.float64)))
    m=torch.tensor([-10.],dtype=torch.float64,requires_grad=True);target=torch.tensor([.5],dtype=torch.float64)
    latent=(m-l.latent_targets(target)).square().mean(); gl=torch.autograd.grad(latent,m)[0]
    bounded=(m.tanh()-target).square().mean();gb=torch.autograd.grad(bounded,m)[0]
    check('saturated mean latent gradient nonzero while bounded gradient small',gl.abs().item()>1 and gb.abs().item()<1e-6,latent_gradient=gl.item(),bounded_gradient=gb.item(),interpretation='analytical saturation weighting difference; not numerical bug or pure loss rescale')
    mu=torch.randn(5,42,dtype=torch.float64,requires_grad=True); yy=torch.rand_like(mu)*2-1;zz=l.latent_targets(yy)
    mse=(mu-zz).square().mean(); sigma=.05
    nll=(-torch.distributions.Normal(mu,sigma).log_prob(zz)).mean()
    gm=torch.autograd.grad(mse,mu,retain_graph=True)[0];gn=torch.autograd.grad(nll,mu)[0]
    check('Gaussian NLL constant and analytical gradient scaling',torch.allclose(nll,mse/(2*sigma*sigma)+math.log(sigma*math.sqrt(2*math.pi)),atol=1e-12,rtol=1e-14) and torch.allclose(gn,gm/(2*sigma*sigma),atol=1e-12,rtol=1e-14) and torch.allclose(gm,2*(mu-zz)/mu.numel(),atol=1e-15,rtol=0))
    core=sw.Core(adhesion=False)
    sw.seed_all(20261110);model=sw.new_model('ff238')
    sw.seed_all(20261110);same=sw.new_model('ff238')
    check('same scratch FFv2 full model seed and physical H',equal(model.state_dict(),same.state_dict()) and np.array_equal(model.actor.initial_halfspan.numpy(),sw.physical_halfspan(core)) and model.identity()['actor_input_transform']=='physical_mlp_plus_trainable_inhibitory_target_v2')
    rng=np.random.default_rng(73);xx=rng.normal(0,.1,(6,238));yy=rng.uniform(-1,1,(6,42));yy[0,0]=1;yy[1,0]=-1
    x=torch.tensor(xx,dtype=torch.float64);y=torch.tensor(yy,dtype=torch.float64)
    loss=l.latent_loss(model,[(x,y)]);loss.backward()
    check('all actor and 42 gains trainable finite nonzero gain gradient',all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.actor.parameters()) and (model.actor.raw_gain.grad!=0).all())
    params=list(model.actor.parameters());dirs=[torch.randn_like(p) for p in params]
    norm=sum(float(d.square().sum()) for d in dirs)**.5;dirs=[d/norm for d in dirs]
    analytical=sum(float((p.grad*d).sum()) for p,d in zip(params,dirs));original=[p.detach().clone() for p in params]
    def shifted(sign):
        with torch.no_grad():
            for p,b,d in zip(params,original,dirs): p.copy_(b+sign*1e-5*d)
        return float(l.latent_loss(model,[(x,y)]).detach())
    fd=(shifted(1)-shifted(-1))/(2e-5)
    with torch.no_grad():
        for p,b in zip(params,original):p.copy_(b)
    check('full actor including gain random directional finite difference',abs(fd-analytical)<1e-7,finite_difference=fd,autograd=analytical)
    grad=model.actor.raw_gain.grad.clone();d=torch.randn_like(grad);d/=d.norm();base=model.actor.raw_gain.detach().clone()
    vals=[]
    for sign in (1,-1):
        with torch.no_grad():model.actor.raw_gain.copy_(base+sign*1e-5*d)
        vals.append(float(l.latent_loss(model,[(x,y)]).detach()))
    with torch.no_grad():model.actor.raw_gain.copy_(base)
    check('gain-only directional finite difference',abs((vals[0]-vals[1])/2e-5-float((grad*d).sum()))<1e-7)
    mm=l.metrics(model,[(x,y)]);pred=model.actor(x).tanh().detach()
    check('raw action MSE includes unclipped labels and clipping delta',mm['action_MSE']==float((pred-y).square().mean()) and abs(mm['action_MSE_raw_minus_clipped']-(float((pred-y).square().mean())-float((pred-y.clamp(-.999,.999)).square().mean())))<1e-15)
    stats=l.label_stats([(x,y)])
    check('raw endpoint and clipped fraction reported',stats['raw_exact_plus_one_count']==1 and stats['raw_exact_minus_one_count']==1 and stats['clipped_count']>=2 and stats['clipping_distortion_MSE']>0)
    source=Path(l.__file__).read_text();tree=ast.parse(source)
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train');train_source=ast.get_source_segment(source,train)
    check('no heldout fit; unchanged FFv2 construction and actor-only optimizer',"latent_loss(model,episodes['train']" in train_source and "latent_loss(model,episodes['validation']" not in train_source and "sw.new_model('ff238')" in train_source and 'Adam(model.actor.parameters()' in train_source)
    check('no phase, lookup, teacher action or physical stepping in trainer',not any(word in train_source for word in ('.tick(','.step_physics(','.obs238(','.action(','.phase','lookup')))
    obs_source=ast.get_source_segment(Path(sw.__file__).read_text(),next(n for n in ast.parse(Path(sw.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='observations'))
    check('inherited actor observes raw physical 238 without phase', 'c.obs238(previous_actual_action, target_speed=TARGET_SPEED)' in obs_source and not any(w in obs_source for w in ('phase','reference','teacher')))
    with tempfile.TemporaryDirectory(prefix='latent-bc-check-') as temp:
        root=Path(temp);dataset=root/'dataset';dataset.mkdir()
        identity=capture(core,dict(purpose='synthetic latent BC check, not gait evidence'),extra_paths=l.SOURCE_FILES)
        entries=[]
        for episode in range(3):
            n=6;y=rng.uniform(-1,1,(n,42));y[0,0]=1.;y[1,0]=-1.
            x=rng.normal(0,.1,(n,238));x[:,-1]=sw.TARGET_SPEED;x[0,-43:-1]=0.;x[1:,-43:-1]=y[:-1]
            v=np.column_stack([x,np.zeros((n,16)),np.arange(n,0,-1)*.005])
            path=dataset/f'episode_{episode}.npz';np.savez_compressed(path,X238=x,Y42=y,V255=v)
            ip=dataset/f'identity_{episode}.json';atomic_json(ip,identity)
            rp=dataset/f'report_{episode}.json';atomic_json(rp,dict(teacher_episode_condition_passed=True,kind='SYNTHETIC ONLY'))
            entries.append(dict(episode=episode,seed=episode+70,split='validation' if episode==2 else 'train',path=str(path),sha256=sha256(path),samples=n,identity_path=str(ip),identity_sha256=sha256(ip),report_path=str(rp),report_sha256=sha256(rp),trace=None,teacher_episode_condition_passed=True))
        meta=dict(format_version=1,collection_complete=True,source_hashes=sw.source_hashes(),teacher_reference_evidence=evidence,episodes=entries,all_episode_conditions_passed=True,supervision=dict(samples_per_episode=6,landing_tail_excluded=True),kind='SYNTHETIC ONLY')
        atomic_json(dataset/'DATASET.json',meta)
        args=l.parser().parse_args(['train','--dataset',str(dataset),'--out',str(root/'trained'),'--steps','2'])
        with redirect_stdout(io.StringIO()): result=l.train(args)
        payload=torch.load(root/'trained/final.pt',weights_only=False)
        check('two synthetic Adam steps finite and loss decreases',result['final']['train']['latent_MSE']<payload['extra']['curve'][0]['train']['latent_MSE'] and math.isfinite(result['final']['validation']['action_MSE']))
        check('BC Adam independent; PPO fresh counters and optimizers',len(payload['extra']['bc_optimizer']['state'])==len(list(model.actor.parameters())) and not payload['state']['actor']['state'] and not payload['state']['critic']['state'] and payload['state']['updates']==payload['state']['accepted_steps']==0)
        sw.seed_all(args.seed);scratch=sw.new_model('ff238')
        check('critic/raw std exactly unchanged',equal(scratch.critic.state_dict(),{k.removeprefix('critic.'):v for k,v in payload['state']['model'].items() if k.startswith('critic.')}) and torch.equal(scratch.raw_logstd,payload['state']['model']['raw_logstd']) and torch.allclose(scratch.logstd().exp(),torch.full((42,),.05,dtype=torch.float64)))
        loaded,loaded_payload=sw.load_student(root/'trained/final.pt',core)
        check('sw load compatible exact actor critic std model identity',equal(loaded.state_dict(),payload['state']['model']) and loaded.identity()==payload['model_identity'] and result['checkpoint_reload_exact'])
        restored=torch.optim.Adam(loaded.actor.parameters(),lr=args.lr);restored.load_state_dict(payload['extra']['bc_optimizer'])
        check('Adam state exact reload',equal(restored.state_dict(),payload['extra']['bc_optimizer']))
        trainer=sw.make_trainer(loaded);load_checkpoint(root/'trained/final.pt',trainer,payload['identity'],restore_rng=True)
        check('complete RNG restored exactly',equal(np.random.get_state(),payload['rng']['numpy']) and equal(random.getstate(),payload['rng']['python']) and torch.equal(torch.get_rng_state(),payload['rng']['torch']))
        check('full identity and base collection closure',verify(payload['identity']) and payload['extra']['dataset_sha256']==sha256(dataset/'DATASET.json') and len(payload['extra']['collection_identities'])==3)
        action=loaded.act(np.zeros(238),np.zeros(255),deterministic=True)['action']
        check('deployment tanh finite bounded action',action.shape==(1,42) and torch.isfinite(action).all() and action.abs().max()<=1)
        rejects('existing outputs refused',lambda:l.train(args))
        bad=deepcopy(meta);bad['episodes'][-1]['split']='train';atomic_json(dataset/'DATASET.json',bad)
        rejects('holdout cannot be merged into train',lambda:sw.validate_dataset(dataset))
    report=dict(all_passed=all(r['passed'] for r in rows),source_hashes=l.source_hashes(),checks=rows,
        scope='S2/S5 latent BC wiring only; two synthetic optimizer steps; full Core compile; zero physical steps; no real-data fit',label_contract=l.LABEL_CONTRACT)
    atomic_json(HERE/'LATENT_CHECKS.json',report)
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(rows),report=str(HERE/'LATENT_CHECKS.json'))),flush=True)


if __name__=='__main__':main()
