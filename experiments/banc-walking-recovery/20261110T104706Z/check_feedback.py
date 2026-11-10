"""Bounded offline checks. No simulation ticks, gait evaluation or main training.

The only physical writes load saved OFFLINE derivative fixtures; no motion claim.
Temporary tiny-training outputs are removed. Final report pins executable sources.
"""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import random
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import feedback_bc as f
import student_wave as sw
from identity import HERE, atomic_json, sha256, verify
from ppo import load_checkpoint


def same(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def teacher_fd(c, meta):
    trace=Path(next(p for p in meta['teacher_reference_evidence']['files'] if p.endswith('TRACE_1MS.npz')))
    with np.load(trace,allow_pickle=False) as z:
        qpos,qvel,target=(z[k].copy() for k in ('qpos','qvel','target'))
    indices=np.linspace(1100,len(qpos)-100,8,dtype=int)
    original=sw.continuity_snapshot(c,np.zeros(42),None)
    rows=[]
    def forbidden(*args,**kwargs):
        raise AssertionError('OFFLINE fixture must never step simulation')
    # Core resets and FK/forward are allowed only in explicitly labelled fixtures.
    import mujoco as mj
    with patch.object(c,'tick',forbidden),patch.object(mj,'mj_step',forbidden):
        for index in indices:
            c.reset_from(qpos[index],qvel[index])
            fixture=sw.continuity_snapshot(c,np.zeros(42),None)
            direction=np.random.default_rng(1337+int(index)).choice([-1.,1.],42)
            def query(t):
                teacher=sw.StaticTeacher(c,sw.teacher_config(2))
                c.actuator.target_rad[:]=t
                before=sw.continuity_snapshot(c,np.zeros(42),None)
                action,diag=teacher.action(c,.05)
                after=sw.continuity_snapshot(c,np.zeros(42),None)
                if not same(before,after):
                    raise AssertionError('teacher action mutated actual physical/target/history state')
                return action,diag
            baseline,diag=query(target[index])
            for eps in (1e-4,1e-3,2e-3):
                ap,dp=query(target[index]+eps*direction)
                am,dm=query(target[index]-eps*direction)
                bounds=c.actuator.target_bounds_rad
                mask=np.abs(baseline)<.8
                for sign,aa in ((1,ap),(-1,am)):
                    tt=target[index]+sign*eps*direction
                    endpoint=tt+.005*2*aa
                    mask &= (tt>bounds[:,0]+1e-6)&(tt<bounds[:,1]-1e-6)
                    mask &= (endpoint>bounds[:,0]+1e-6)&(endpoint<bounds[:,1]-1e-6)
                    mask &= np.abs(aa)<1-1e-8
                n=int(mask.sum())
                if not n: raise AssertionError('fixture lacks interior teacher outputs')
                error=float(np.max(np.abs((ap-am)[mask]/(2*eps)+10*direction[mask])))
                rows.append(dict(fixture_trace_index=int(index),phase_s=.05,epsilon=eps,
                                 interior_actions=n,max_derivative_error_per_rad=error,
                                 fresh_teacher_per_query=True,physical_ticks=0))
                if error>1e-8: raise AssertionError('complete teacher FD differs from -10')
            # The fixture writes are restored without advancing time.
            c.actuator.target_rad[:]=fixture['target']
        c.reset_from(original['qpos'],original['qvel'])
        c.d.ctrl[:]=original['ctrl'];c.actuator.target_rad[:]=original['target']
        c.buf[:]=original['history'];c._padz_prev[:]=original['padz_previous']
        c._pad_rate[:]=original['pad_rate'];c.requested_rate[:]=original['requested_rate']
        c.d.time=float(original['time'])
    if not same(original,sw.continuity_snapshot(c,np.zeros(42),None)):
        raise AssertionError('offline fixture state restoration mismatch')
    return dict(trace=str(trace),trace_sha256=sha256(trace),fixtures=8,queries=56,
                rows=rows,physical_ticks=0,fixture_writes='OFFLINE saved qpos/qvel only; restored',
                history='fresh identical teacher q_des/rest/shadow initialization before each target assignment',
                scope='same body/fresh frozen teacher history; not dataset teacher history or physical rollout')


def main():
    sw.seed_all(6331)
    checks=[]
    def check(name,condition,**extra):
        checks.append(dict(name=name,passed=bool(condition),**extra))
        if not condition: raise AssertionError(name)
    c=sw.Core(adhesion=False)
    config=f.pair_config()
    meta,data,pairs,selection=f.prepare(f.BASE,c,config)
    counts=[{k:e[k] for k in ('episode','split','total_rows','eligible_rows')} for e in selection['episodes']]
    print(json.dumps(dict(eligible_counts=counts)),flush=True)
    check('episode 0/1 training and episode 2 heldout',[(e['episode'],e['split']) for e in counts]==[(0,'train'),(1,'train'),(2,'validation')])
    check('all original BC rows retained',sum(len(x) for x,y in data['train'])==3840 and sum(len(x) for x,y in data['validation'])==1920)
    half,q0,bounds=f.geometry(c)
    for entry,(x,y) in zip(meta['episodes'],data['train']+data['validation']):
        p=f.make_pairs(x.numpy(),y.numpy(),half,q0,bounds,config,entry['episode'])
        ix=p['indices'];delta=config['epsilon']*p['direction']
        check('episode %d both target blocks correct sign and scale'%entry['episode'],
              all(np.allclose(p['plus'][:,a:b]-x.numpy()[ix,a:b],delta/half,atol=1e-15,rtol=0) and
                  np.allclose(p['minus'][:,a:b]-x.numpy()[ix,a:b],-delta/half,atol=1e-15,rtol=0) for a,b in ((42,84),(126,168))))
        changed=np.zeros(238,dtype=bool);changed[42:84]=True;changed[126:168]=True
        check('episode %d fixed body/history and previous action'%entry['episode'],
              np.array_equal(p['plus'][:,~changed],x.numpy()[ix][:,~changed]) and np.array_equal(p['minus'][:,~changed],x.numpy()[ix][:,~changed]))
        # Recover q independently from each target/error block: perturbation cancels.
        check('episode %d independent target coherency'%entry['episode'],
              np.allclose((p['plus'][:,126:168]-p['plus'][:,42:84]),
                          (x.numpy()[ix,126:168]-x.numpy()[ix,42:84]),atol=1e-15,rtol=0))
        check('episode %d exact corrective labels'%entry['episode'],np.allclose(p['expected_difference'],-20*delta,atol=2e-16,rtol=0))
        check('episode %d fixed deterministic independent directions'%entry['episode'],
              np.array_equal(p['direction'],f.make_pairs(x.numpy(),y.numpy(),half,q0,bounds,config,entry['episode'])['direction']))
    fd=teacher_fd(c,meta)
    check('complete teacher finite differences at eight frozen fixtures',len(fd['rows'])==24)
    print(json.dumps(dict(teacher_FD_max_error=max(r['max_derivative_error_per_rad'] for r in fd['rows']),fixtures=8)),flush=True)
    # A tiny analytic linear network makes a sign error or detached pair gradient detectable.
    class LinearActor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(42,dtype=torch.float64))
        def forward(self,x): return x[:,126:168]*self.weight
    class Tiny: pass
    tiny=Tiny();tiny.actor=LinearActor()
    xx=np.zeros((8,238));yy=np.zeros((8,42));xx[:,126:168]=(bounds.mean(axis=1)-q0)/half
    pp=f.make_pairs(xx,yy,half,q0,bounds,config,917)
    tp=[{k:torch.as_tensor(pp[k]) for k in ('plus','minus','expected_difference')}]
    loss=f.pair_loss(tiny,tp);loss.backward()
    analytic=float(tiny.actor.weight.grad[0]);step=1e-5
    with torch.no_grad():
        tiny.actor.weight[0]=step;lp=float(f.pair_loss(tiny,tp))
        tiny.actor.weight[0]=-step;lm=float(f.pair_loss(tiny,tp));tiny.actor.weight.zero_()
    numeric=(lp-lm)/(2*step)
    check('pair gradient matches independent finite difference and corrects negative slope',analytic>0 and abs(numeric-analytic)<1e-10,analytic=analytic,numeric=numeric)
    opt=torch.optim.Adam(tiny.actor.parameters(),lr=.02)
    initial=float(loss.detach())
    for _ in range(20):
        opt.zero_grad();ll=f.pair_loss(tiny,tp);ll.backward();opt.step()
    final=float(f.pair_loss(tiny,tp).detach())
    check('tiny corrective optimization improves pair loss',final<initial and bool((tiny.actor.weight<0).all()),initial=initial,final=final)
    # Bounded complete train path on real base: two steps only, no physical stepping.
    report_path=HERE/'FEEDBACK_CHECKS.json'
    atomic_json(report_path,dict(all_passed=False,status='bounded checks in progress'))
    with tempfile.TemporaryDirectory(prefix='feedback-check-') as temporary:
        args=f.parser().parse_args(['train','--out',str(Path(temporary)/'tiny'),'--steps','2','--lr','0.000001'])
        with patch.object(f,'check_preflight',lambda:None),redirect_stdout(io.StringIO()):
            result=f.train(args)
        path=Path(result['checkpoint'])
        model,payload=sw.load_student(path,c)
        check('unchanged raw FF238 deploys through frozen student loader',model.architecture=='ff238' and payload['model_identity']==sw.new_model('ff238').identity())
        parent,_=sw.load_student(f.PARENT,c)
        check('model-only warmstart leaves critic and exploration std unchanged',
              same(parent.critic.state_dict(),model.critic.state_dict()) and torch.equal(parent.raw_logstd,model.raw_logstd))
        check('fresh PPO optimizers and retained BC Adam',not payload['state']['actor']['state'] and not payload['state']['critic']['state'] and bool(payload['extra']['bc_optimizer']['state']))
        check('full base identities and parent closure retained',len(payload['extra']['collection_identities'])==3 and payload['extra']['dataset_sha256']==sha256(f.BASE/'DATASET.json') and payload['extra']['parent_sha256']==sha256(f.PARENT))
        check('validation pairs never supplied to training loss',"losses(model,data['train'],pairs['train'],args.pair_weight)" in Path(f.__file__).read_text())
        check('actual tiny training improves total objective',result['final']['train_total']<result['initial']['train_total'])
        trainer=sw.make_trainer(sw.new_model('ff238'),payload['config'])
        extra=load_checkpoint(path,trainer,payload['identity'],restore_rng=True)
        adam=torch.optim.Adam(trainer.model.actor.parameters(),lr=args.lr);adam.load_state_dict(extra['bc_optimizer'])
        check('atomic checkpoint exact model and BC optimizer reload',same(trainer.model.state_dict(),model.state_dict()) and same(adam.state_dict(),payload['extra']['bc_optimizer']) and not list(path.parent.glob('*.tmp')))
        expected=(np.random.random(4),random.random(),torch.rand(4))
        load_checkpoint(path,trainer,payload['identity'],restore_rng=True)
        actual=(np.random.random(4),random.random(),torch.rand(4))
        check('exact numpy python torch RNG reload',same(expected,actual))
        # Identical next Adam step demonstrates retained moments/counters matter.
        clone=copy.deepcopy(trainer.model);other=torch.optim.Adam(clone.actor.parameters(),lr=args.lr)
        other.load_state_dict(copy.deepcopy(adam.state_dict()))
        for mm,oo in ((trainer.model,adam),(clone,other)):
            oo.zero_grad();ll,_,_=f.losses(mm,data['train'],pairs['train'],config['pair_weight']);ll.backward();oo.step()
        check('BC Adam resumed next update exactly',same(trainer.model.state_dict(),clone.state_dict()) and same(adam.state_dict(),other.state_dict()))
        tiny_result={k:result[k] for k in ('steps','initial','final','checkpoint_reload_exact')}
    sources={str(p.resolve()):sha256(p) for p in sw.SOURCE_FILES+[Path(f.__file__),Path(__file__)]}
    report=dict(all_passed=all(r['passed'] for r in checks),checks=checks,source_hashes=sources,
                eligible_counts=counts,pair_config=config,teacher_complete_FD=fd,tiny_training=tiny_result,
                scope='S2/S5 offline checks only; no gait, no full training, no physical stability claim',
                physical_simulation_steps=0)
    atomic_json(report_path,report)
    print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),report=str(report_path))),flush=True)
    return report

if __name__=='__main__': main()
