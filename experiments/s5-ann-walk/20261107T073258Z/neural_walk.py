"""S5 replacement ANN: causal target-history neural dynamics, no phase clock at deployment.

The reference is used ONLY for offline learning and the first 1 s of physics.
An ordinary MLP learns the next actuator target from its last 200 ms of targets.
Training includes unrolled self predictions and perturbed histories, not merely
one-step regression. The MLP is continuously observed during teacher warmup.
This is a learned replacement motor controller, NOT evidence of connectome control.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = ROOT / 'experiments/s5-ann-incremental/20261107T010000Z'
HERE = Path(__file__).resolve().parent
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T073258Z')
GAIT = Path('/home/sensen/Desktop/engine修复验收/s5-drive-chain-diagnosis/20261105T054144Z/baseline/EVAL/RUN/GAIT.npz')
sys.path[:0] = [str(BASE), str(ROOT), str(ROOT / 'tools')]
import mujoco as mj
from plant_probe import Plant
from incremental_actuator import IncrementalActuator, DEFAULT_MODEL_PATH
import dynamic_support_metrics as dsm

DT = .0000125
CDT = .001
PDT = .005
CAP = 30.
FREQ = 3.
RAMP = .5
HANDOVER = 1.
K = 40
LAGS = np.array([0, 8, 24, 32, 36, 38, 39])  # chronological indices in 40-frame history


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


class Reference:
    def __init__(self, p):
        with np.load(GAIT) as z:
            assert np.array_equal(z['qpos_addresses'], p.qa)
            self.spline = CubicSpline(z['phase'], z['qpos_hinges'], axis=0, bc_type='periodic')
        self.lo = p.m.jnt_range[p.jids, 0]
        self.hi = p.m.jnt_range[p.jids, 1]
        self.q0 = p.q0[p.qa].copy()
        grid = self.at(np.linspace(0, 2*np.pi, 4096, endpoint=False))
        self.center = (grid.max(0) + grid.min(0)) / 2
        self.scale = np.maximum((grid.max(0)-grid.min(0))/2, .025)

    def at(self, phase):
        return np.clip(self.spline(np.asarray(phase) % (2*np.pi)), self.lo, self.hi)

    def warm(self, t):
        if t < RAMP:
            u = t / RAMP
            phase = 2*np.pi*FREQ*RAMP*(u**3-.5*u**4)
            b = 3*u*u-2*u**3
            return self.q0 + b*(self.at(phase)-self.q0)
        phase = 2*np.pi*FREQ*(t-RAMP/2)
        return self.at(phase)


def prereg():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'PREREGISTRATION.json'
    if path.exists():
        return
    source = [Path(__file__), BASE/'incremental_actuator.py', BASE/'plant_probe.py', GAIT, DEFAULT_MODEL_PATH]
    dump(path, dict(importance='核心必需', stage='S5 stable neural locomotion',
        controller='replacement learned ANN; original connectome is not run in this arm',
        architecture='MLP 294->128 tanh->128 tanh->42; causal 40-frame own-target history, 7 sampled lags; next-target prediction converted to incremental rate',
        train='real-motion-derived joint reference offline; perturbed histories and multistep self-rollout loss',
        deployment='model weights and causal target history only; no clock, phase, gait lookup, heading feedback or teacher after t=1s',
        physics=dict(nq=49,nv=48,nu=42,nmesh=71,neq=0,gravity=[0,0,-9801],dt_s=DT,
                     command_s=CDT,policy_s=PDT,rate_cap=CAP,gain_scale=2.,forces='no external forces, no root writes'),
        teacher=dict(handover_s=HANDOVER,ramp_s=RAMP,frequency_hz=FREQ,source=str(GAIT)),
        assessment=dict(duration='first 4s screen, then at least 11s total =10s ANN release, multiple initial histories and hold/motor cut',
                        support='unchanged D1-D8/D4b, whole release; no trimming bad frames',
                        locomotion='per-cycle stance/swing and material contact point slip, cycle-averaged body travel, height/tilt; circling separate from straight walking',
                        limitation='brain necessity and biological validity NOT_ESTABLISHED; teacher cannot count as neural success'),
        source_hashes={str(p):sha(p) for p in source},
        ledger=[dict(old=str(BASE/'ann_translator.py'),old_sha256=sha(BASE/'ann_translator.py'),
                     new=str(Path(__file__)),new_sha256=sha(__file__),
                     reason='replace one-step memoryless fit with causal-history neural dynamics learned under unrolled predictions; keep actuator and body unchanged')],
        units='native; absolute SI force NOT_ESTABLISHED'))


def train_model(steps=1000, name='MODEL'):
    import torch
    import torch.nn as nn
    torch.set_num_threads(4)
    torch.manual_seed(731)
    rng = np.random.default_rng(731)
    p = Plant()
    ref = Reference(p)
    center = torch.tensor(ref.center, dtype=torch.float32)
    scale = torch.tensor(ref.scale, dtype=torch.float32)
    lo = torch.tensor((ref.lo-ref.center)/ref.scale, dtype=torch.float32)
    hi = torch.tensor((ref.hi-ref.center)/ref.scale, dtype=torch.float32)
    stepcap = torch.tensor(PDT*CAP/ref.scale, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(len(LAGS)*42, 128), nn.Tanh(), nn.Linear(128,128), nn.Tanh(), nn.Linear(128,42))
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    log = []
    start = time.monotonic()
    for it in range(steps):
        # First learn local dynamics, then learn their own closed-loop trajectories.
        horizon = 1 if it < 400 else 16 if it < 650 else 64
        batch = 128 if horizon == 1 else 32
        phase = rng.uniform(0, 2*np.pi, (batch,1))
        offsets = np.arange(-K+1, horizon+1)[None, :] * (2*np.pi*FREQ*PDT)
        truth = (ref.at(phase+offsets)-ref.center)/ref.scale
        hist = torch.tensor(truth[:, :K], dtype=torch.float32)
        if it % 3 != 0:
            # correlated offsets + uncorrelated observation perturbations
            noise = rng.normal(0, .015, (batch,K,42)) + rng.normal(0,.025,(batch,1,42))
            hist += torch.tensor(noise, dtype=torch.float32)
        target = torch.tensor(truth[:, K:], dtype=torch.float32)
        loss = torch.zeros(())
        for h in range(horizon):
            pred = net(hist[:, LAGS].reshape(batch,-1))
            # Same joint/rate limits as physical inference.
            delta = torch.clamp(pred-hist[:,-1], -stepcap, stepcap)
            nxt = torch.clamp(hist[:,-1]+delta, lo, hi)
            loss = loss + ((nxt-target[:,h])**2).mean()/horizon
            hist = torch.cat((hist[:,1:],nxt[:,None]),dim=1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        if it in [400,650]:
            for g in opt.param_groups:
                g['lr'] = .0003 if it == 400 else .0001
        if it % 50 == 0 or it == steps-1:
            item = dict(step=it,horizon=horizon,loss=float(loss.detach()),wall_s=round(time.monotonic()-start,1))
            log.append(item)
            print(json.dumps(item),flush=True)
    arrays = dict(center=ref.center.astype(np.float32),scale=ref.scale.astype(np.float32),
                  lo=ref.lo,hi=ref.hi,lags=LAGS,K=np.array(K),freq=np.array(FREQ))
    for i, layer in enumerate([net[0],net[2],net[4]]):
        arrays['w'+str(i)] = layer.weight.detach().numpy().copy()
        arrays['b'+str(i)] = layer.bias.detach().numpy().copy()
    path=OUT/(name+'.npz')
    np.savez(path,**arrays)
    model=Policy(path)
    hist_np=hist.detach().numpy()
    expected=net(torch.tensor(hist_np[:,LAGS].reshape(len(hist_np),-1))).detach().numpy()
    actual=model.predict_normalized(hist_np)
    parity=float(np.abs(expected-actual).max())
    assert parity<1e-5, parity
    dump(OUT/(name+'_TRAINING.json'),dict(history=log,parity_max=parity,hash=sha(path),
        reference=str(GAIT),neural_source='replacement ANN, no connectome claim'))
    return path


class Policy:
    def __init__(self,path):
        with np.load(path) as z:
            self.w = {k:z[k].copy() for k in z.files}
        self.buf=None
        self.K=int(self.w['K'])
        self.lags=self.w['lags']

    def predict_normalized(self,hist):
        w=self.w
        x=hist[:,self.lags].reshape(len(hist),-1)
        x=np.tanh(x@w['w0'].T+w['b0'])
        x=np.tanh(x@w['w1'].T+w['b1'])
        return x@w['w2'].T+w['b2']

    def observe(self,target):
        x=(np.asarray(target,dtype=np.float32)-self.w['center'])/self.w['scale']
        if self.buf is None:
            self.buf=np.tile(x,(self.K,1))
        else:
            self.buf[:-1]=self.buf[1:]
            self.buf[-1]=x

    def action(self,target):
        pred=self.predict_normalized(self.buf[None])[0]
        qnext=self.w['center']+self.w['scale']*pred
        return np.clip((qnext-target)/PDT,-CAP,CAP)


def offline(path):
    p=Plant(); ref=Reference(p); results=[]
    for phi in [0.,1.,2.,3.,4.,5.]:
        model=Policy(path)
        hist=ref.at(phi+np.arange(-K+1,1)*2*np.pi*FREQ*PDT)
        for q in hist:
            model.observe(q)
        q=hist[-1].copy(); states=[]
        for i in range(4000):
            rate=model.action(q)
            q=np.clip(q+PDT*rate,ref.lo,ref.hi)
            states.append(q.copy());model.observe(q)
        states=np.array(states)
        expected=ref.at(phi+np.arange(1,4001)*2*np.pi*FREQ*PDT)
        grid=ref.at(np.linspace(0,2*np.pi,720,endpoint=False))
        errors=[];phases=[]
        for q in states[::10]:
            ds=np.mean(((grid-q)/ref.scale)**2,axis=1)
            j=int(ds.argmin()); errors.append(float(np.sqrt(ds[j])));phases.append(j/720*2*np.pi)
        phase=np.unwrap(phases)
        results.append(dict(initial_phase=phi,final_amplitude_ratio=float(np.std(states[-400:],axis=0).mean()/np.std(expected[-400:],axis=0).mean()),
            normalized_loop_error_mean=float(np.mean(errors)),loop_error_max=float(np.max(errors)),
            cycles=float((phase[-1]-phase[0])/(2*np.pi))))
    dump(OUT/(Path(path).stem+'_OFFLINE.json'),results)
    print(json.dumps(dict(offline=results)),flush=True)


def simulate(path,tag,seconds=4.,mode='ann',disturb=False):
    folder=OUT/tag; folder.mkdir(parents=True,exist_ok=False)
    p=Plant();m=p.m;d=p.fresh();ref=Reference(p)
    a=IncrementalActuator(m,d,kp=p.limits/.2*2.,kd=p.limits*.005*np.sqrt(2.),max_rate_rad_s=CAP)
    model=Policy(path)
    nt=int(round(seconds/CDT))
    # Fixed arrays avoid Python-object overhead in long native-contact records.
    native=np.empty((nt*80,20),np.float64)
    commands=np.empty((nt,1+42*3),np.float64)
    actions=[];observations=[];body=[];rate=np.zeros(42)
    start=time.monotonic(); warm_before=None; warm_after=None
    for k in range(nt):
        t=float(d.time)
        if k%5==0:
            model.observe(a.target_rad)   # EVERY tick, including warmup: no cold-history takeover
            if k==1000:
                warm_before=p.snapshot(d)
                np.savez_compressed(folder/'HANDOVER.npz',history=model.buf,target=a.target_rad,
                                    physics_integration=warm_before)
            if k<1000 or mode=='teacher':
                rate=np.clip((ref.warm(t)-a.target_rad)/PDT,-CAP,CAP)
            elif mode in ['hold','motor_cut']:
                rate=np.zeros(42)
            else:
                rate=model.action(a.target_rad)
            if k==1000:
                warm_after=p.snapshot(d)
                assert np.array_equal(warm_before,warm_after)
            observations.append(a.target_rad.copy()); actions.append(rate.copy())
        perturb=np.zeros(42)
        if disturb and 1300<=k<1350:
            perturb=.3*np.sin(np.arange(42)*1.731)
        commands[k]=np.r_[t,a.target_rad,rate,perturb]
        a.update(rate+perturb,CDT)
        for sub in range(80):
            d.ctrl[:]=0 if mode=='motor_cut' and k>=1000 else a.torque(d)
            assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
            mj.mj_step(m,d);mj.mj_forward(m,d)
            native[k*80+sub]=p.contact(d)
        if k%5==4:
            body.append(p.bodyrow(d))
        if (k+1)%1000==0:
            print(json.dumps(dict(tag=tag,sim_s=(k+1)/1000,wall_s=round(time.monotonic()-start,1))),flush=True)
        assert np.isfinite(d.qpos).all() and not d.warning.number.any()
    bd={key:np.array([x[key] for x in body]) for key in body[0]}
    np.savez_compressed(folder/'BODY.npz',**bd)
    np.savez_compressed(folder/'NATIVE.npz',rows=native)
    np.savez_compressed(folder/'COMMANDS.npz',rows=commands,kp=a.kp,kd=a.kd,limits=a.torque_limit,q0=p.q0[p.qa])
    np.savez_compressed(folder/'POLICY.npz',targets=observations,actions=actions)
    assist=native[(native[:,0]>=.8)&(native[:,0]<1)]
    release=native[native[:,0]>=1]
    gate=dsm.evaluate_native(dict(foot_support_mean=float(assist[:,1:7].sum(1).mean()/dsm.WEIGHT),
                                   com_z_mean=float(assist[:,10].mean())),release)
    gate['checks']['D1_no_nonfoot_ground_force']=bool(release[:,7].max()==0)
    gate['passed']=all(gate['checks'].values())
    result=dict(tag=tag,controller=mode,seconds=seconds,release_seconds=seconds-HANDOVER,
                validation=gate,finite=True,warnings=int(d.warning.number.sum()),
                model=str(path),model_hash=sha(path),body_hash=sha(folder/'BODY.npz'),
                warm_history_preserved=True,physics_handover_bitwise=True,
                original_brain='not run; replacement ANN arm',disturb=disturb,
                wall_s=time.monotonic()-start,folder=str(folder))
    dump(folder/'RESULT.json',result)
    print(json.dumps(dict(tag=tag,passed=gate['passed'],checks=gate['checks'],release=gate['release'])),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['train','offline','sim'])
    ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--name',default='MODEL')
    ap.add_argument('--tag',default='ann_screen');ap.add_argument('--seconds',type=float,default=4.)
    ap.add_argument('--control',default='ann',choices=['ann','teacher','hold','motor_cut'])
    ap.add_argument('--disturb',action='store_true');args=ap.parse_args();prereg()
    path=OUT/(args.name+'.npz')
    if args.mode=='train':
        path=train_model(args.steps,args.name);offline(path)
    elif args.mode=='offline':offline(path)
    else:simulate(path,args.tag,args.seconds,args.control,args.disturb)


if __name__=='__main__':main()
