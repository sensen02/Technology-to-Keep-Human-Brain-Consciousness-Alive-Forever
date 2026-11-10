"""Read-only model probes. No training, parameter edits, injury, or rendering."""
from pathlib import Path
import importlib.util, sys, json, hashlib, os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
HERE = Path(__file__).resolve().parent
TRAIN = ROOT/'experiments/banc-adh-namebug/20261110T150000Z/train_adhfix.py'
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
tr=load('audit_train',TRAIN)
tr.PAD_XML=ROOT/'experiments/banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml'
act=load('audit_act',tr.ACT); sens=load('audit_sens',tr.SENSORY)
base=np.load(tr.BASE); checkpoint=np.load(TRAIN.parent/'MODEL_adhfix.npz')
sim=tr.Sim(base,act,sens)
net=tr.Net(np.random.default_rng(33000))
for key in net.p:
    ck='actor_'+key if key in ('w0','b0','w1','b1','w2','b2') else {'wc':'critic_w','bc':'critic_b','logstd':'logstd'}[key]
    net.p[key]=checkpoint[ck].astype(float)

def rollout(mode, ticks=30):
    sim.reset()
    if mode=='video': sim.buf=np.repeat(tr.POSE_Q[None],sim.K,axis=0)
    rows=[]; targets=[]; commands=[]; limited=[]; q=[]; obsrows=[]; base_limited=[]
    for t in range(ticks):
        if mode=='evaluation' or t%tr.BASE_EVERY==0:
            target=sim.base_target()
            if mode!='evaluation': sim.buf=np.vstack([sim.buf[1:],target[None]])
        loads=sim.loads_and_speed()[0]
        obs=np.concatenate([sens.features(sim.shim,sim.d,target,loads),sim.buf[sim.lags].reshape(-1)])
        _,_,mean,_=net.forward(obs[None])
        base_rate=(target-sim.actuator.target_rad)/tr.PDT
        base_limited.append(np.clip(base_rate,-2.,2.))
        rate=base_rate+np.tanh(mean[0])*tr.C_END/tr.PDT
        sim.step(rate)
        if mode=='evaluation': sim.buf=np.vstack([sim.buf[1:],target[None]])
        targets.append(target.copy()); commands.append(rate.copy());limited.append(sim.actuator.last_limited_rate_rad_s.copy());q.append(sim.d.qpos.copy());obsrows.append(obs)
    return dict(target=np.array(targets),command=np.array(commands),limited=np.array(limited),q=np.array(q),obs=np.array(obsrows),base_limited=np.array(base_limited))
runs={m:rollout(m) for m in ('training_deterministic','evaluation','video')}
level=json.loads((TRAIN.parent/'LEVEL_0_v0.05/RESULT.json').read_text())
updates=[x for x in level['updates'] if 'decisions' in x]
actual_decisions=sum(x['decisions'] for x in updates)
log=json.loads((TRAIN.parent/'TRAIN_LOG_adhfix.json').read_text())
observed={
 'scope':'30 ticks each, deterministic final checkpoint; training schedule uses final C, no stochastic exploration. 7200 native steps total; no retraining.',
 'actor_shapes':{k:list(v.shape) for k,v in net.p.items()},
 'actor_parameters':sum(net.p[k].size for k in ('w0','b0','w1','b1','w2','b2')),
 'all_optimizer_parameters':sum(v.size for v in net.p.values()),
 'base_shapes':{k:list(base[k].shape) for k in base.files},
 'K':sim.K,'lags':sim.lags.tolist(),
 'actual_max_rate_range': [float(sim.actuator.max_rate_rad_s.min()),float(sim.actuator.max_rate_rad_s.max())],
 'actual_kp_range':[float(sim.actuator.kp.min()),float(sim.actuator.kp.max())],
 'actual_kd_range':[float(sim.actuator.kd.min()),float(sim.actuator.kd.max())],
 'actual_torque_limit_range':[float(sim.actuator.torque_limit.min()),float(sim.actuator.torque_limit.max())],
 'gamma_time_constant_s':float(-tr.PDT/np.log(tr.CFG['gamma'])),
 'gamma_gae_time_constant_s':float(-tr.PDT/np.log(tr.CFG['gamma']*tr.CFG['gae'])),
 'training_updates_recorded':len(updates),'training_decisions_recorded':actual_decisions,
 'training_native_steps_reconstructed':actual_decisions*80,
 'logged_total_physics_steps':log['total_physics_steps'],
 'actual_control_legs':tr.LEGS,'sensory_declared_legs':list(sens.LEGS),
 'pad_bottom_z_at_last_state':sim.pad_bottom_z().tolist(),
 'run_diagnostics':{},'schedule_differences':{},
 'source_sha256':{}
}
for mode,r in runs.items():
    observed['run_diagnostics'][mode]={'requested_over_2_count':int(np.sum(np.abs(r['command'])>2)), 'channel_tick_count':int(r['command'].size), 'effective_rate_unchanged_if_actor_removed_at_same_states_count':int(np.sum(np.isclose(r['limited'],r['base_limited'],rtol=0,atol=1e-12))), 'limited_abs_max':float(np.abs(r['limited']).max()),'obs_shape':list(r['obs'].shape)}
for mode in ('evaluation','video'):
    observed['schedule_differences'][mode+'_vs_training']={key+'_max_abs':float(np.abs(runs[mode][key]-runs['training_deterministic'][key]).max()) for key in ('target','obs','q')}
for p in (TRAIN,tr.ACT,tr.SENSORY,tr.BASE,tr.PAD_XML,TRAIN.parent/'MODEL_adhfix.npz',ROOT/'experiments/banc-walk-video/20261110T130000Z/render_walk.py'):
    observed['source_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(HERE/'PROBE_RESULTS.json').write_text(json.dumps(observed,indent=2)+'\n')
print(json.dumps(observed,indent=2))
