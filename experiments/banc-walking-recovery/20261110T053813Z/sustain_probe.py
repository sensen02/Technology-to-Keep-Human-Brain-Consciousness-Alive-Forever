"""Frozen bounded six-second deterministic sustain diagnostic; no training or source mutation."""
from pathlib import Path
import json, os, time
import numpy as np
import torch
from models import ActorCritic
from physics import Core, LEGS, W, summarise
from policy_io import actor_observation, critic_observation
from identity import capture, verify_runtime, atomic_json, sha256

HERE=Path(__file__).resolve().parent
OUT=HERE/'sustain_diag'
CP=HERE/'walk_zero_ff238_3u/final.pt'


def events_of(a):
    """Require 3 consecutive ms at load transitions; no events without prior loaded stance."""
    result={}
    for leg,name in enumerate(LEGS):
        load=a['loads'][:,leg]/W; z=a['padz'][:,leg]; x=a['relfoot'][:,leg,0]
        loaded=(load>=.1)&(z<=.004); unloaded=load<=.05
        armed=False; active=None; lc=uc=0; found=[]; attempts=[]
        for i in range(len(load)):
            lc=lc+1 if loaded[i] else 0
            uc=uc+1 if unloaded[i] else 0
            if active is None:
                if lc>=3: armed=True
                if armed and uc>=3:
                    s=i-2; active={'start':s,'baseline_z':float(z[max(0,s-1)]),'baseline_x':float(x[s])}
                    armed=False
            else:
                if lc>=3:
                    s=active['start']; end=i-2
                    ids=np.arange(s,end+1); ids=ids[unloaded[ids]]
                    rise=float(np.max(z[ids]-active['baseline_z'])) if len(ids) else 0.
                    advance=float(np.max(x[ids]-active['baseline_x'])) if len(ids) else 0.
                    ev=dict(unload_s=float(a['time'][s]),reload_s=float(a['time'][end]),
                            confirmed_s=float(a['time'][i]),height_rise=rise,
                            max_bottom_height=float(np.max(z[ids])) if len(ids) else 0.,
                            forward_while_unloaded=advance,reload_load_W=float(load[end]),
                            reload_bottom=float(z[end]),complete=bool(rise>.015 and advance>=.03))
                    attempts.append(ev)
                    if ev['complete']: found.append(ev)
                    active=None;armed=True
        result[name]=dict(complete_count=len(found),complete_events=found,all_reload_attempts=attempts,
                         unfinished_unload_s=None if active is None else float(a['time'][active['start']]))
    return result


def snapshot(c,m,action):
    return dict(time=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),
                com=c.d.subtree_com[c.th].copy(),loads=m['loads'].copy(),nonfoot=float(m['nonfoot']),
                speed=m['speed'].copy(),padz=m['padz'].copy(),relfoot=m['foot'].copy(),
                action=action.copy(),target=c.actuator.target_rad.copy(),
                torque_command_last_substep=c.d.ctrl.copy(),torque_actual=c.d.actuator_force.copy(),
                target_angle_clip=c.actuator.angle_clip_mask.copy())


def main():
    start=time.monotonic(); torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if OUT.exists(): raise FileExistsError('Refusing to overwrite diagnostic outputs')
    OUT.mkdir()
    checkpoint_hash=sha256(CP);script_hash=sha256(__file__)
    payload=torch.load(CP,map_location='cpu',weights_only=False)
    c=Core(adhesion=False)
    verify_runtime(payload['identity'],c)
    mid=payload['model_identity']; saved=payload['identity']['config']
    assert mid['architecture']=='ff238' and mid['critic_obs_dim']==255 and mid['dtype']=='torch.float64'
    assert saved['target_speed']==.10 and saved['ticks']==2000 and saved['decision_ms']==5
    assert saved['adhesion_enabled'] is False and payload['identity']['adhesion']['enabled'] is False
    model=ActorCritic('ff238',critic_obs_dim=255,dtype=torch.float64,logstd_bounds=tuple(mid['logstd_bounds']))
    assert model.identity()==mid
    model.load_state_dict(payload['state']['model'],strict=True);model.eval()
    config=dict(checkpoint=str(CP),checkpoint_sha256=checkpoint_hash,model_identity=mid,
                source_identity=payload['identity']['identity_sha256'],script_sha256=script_hash,
                total_s=6.,ticks=6000,decision_ms=5,target_speed=.10,adhesion_enabled=False,
                critic_remaining='max(0,2-time_s), raw seconds exactly as original trainer (not /6)',
                reset='Core initial reset only; no reset at 2s or 4s',
                sampling='1ms endpoints plus initial state; NOT native recording',
                event_criteria='prior loaded>=.1W and bottom<=.004; unload<=.05W; height rise>.015; forward>=.03 during unloaded; reload>=.1W and bottom<=.004; 3ms transition confirmation',
                scope='bounded derived-world diagnostic windows, NOT final acceptance',
                threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    atomic_json(OUT/'CONFIG.json',config)
    manifest=capture(c,config,extra_paths=[CP,Path(__file__)])
    atomic_json(OUT/'IDENTITY_BEFORE.json',manifest)
    previous=np.zeros(42); action=np.zeros(42)
    rows=[snapshot(c,c.measure(),action)]
    with torch.inference_mode():
        for k in range(0,6000,5):
            obs=actor_observation(c,model,previous,.10)
            cv=critic_observation(c,model,obs,max(0.,2.-k*.001))
            action=model.act(obs,cv,deterministic=True)['action'][0].numpy().copy()
            for _ in range(5):
                m=c.tick(action); rows.append(snapshot(c,m,action))
            previous=action.copy()
            if (k+5)%2000==0: print('completed_ms',k+5,flush=True)
    a={key:np.asarray([r[key] for r in rows]) for key in rows[0]}
    assert len(a['time'])==6001 and np.allclose(a['time'],np.arange(6001)*.001,rtol=0,atol=2e-9)
    assert all(np.isfinite(v).all() for v in a.values())
    verify_runtime(payload['identity'],c);verify_runtime(manifest,c)
    assert sha256(__file__)==script_hash and sha256(CP)==checkpoint_hash
    atomic_json(OUT/'IDENTITY_AFTER.json',dict(source_verified=True,diagnostic_verified=True,
                checkpoint_sha256=sha256(CP),script_sha256=sha256(__file__),physical_time_s=float(c.d.time),
                source_identity=payload['identity']['identity_sha256'],diagnostic_identity=manifest['identity_sha256']))
    np.savez_compressed(OUT/'TRACE_1MS.npz',**a,leg_order=np.asarray(LEGS))
    windows={}
    for lo,hi in ((0,2),(2,4),(4,6)):
        # Historical metric: endpoint samples (lo,hi], excluding initial state.
        ids=np.arange(lo*1000+1,hi*1000+1)
        rec=[(a['time'][i],a['com'][i],a['loads'][i],a['nonfoot'][i],a['speed'][i]) for i in ids]
        metric=summarise(rec,len(ids)*80,initial_heading=c.initial_heading)
        metric.update(scope='historical engineering metric on 1ms endpoints; NOT final acceptance',
                      target_clip_fraction=float(a['target_angle_clip'][ids].mean()),
                      target_clip_per_joint=a['target_angle_clip'][ids].mean(0).tolist(),
                      target_net_shift_rad=(a['target'][hi*1000]-a['target'][lo*1000]).tolist(),
                      actual_joint_net_shift_rad=(a['qpos'][hi*1000,c.a_qpos]-a['qpos'][lo*1000,c.a_qpos]).tolist())
        windows[f'{lo}-{hi}s']=metric
    delta=np.diff(a['com'][:,0]); vx=delta/.001
    blocks=np.diff(a['com'][::100,0])/.1
    ev=events_of(a)
    result=dict(config=config,windows=windows,events=ev,
                continuity=dict(world_x_at_0_2_4_6=a['com'][::2000,0].tolist(),
                                signed_100ms_speeds=blocks.tolist(),
                                positive_100ms_fraction=float(np.mean(blocks>0)),
                                negative_100ms_fraction=float(np.mean(blocks<0)),
                                instantaneous_vx_min=float(vx.min()),instantaneous_vx_max=float(vx.max())),
                drift_diagnostics=dict(target_net_shift_rad=(a['target'][-1]-a['target'][0]).tolist(),
                    target_total_variation_rad=np.abs(np.diff(a['target'],axis=0)).sum(0).tolist(),
                    target_monotonicity=np.abs(a['target'][-1]-a['target'][0]).__truediv__(np.maximum(np.abs(np.diff(a['target'],axis=0)).sum(0),1e-12)).tolist(),
                    complete_events_per_leg={leg:ev[leg]['complete_count'] for leg in LEGS},
                    inference_rule='Repeated complete unload/lift/advance/reload events across later windows support cycling; net target drift and rising clipping without repeated complete events support finite-stroke motion; neither is proof of final acceptance'),
                wall_s=time.monotonic()-start)
    atomic_json(OUT/'RESULT.json',result)
    # Read back every data field, not merely file existence.
    with np.load(OUT/'TRACE_1MS.npz',allow_pickle=False) as z:
        assert list(z['leg_order'])==list(LEGS)
        for key,value in a.items(): assert np.array_equal(z[key],value),key
    assert json.loads((OUT/'RESULT.json').read_text())['events']==ev
    atomic_json(OUT/'READBACK.json',dict(all_arrays_exact=True,results_exact=True,
                samples=6001,sampling_ms=1,script_frozen_sha256=script_hash,reset_count=1,
                source_runtime_before_after_verified=True))
    inventory={str(p):dict(sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.is_file()}
    atomic_json(OUT/'OUTPUT_HASHES.json',inventory)
    hashes=json.loads((OUT/'OUTPUT_HASHES.json').read_text())
    assert all(sha256(p)==meta['sha256'] for p,meta in hashes.items())
    print(json.dumps(dict(windows={k:{q:v[q] for q in ('forward_signed_native_per_s','slip_ratio_max','com_height_min','nonfoot_force_max_native','target_clip_fraction')} for k,v in windows.items()},complete_events=result['drift_diagnostics']['complete_events_per_leg'],wall_s=result['wall_s'])),flush=True)

if __name__=='__main__': main()
