"""Same-command ANN/teacher physical execution and controlled command swaps."""
from scope_diag import *
from live import measure


def command_for(seed,name):
    if name.startswith('swap_'):
        folder=HERE/'NECK'/f'seed{seed}';meta=read(folder/'RESULT.json')
        assert all(meta['canonical_command_exact'].values());assert sha(folder/'TRACE.npz')==meta['trace_sha256']
        with np.load(folder/'TRACE.npz',allow_pickle=False) as a:command=a['command'][:,meta['names'].index(name)].copy()
        return command,[folder/'TRACE.npz',folder/'RESULT.json']
    folder=HERE/'COMMANDS'/f'seed{seed}';meta=read(folder/'RESULT.json')
    assert sha(folder/'BANK.npz')==meta['command_sha256']
    with np.load(folder/'BANK.npz',allow_pickle=False) as a:command=a[name].copy()
    return command,[folder/'BANK.npz',folder/'RESULT.json']


@torch.inference_mode()
def run(seed,name,mode,out):
    torch.set_num_threads(1);start=time.monotonic();commands,inputs=command_for(seed,name)
    b=Body(height=0,adhesion=True,seed=seed);b.restore(seed);setup(b,'terminal')
    actor=load_actor(BODY_WEIGHT) if mode=='ann' else None
    cpg=copy.deepcopy(b.motor.cpg_network);freq=cpg.intrinsic_freqs.copy()
    out.mkdir(parents=True);mujoco.mj_saveModel(b.m,str(out/'BODY.mjb'),None)
    assert sha(out/'BODY.mjb')==sha(PREVIOUS/'FORMAL'/f'seed{seed}'/'native_full.mjb')
    pins=sources([BODY_WEIGHT,*inputs]);write(out/'INPUTS.json',dict(seed=seed,name=name,mode=mode,
       source_sha256=pins,body_sha256=sha(out/'BODY.mjb'),control_source='offline diagnostic replay of actual post-neck commands or declared swaps',
       brain_disconnected=True,weights_modified=False))
    rows={k:[] for k in ('qpos','qvel','position','yaw','upright','nonfoot','control','pads','normal_load','command')}
    initial=b.d.xpos[b.thorax].copy();warning=False;teacher_calls=0
    for tick,command in enumerate(commands):
        if mode=='teacher':
            action=b.motor.step(command,observation(b));teacher_calls+=1
            if tick%10==0:control=np.clip(action.joint_angles,b.minimum,b.maximum);pads=action.adhesion_onoff.copy()
        else:
            advance(cpg,freq,command);phase=phase_state(cpg)
            if tick%10==0:
                obs=observation(b)
                x=np.r_[feature(b.d.qpos.copy(),b.d.qvel.copy(),obs.stumbling_contact_forces,command,phase,b.weight),extra_live(b)].astype(np.float32)
                control,pads,_=actor.act(x,phase)
        b.apply(control,pads);load,_,nf,_=b.contact();pos=b.d.xpos[b.thorax];rot=b.d.xmat[b.thorax].reshape(3,3)
        for key,value in dict(qpos=b.d.qpos,qvel=b.d.qvel,position=pos,yaw=np.arctan2(rot[1,0],rot[0,0]),
             upright=rot[2,2],nonfoot=nf,control=control,pads=pads,normal_load=load,command=command).items():rows[key].append(np.array(value,copy=True))
        if b.d.warning.number.any() or not np.isfinite(b.d.qpos).all():warning=True;break
        if time.monotonic()-start>120:warning=True;break
    arrays={k:np.asarray(v) for k,v in rows.items()};arrays['initial_position']=initial
    np.savez_compressed(out/'TRACE.npz',**arrays)
    metrics=measure(arrays,b);matching=None
    baseline={'native':'native_full','reference':'reference_boundary','translated':'translated'}
    if mode=='ann' and name in baseline:
        old=load_trace(seed,baseline[name]);matching={k:bool(np.array_equal(v,old[k])) for k,v in arrays.items()}
    verify(pins)
    result=dict(seed=seed,name=name,mode=mode,physical_steps=len(commands) if not warning else len(arrays['qpos']),
          warning_or_timeout=warning,metrics=metrics,live_trace_equal=matching,teacher_calls=teacher_calls,
          source_sha256=pins,trace_sha256=sha(out/'TRACE.npz'),wall_seconds=time.monotonic()-start,
          optimizer_updates=0,neural_steps=0)
    write(out/'RESULT.json',result)
    print(seed,name,mode,'yaw',round(metrics['yaw_degrees'],6),'safety',metrics['safety_pass'],
          'exact',all(matching.values()) if matching is not None else None,round(time.monotonic()-start,2),flush=True)
    if warning:raise RuntimeError('physical warning/timeout; partial record retained')
    if matching is not None and not all(matching.values()):raise ValueError('same command replay mismatch')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('name');p.add_argument('mode',choices=('ann','teacher'));p.add_argument('output',type=Path)
    a=p.parse_args();run(a.seed,a.name,a.mode,a.output)
