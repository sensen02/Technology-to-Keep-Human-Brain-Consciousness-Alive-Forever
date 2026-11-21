"""Teacher sampling, pure ANN evaluation and same-state DAgger labels."""
import argparse
import copy
import time
from common import *
from policy_plus import Policy,read
from proprioception import extra_live


@torch.inference_mode()
def run(height,seed,mode,out,checkpoint=None,beta=0.):
    started=time.monotonic();torch.set_num_threads(1)
    b=Body(height=height,adhesion=True,seed=seed);b.restore(seed);setup(b,'terminal')
    commands=command_trace();actor=read(checkpoint) if checkpoint else None
    cpg=copy.deepcopy(b.motor.cpg_network);freq=cpg.intrinsic_freqs.copy()
    out.mkdir(parents=True,exist_ok=False)
    identity=manifest([checkpoint] if checkpoint else [])
    write(out/'INPUTS.json',dict(source_sha256=identity,mode=mode,beta=beta,height=height,seed=seed))
    mujoco.mj_saveModel(b.m,str(out/'BODY.mjb'),None)
    initial=b.d.xpos[b.thorax].copy();rows={k:[] for k in ('qpos','qvel','position','upright','nonfoot','block_force','control','pads','force')}
    examples={k:[] for k in ('x','phase','y','raw_y','lambda','pad_y','applied')};teacher_calls=0
    for tick,command in enumerate(commands):
        obs=observation(b)
        if mode!='student':
            action=b.motor.step(command,obs);teacher_calls+=1
            label=np.clip(action.joint_angles,b.minimum,b.maximum);drive=b.motor.last_info['net_corrections'].copy()
        advance(cpg,freq,command);phase=phase_state(cpg)
        if mode!='student':assert np.max(np.abs(cpg.curr_phases-b.motor.cpg_network.curr_phases))<1e-11
        if tick%HOLD==0:
            x=feature(b.d.qpos.copy(),b.d.qvel.copy(),obs.stumbling_contact_forces,command,phase,b.weight)
            x=np.r_[x,extra_live(b)].astype(np.float32)
            assert x.shape==(254,)
            if actor:estimate,pad_estimate,raw=actor.act(x,phase)
            if mode=='teacher':control=label.copy();pads=action.adhesion_onoff.copy()
            elif mode in ('student','shadow'):control=estimate;pads=pad_estimate
            elif mode=='oracle':
                control=np.clip(actor.rhythmic(phase[None])[0]+actor.synergy.numpy()@drive,b.minimum,b.maximum)
                pads=pad_estimate
            else:
                control=(1-beta)*estimate+beta*label
                # Both pad controllers are phase-only. Keep their equality visible.
                pads=pad_estimate
            examples['x'].append(x);examples['phase'].append(phase.copy());examples['applied'].append(control.copy())
            if mode!='student':
                examples['y'].append(label.copy());examples['raw_y'].append(action.joint_angles.copy())
                examples['lambda'].append(drive);examples['pad_y'].append(action.adhesion_onoff.copy())
        b.apply(control,pads)
        load,force,nf,bf=b.contact()
        vals=dict(qpos=b.d.qpos.copy(),qvel=b.d.qvel.copy(),position=b.d.xpos[b.thorax].copy(),upright=b.d.xmat[b.thorax,8],
                  nonfoot=nf,block_force=bf,control=control.copy(),pads=pads.copy(),force=b.d.actuator_force[b.actuators].copy())
        for key in rows:rows[key].append(vals[key])
        if b.d.warning.number.any() or not np.isfinite(b.d.qpos).all():break
    arrays={k:np.asarray(v) for k,v in rows.items()};arrays.update({k:np.asarray(v) for k,v in examples.items()})
    arrays['initial_position']=initial
    np.savez_compressed(out/'TRACE.npz',**arrays)
    result=assess({k:arrays[k] for k in list(rows)+['initial_position']},b,height)
    result.update(mode=mode,height=height,seed=seed,beta=beta,steps=len(arrays['qpos']),frames=len(arrays['x']),
       teacher_calls=teacher_calls,weight_sha256=sha(checkpoint) if checkpoint else None,trace_sha256=sha(out/'TRACE.npz'),
       wall_seconds=time.monotonic()-started)
    assert all(sha(p)==h for p,h in json.loads((out/'INPUTS.json').read_text())['source_sha256'].items())
    write(out/'RESULT.json',result);print(height,seed,mode,result['pass_'],round(result['dx'],3),
                  [k for k,v in result['gates'].items() if not v],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('teacher','student','dagger','oracle','shadow'))
    p.add_argument('height',type=float);p.add_argument('seed',type=int);p.add_argument('out',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--beta',type=float,default=0.);a=p.parse_args()
    run(a.height,a.seed,a.mode,a.out,a.checkpoint,a.beta)
