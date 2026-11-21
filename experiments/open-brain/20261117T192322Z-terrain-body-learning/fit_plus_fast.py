"""Actual supervised neural updates, with motor readout identification and provenance."""
import argparse
import copy
import hashlib
import math
import time
from shared import *
from policy import basis,calibrate_pad
from policy_plus import Policy,migrate
from proprioception_fast import extra_saved


def load(folder,correction=False):
    packed={key:[] for key in ('x','phase','y','raw_y','lambda','pad_y')};sources=[]
    for path in sorted(Path(folder).glob('*/TRACE.npz')):
        meta=json.loads((path.parent/'RESULT.json').read_text())
        if not correction and not meta['pass_']:continue
        assert sha(path)==meta['trace_sha256']
        with np.load(path,allow_pickle=False) as a:
            n=len(a['x'])
            if correction:
                safe=(a['upright']>=.8)&(a['position'][:,2]>=.7*a['initial_position'][2])&(a['nonfoot']<=.2*10.0484811)
                native=mujoco.MjModel.from_binary_path(str(path.parent/'BODY.mjb'))
                joint=np.where(native.jnt_limited)[0];q=a['qpos'][:,native.jnt_qposadr[joint]]
                excess=np.maximum(np.maximum(native.jnt_range[joint,0]-q,q-native.jnt_range[joint,1]),0)
                safe &= excess.max(1)<=.02
                bad=np.where(~safe)[0];stop=int(bad[0]) if len(bad) else len(safe)
                n=min(n,max(0,stop//HOLD))
            if not n:continue
            for key in packed:
                value=a[key][:n].copy()
                if key=='x' and value.shape[1]==230:value=np.c_[value,extra_saved(path)[:n]]
                packed[key].append(value)
        sources.append(dict(path=str(path),sha256=sha(path),used_frames=n,episode_pass=meta['pass_']))
    assert packed['x'],'no usable complete teacher/correction examples'
    return {key:np.concatenate(value) for key,value in packed.items()},sources


def identify(data):
    b=Body();b.restore(119);model=Policy();model.neutral.copy_(torch.from_numpy(b.neutral))
    model.minimum.copy_(torch.from_numpy(b.minimum));model.maximum.copy_(torch.from_numpy(b.maximum))
    model.q_feature_columns.copy_(torch.from_numpy(b.m.jnt_qposadr[b.jids]-5))
    phases=np.arange(4096)*2*np.pi/4096;design=basis(phases);mapping={};coeff=[];errors=[]
    for i,leg in enumerate(b.legs):
        targets=b.motor.preprogrammed_steps.get_joint_angles(leg,phases).T
        neutral=b.motor.preprogrammed_steps.get_joint_angles(leg,0.,0.)
        solution=np.linalg.lstsq(design,targets-neutral,rcond=None)[0]
        coeff.append(solution);errors.append(float(np.max(np.abs(design@solution-(targets-neutral)))))
        for j,spec in enumerate(b.motor.preprogrammed_steps.dofs_per_leg):mapping[dof_spec_to_jointdof(leg,spec)]=7*i+j
    model.fourier.copy_(torch.from_numpy(np.asarray(coeff)))
    model.order.copy_(torch.tensor([mapping[d] for d in b.dofs]))
    base=model.rhythmic(data['phase']);residual=data['raw_y']-base;synergy=np.zeros((42,6))
    for i,leg in enumerate(b.legs):
        indexes=[k for k,d in enumerate(b.dofs) if d.child.pos==leg]
        drive=data['lambda'][:,i];assert np.sum(drive*drive)>1e-6,'no source events for a leg'
        synergy[indexes,i]=drive@residual[:,indexes]/np.sum(drive*drive)
    model.synergy.copy_(torch.from_numpy(synergy))
    model.lambda_scale.copy_(torch.from_numpy(np.maximum(data['lambda'].max(0),1.).astype(np.float32)))
    pads,calls=calibrate_pad(b);model.pad_weights.copy_(torch.from_numpy(pads))
    phase=data['phase'][:,:6];pf=np.stack((np.sin(phase),np.cos(phase),np.ones_like(phase)),axis=-1)
    prediction=np.sum(pf*pads,axis=-1)>=0
    reconstruction=np.clip(base+data['lambda']@synergy.T,b.minimum,b.maximum)
    error=reconstruction-data['y']
    assert max(errors)<.002 and np.max(np.abs(error))<.002
    assert np.array_equal(prediction,data['pad_y'])
    write(HERE/'READOUT_IDENTIFICATION.json',dict(template_max_errors=errors,
        joint_target_rmse=float(np.sqrt(np.mean(error*error))),max_abs_joint_error=float(np.max(np.abs(error))),
        pad_teacher_query_calls=calls,pad_train_classification_exact=True,
        synergy=synergy.tolist(),lambda_scale=model.lambda_scale.tolist(),
        statement='Fitted teacher-derived motor/pad readouts; sensory neural drive is trained separately with AdamW.'))
    return model


def objective(model,x,base,y,drive):
    pred,raw,gate=model(x,base)
    latent=((raw*gate-drive/model.lambda_scale)**2).mean()
    joint=((pred-y)/.05).square().mean()
    return latent+joint


def params_digest(model):
    h=hashlib.sha256()
    for name,value in model.named_parameters():h.update(name.encode());h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def run(out,epochs,initial=None,corrections=()):
    torch.set_num_threads(1);torch.manual_seed(29371);started=time.monotonic()
    data,sources=load(HERE/'TEACHER_TRAIN');dev,dev_sources=load(HERE/'TEACHER_DEV')
    rng=torch.Generator().manual_seed(48317)
    model=read_model=None;migration=False
    parent_hash=sha(initial) if initial else None
    if initial:
        saved=torch.load(initial,weights_only=True,map_location='cpu')
        migration=len(saved['state']['center'])==272
        if migration:model=migrate(saved['state'])
        else:
            model=Policy();model.load_state_dict(saved['state'])
    else:model=identify(data)
    original_frames=len(data['x']);extra_frames=0
    for folder in corrections:
        extra,extra_sources=load(folder,True);sources+=extra_sources;extra_frames+=len(extra['x'])
        data={key:np.concatenate((data[key],extra[key])) for key in data}
    base=model.rhythmic(data['phase']);dev_base=model.rhythmic(dev['phase'])
    if initial is None:
        z=np.c_[data['x'],data['x'][:,model.q_feature_columns.numpy()]-base]
        model.center.copy_(torch.from_numpy(z.mean(0).astype(np.float32)))
        model.scale.copy_(torch.from_numpy(np.maximum(z.std(0),.03).astype(np.float32)))
    if migration:
        model.center[230:254].copy_(torch.from_numpy(data['x'][:,230:254].mean(0).astype(np.float32)))
        model.scale[230:254].copy_(torch.from_numpy(np.maximum(data['x'][:,230:254].std(0),.03).astype(np.float32)))
    tensors=[torch.from_numpy(data[key].astype(np.float32 if key in ('x','lambda') else np.float64)) for key in ('x','y','lambda')]
    x,y,drive=tensors;base_t=torch.from_numpy(base)
    dx=torch.from_numpy(dev['x']);dy=torch.from_numpy(dev['y']);dd=torch.from_numpy(dev['lambda'].astype(np.float32));db=torch.from_numpy(dev_base)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001 if initial is None else .0005,weight_decay=1e-5)
    update=0;start_epoch=0
    if initial and not migration:
        optimizer.load_state_dict(saved['optimizer']);rng.set_state(saved['rng']);torch.set_rng_state(saved['torch_rng'])
        update=saved['update'];start_epoch=saved['epoch']
    out.mkdir(parents=True,exist_ok=False)
    identity={str(p):sha(p) for p in [Path(__file__).resolve(),HERE/'policy.py',HERE/'shared.py',HERE/'policy_plus.py',HERE/'proprioception.py',HERE/'proprioception_fast.py']}
    if initial:identity[str(initial)]=parent_hash
    identity.update({r['path']:r['sha256'] for r in sources+dev_sources})
    write(out/'INPUTS.json',dict(source_sha256=identity,train=sources,develop=dev_sources,
        original_frames=original_frames,correction_frames=extra_frames,epochs=epochs,initial=str(initial),
        body_source_sha256=sha(TEACHER/'body_model.py'),optimizer_reset_for_new_observations=migration,initial_sha256=parent_hash))
    if initial is None or migration:
        # Check the actual composite loss in float64 away from clipping corners.
        event_ids=np.where((data['lambda'].max(1)>.5)&(data['phase'][:,6:].max(1)>.1))[0][:128]
        assert len(event_ids)>=32
        probe=copy.deepcopy(model).double();args=(x[event_ids].double(),base_t[event_ids],y[event_ids],drive[event_ids].double())
        loss=objective(probe,*args);loss.backward();checks=[]
        for parameter in (probe.reflex[-1].bias,probe.reflex[-1].weight):
            index=int(parameter.grad.abs().argmax());flat=parameter.data.reshape(-1);old=float(flat[index]);eps=1e-6
            analytic=float(parameter.grad.reshape(-1)[index]);flat[index]=old+eps;plus=float(objective(probe,*args));flat[index]=old-eps;minus=float(objective(probe,*args));flat[index]=old
            assert abs(analytic)>1e-6
            numeric=(plus-minus)/(2*eps);checks.append(dict(analytic=analytic,numeric=numeric,error=abs(analytic-numeric)))
            assert abs(analytic-numeric)<=1e-7+1e-4*abs(analytic)
        write(out/'GRADIENT_CHECK.json',dict(pass_=True,checks=checks,objective='actual lambda+clipped joint loss'))
    def save(epoch):
        path=out/f'epoch{epoch:03d}.pt';temp=path.with_suffix('.tmp')
        torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.get_state(),torch_rng=torch.get_rng_state(),
                        epoch=epoch,update=update),temp);temp.replace(path);return sha(path)
    save(start_epoch)
    event=torch.from_numpy(np.where(data['lambda'].max(1)>.5)[0]);uniform=len(x)
    history=[]
    with (out/'UPDATES.jsonl').open('w') as log:
      for local_epoch in range(1,epochs+1):
        if time.monotonic()-started>600:raise TimeoutError('training budget')
        lr=(.001 if initial is None else .0005)*(.2+.8*.5*(1+math.cos(math.pi*local_epoch/epochs)))
        for group in optimizer.param_groups:group['lr']=lr
        for _ in range(math.ceil(uniform/512)):
            left=torch.randint(uniform,(256,),generator=rng)
            right=event[torch.randint(len(event),(256,),generator=rng)] if len(event) else torch.randint(uniform,(256,),generator=rng)
            if extra_frames:
                correction_ids=original_frames+torch.randint(extra_frames,(256,),generator=rng)
                ids=torch.cat((left[:128],right[:128],correction_ids))
            else:
                ids=torch.cat((left,right))
            optimizer.zero_grad(set_to_none=True);loss=objective(model,x[ids],base_t[ids],y[ids],drive[ids]);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();update+=1
            log.write(json.dumps(dict(update=update,epoch=start_epoch+local_epoch,loss=float(loss.detach()),lr=lr,
                                      parameter_sha256=params_digest(model)))+'\n')
        with torch.inference_mode():
            estimate,raw,g=model(dx,db);rmse=float(torch.sqrt(torch.mean((estimate-dy)**2)))
            devloss=float(objective(model,dx,db,dy,dd))
        epoch=start_epoch+local_epoch;entry=dict(epoch=epoch,update=update,dev_joint_rmse=rmse,dev_loss=devloss)
        if local_epoch%10==0 or local_epoch==epochs:entry['checkpoint_sha256']=save(epoch)
        history.append(entry);write(out/'HISTORY.json',history)
        if local_epoch%10==0:print('epoch',epoch,'updates',update,'dev joint RMSE',rmse,flush=True)
    assert all(sha(p)==h for p,h in identity.items())
    choices=[h for h in history if 'checkpoint_sha256' in h];choice=min(choices,key=lambda h:h['dev_joint_rmse'])
    write(out/'SELECTED.json',dict(**choice,path=str(out/f"epoch{choice['epoch']:03d}.pt"),selection='lowest declared development joint RMSE; physical qualification still required'))
    write(out/'RESULT.json',dict(actual_updates=update-(saved['update'] if initial and not migration else 0),total_update=update,epochs=epochs,
        frames=len(x),original_frames=original_frames,correction_frames=extra_frames,wall_seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--initial',type=Path);p.add_argument('--correction',type=Path,action='append',default=[])
    a=p.parse_args();run(a.out,a.epochs,a.initial,a.correction)
