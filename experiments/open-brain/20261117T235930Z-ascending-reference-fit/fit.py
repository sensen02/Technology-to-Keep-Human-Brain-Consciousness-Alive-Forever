"""Actual fixed-budget supervised training on the declared engineering reference."""
import copy
import time
from shared import *
from encoder import Encoder


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def main():
    torch.set_num_threads(1);torch.manual_seed(31057)
    b=read(HERE/'BINDING.json');verify(b['source_sha256'])
    arrays=[load_data(c)[0] for c in TRAIN_CASES]
    x=torch.from_numpy(np.concatenate([a['features_hz'] for a in arrays])).double()
    y=torch.from_numpy(np.concatenate([a['teacher_rate_hz'] for a in arrays])).double()
    assert not (HERE/'DATA/held_left').exists() and not (HERE/'DATA/held_right').exists(), 'held teacher labels must be generated after freeze'
    pins=sources({HERE/'BINDING.json',*(HERE/'DATA'/c/'PAIRED.npz' for c in TRAIN_CASES),
                  *(HERE/'DATA'/c/'RESULT.json' for c in TRAIN_CASES)})
    write(HERE/'FIT_INPUTS.json',dict(source_sha256=pins,training_cases=TRAIN_CASES,frames=len(x),
         updates=UPDATES,rate_cap_hz=CAP_HZ,teacher_kind=b['teacher_kind'],
         selection='fixed final update250; no heldout teacher labels exist at fit start',brain_weights_frozen=True))
    model=Encoder()
    def objective(m):return ((m(x)-y)/CAP_HZ).square().mean()
    # Actual compound loss and all eight parameters, not a toy differentiable proxy.
    loss=objective(model);loss.backward();checks=[]
    for name,param in model.named_parameters():
        analytic=param.grad.clone()
        for index in np.ndindex(tuple(param.shape)):
            old=param[index].item()
            with torch.no_grad():
                param[index]=old+1e-5;plus=objective(model).item()
                param[index]=old-1e-5;minus=objective(model).item();param[index]=old
            fd=(plus-minus)/2e-5;error=abs(fd-analytic[index].item())
            assert error<1e-8,(name,index,error)
            checks.append(dict(parameter=name,index=index,analytic=analytic[index].item(),finite_difference=fd,error=error))
    model.zero_grad()
    fit=HERE/'FIT';fit.mkdir();opt=torch.optim.AdamW(model.parameters(),lr=.03,weight_decay=0.)
    began=time.monotonic();log=[];shadow=None;shadow_steps=0;resume_pass=False
    initial_parameters={k:v.clone() for k,v in model.named_parameters()}
    first_gradient=None;adam_error=None
    for update in range(UPDATES+1):
        if time.monotonic()-began>180:raise TimeoutError('fixed training wall budget')
        if update:
            opt.zero_grad();loss=objective(model);loss.backward()
            if update==1:first_gradient={k:v.grad.clone() for k,v in model.named_parameters()}
            opt.step()
            if update==1:
                errors=[]
                for name,param in model.named_parameters():
                    g=first_gradient[name];m=.1*g;v=.001*g.square()
                    expected=initial_parameters[name]-(.03/(1-.9))*m/(v.sqrt()/((1-.999)**.5)+1e-8)
                    errors.append((expected-param).abs().max().item())
                    state=opt.state[param];assert state['step'].item()==1
                    assert torch.allclose(state['exp_avg'],m,rtol=0,atol=1e-15)
                    assert torch.allclose(state['exp_avg_sq'],v,rtol=0,atol=1e-15)
                adam_error=max(errors);assert adam_error<1e-12,adam_error
            if update==126:
                sm,so=shadow;so.zero_grad();objective(sm).backward();so.step();shadow_steps+=1
                assert equal(model.state_dict(),sm.state_dict()) and equal(opt.state_dict(),so.state_dict())
                resume_pass=True
        with torch.no_grad():
            prediction=model(x);mse=float(((prediction-y)/CAP_HZ).square().mean())
            rmse=((prediction-y).square().mean(0).sqrt()/y.square().mean(0).sqrt()).tolist()
        metrics=dict(update=update,normalized_mse=mse,nrmse_by_source=rmse)
        path=fit/f'update{update:03d}.pt'
        torch.save(dict(state=model.state_dict(),optimizer=opt.state_dict(),update=update,
                        rng=torch.get_rng_state(),input_manifest_sha256=sha(HERE/'FIT_INPUTS.json'),metrics=metrics),path)
        metrics['sha256']=sha(path);log.append(metrics)
        if update==125:
            saved=torch.load(path,weights_only=True,map_location='cpu')
            sm=Encoder();sm.load_state_dict(saved['state']);so=torch.optim.AdamW(sm.parameters(),lr=.03,weight_decay=0.)
            so.load_state_dict(saved['optimizer']);assert torch.equal(saved['rng'],torch.get_rng_state())
            shadow=(sm,so)
        if update%50==0:print('update',update,'mse',mse,'nrmse',rmse,flush=True)
    verify(pins)
    write(HERE/'FIT_RESULT.json',dict(status='ENGINEERING_REFERENCE_FITTED_PENDING_INDEPENDENT_VALIDATION',
       optimizer_updates=UPDATES,qualification_shadow_updates=shadow_steps,frames=len(x),log=log,
       actual_loss_gradient_checks=checks,first_adam_max_error=adam_error,checkpoint_resume_exact=resume_pass,
       candidate=str(fit/f'update{UPDATES:03d}.pt'),candidate_sha256=sha(fit/f'update{UPDATES:03d}.pt'),
       source_sha256=pins,brain_steps=0,new_neck_ticks=0,new_physical_steps=0,wall_seconds=time.monotonic()-began))


if __name__=='__main__':main()
