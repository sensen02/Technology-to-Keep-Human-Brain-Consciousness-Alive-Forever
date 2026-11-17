"""Parent NN-only full-history loss/gradient/Adam and actualsource preflight."""
import argparse
import copy
import hashlib
import sys
import time
from sources import ROOT,read,save,verify


def tensor_digest(values):
    h=hashlib.sha256()
    for name,tensor in sorted(values.items()):h.update(name.encode());h.update(str(tensor.dtype).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def max_error(a,b):
    import torch
    if torch.is_tensor(a):
        if a.dtype==torch.bool:return 0. if torch.equal(a,b) else 1.
        return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Optimizer/state keys differ')
        return max([max_error(a[k],b[k]) for k in a] or [0.])
    if isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError('State lengths differ')
        return max([max_error(x,y) for x,y in zip(a,b)] or [0.])
    if a!=b:raise ValueError('Nontensor state differs')
    return 0.


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-parent-preflight',action='store_true');args=parser.parse_args()
    if not args.execute_parent_preflight:parser.error('Explicit parentNNpreflight required')
    if (ROOT/'PREFLIGHT.json').exists():raise FileExistsError(ROOT/'PREFLIGHT.json')
    from learning import cpu1,rng_snapshot,restore_rng,objective,adapter_anchor,resource_gate
    cpu1()
    import torch,numpy as np,random
    import common,model
    from data import prepare,production_sequence
    torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    binding=read(ROOT/'INPUT_BINDINGS.json');verify(binding['source_hashes']);verify(binding['data_hashes']);start=time.monotonic()
    setup_start=time.monotonic();m,payload=common.initialize();scan=model.make_scan(m);data=prepare(m);setup_seconds=time.monotonic()-setup_start;checks={}
    checks['actual8sources_1920_370']=data['features'].shape==(1920,8,370)
    checks['oldmask112_state352']=int(m.old.base.physical_mask.bool().sum())==112 and [x.shape[-1] for x in m.initial_state(1)]==[128,64,64,96]
    params=list(m.adapter_parameters());checks['new8params_only']=len(params)==8 and all(not p.requires_grad for p in m.old.parameters())
    frozen=tensor_digest(m.old.state_dict());initial={k:v.detach().clone() for k,v in m.state_dict().items()};rng=rng_snapshot();parities=[]
    with torch.no_grad():
        x=data['x'][0,0:1];states=[m.initial_state(1),m.initial_state(1),m.initial_state(1)];outputs=[]
        for speed,state in zip((0.,.37,1.),states):
            mean,after=m.step(x,x.new_tensor([[speed,.5]]),state);outputs.append((mean,after))
        checks['headzero_arbitrary_independent_speed_oldparity']=all(torch.equal(outputs[0][0],v[0]) and all(torch.equal(a,b) for a,b in zip(outputs[0][1][:3],v[1][:3])) for v in outputs[1:])
    # Full1920 production vs cached scan, all8actual inputs, headzero exactoldmean.
    parity_start=time.monotonic()
    with torch.no_grad():
        residual,h=scan(data['features']);checks['zerohead_all8_oldmean_exact']=bool(torch.equal(residual,torch.zeros_like(residual)))
        for i,raw in enumerate(data['raw']):
            prod,hidden=production_sequence(m,raw)
            checks[raw['entry']['id']+'_head0_production_oldmean_exact']=torch.equal(prod,data['oldmean'][:,i])
            checks[raw['entry']['id']+'_cache_production_old256_exact']=torch.equal(hidden[:,:256],data['features'][:,i,112:368])
            err=float((hidden[:,256:]-h[:,i]).abs().max());checks[raw['entry']['id']+'_cache_python_JIT_new96']=err<=1e-10;parities.append(err)
    parity_seconds=time.monotonic()-parity_start
    del residual,h,prod,hidden
    # Nonzero head temporary probe makes encoder+GRU gradients informative.
    with torch.no_grad():m.head.weight.copy_(torch.linspace(-.01,.01,m.head.weight.numel(),dtype=torch.float64).reshape_as(m.head.weight))
    probe_state={k:v.detach().clone() for k,v in m.state_dict().items()};anchor=adapter_anchor(m)
    with torch.no_grad():
        scripted_r,scripted_h=scan(data['features'])
        python_r,python_h=m.residual_sequence(data['features'])
        checks['FULL1920_nonzerohead_Python_JIT_forward']=float((scripted_r-python_r).abs().max())<=1e-10 and float((scripted_h-python_h).abs().max())<=1e-10
        del scripted_r,scripted_h,python_r,python_h
    # Auxiliary2coordinate central differences on32ticks; never substitutes fullBPTT.
    m.zero_grad(set_to_none=True)
    fd_r,fd_h=scan(data['features'][:32]);fd_loss,_=objective(fd_r,data['oldmean'][:32],data['y'][:32],data['v1'],m,anchor);fd_loss.backward();fd_errors=[]
    del fd_loss,fd_r,fd_h
    for parameter,index in ((m.head.weight,(0,0)),(m.encoder.weight,(0,0))):
        analytic=float(parameter.grad[index]);original=float(parameter[index]);values=[];epsilon=1e-6
        with torch.no_grad():
            for sign in (1.,-1.):
                parameter[index]=original+sign*epsilon;rr,hh=scan(data['features'][:32]);ll,_=objective(rr,data['oldmean'][:32],data['y'][:32],data['v1'],m,anchor);values.append(float(ll));del rr,hh,ll
            parameter[index]=original
        fd_errors.append(abs((values[0]-values[1])/(2*epsilon)-analytic))
    checks['aux32_two_parameter_central_difference']=max(fd_errors)<=1e-6
    m.zero_grad(set_to_none=True)
    full_opt=torch.optim.Adam(params,lr=3e-4);full_opt.zero_grad(set_to_none=True)
    residual,h=scan(data['features']);full_loss,terms=objective(residual,data['oldmean'],data['y'],data['v1'],m,anchor);full_loss.backward();full_value=float(full_loss.detach())
    full_grad=[p.grad.detach().clone() for p in params];del full_loss,residual,h
    full_opt.step();full_after={k:v.detach().clone() for k,v in m.state_dict().items()};full_adam=copy.deepcopy(full_opt.state_dict())
    del full_opt
    m.load_state_dict(probe_state,strict=True);micro_opt=torch.optim.Adam(params,lr=3e-4);micro_opt.zero_grad(set_to_none=True);micro_value=0.;timed=time.monotonic()
    for lo in (0,4):
        residual,h=scan(data['features'][:,lo:lo+4]);loss,terms=objective(residual,data['oldmean'][:,lo:lo+4],data['y'][:,lo:lo+4],data['v1'][lo:lo+4],m,anchor);loss.backward();micro_value+=float(loss.detach());del loss,residual,h
    grad_error=max(float((p.grad-q).abs().max()) for p,q in zip(params,full_grad));micro_opt.step();update_seconds=time.monotonic()-timed
    adam_error=max_error(full_adam,micro_opt.state_dict());parameter_error=max_error(full_after,m.state_dict())
    checks['FULL1920_loss_full8_vs_micro4x2']=abs(full_value-micro_value)<=1e-10
    checks['FULL1920_all8grad_full_vs_micro']=grad_error<=1e-10
    checks['FULL1920_Adam_full_vs_micro']=adam_error<=1e-10 and parameter_error<=1e-10
    checks['all8Adam_moments_present']=len(micro_opt.state_dict()['state'])==8
    checks['old_frozen_unchanged']=tensor_digest(m.old.state_dict())==frozen and all(p.grad is None for p in m.old.parameters())
    m.load_state_dict(initial,strict=True);m.zero_grad(set_to_none=True);restore_rng(rng)
    checks['initial_zerohead_restored']=all(torch.equal(v,m.state_dict()[k]) for k,v in initial.items())
    checks['no_physics_or_sensorybackend']=not any(k in sys.modules for k in ('mujoco','physics','flygym','student_wave'))
    verification_allowance=3.*(setup_seconds+parity_seconds)*1.3
    estimated=update_seconds*600*1.3+verification_allowance;timing_ok=estimated<=1800.;rss=resource_gate(start,1800.)
    verify(binding['source_hashes']);verify(binding['data_hashes'])
    save(ROOT/'PREFLIGHT.json',dict(all_correctness_checks_passed=bool(all(checks.values())),all_passed=bool(all(checks.values()) and timing_ok),timing_status='READY_FIXED600' if timing_ok else 'BLOCKED_BEFORE_TRAINING_EXPECTED_WALL',checks=checks,full_history=1920,grad_error=grad_error,Adam_error=adam_error,parameter_error=parameter_error,loss_error=abs(full_value-micro_value),production_cache_errors=parities,actual8data_proofs=data['proofs'],micro4x2_full_update_s=update_seconds,actual_setup_cache_s=setup_seconds,actual_full8production_parity_s=parity_seconds,fresh3savedverification_allowance_s=verification_allowance,predicted600_wall_s_with1p3margin=estimated,peak_RSS_MiB=rss,old_weights_digest=frozen,source_hashes=binding['source_hashes'],data_hashes=binding['data_hashes'],physical_steps=0,optimizer_probe_updates=2,training_updates=0,scope='Numerical source/gradient/cache/storage readiness only, not body motion or trainingcomplete'))
    if not all(checks.values()):raise ValueError('Correctness preflight failed')
    if not timing_ok:raise ValueError('Timing blocked beforetraining; parentmust preregister newbudget/architecture, no automatic shrink')
    print('PREFLIGHT_PASS_FIXED600_NO_PHYSICS',flush=True)


if __name__=='__main__':main()
