"""Parent numericrange/migration/full1920 lossgradAdam/timing gates, zero physics."""
import argparse
import copy
import sys
import time
from sources import ROOT,read,save,verify
from adapter import PARENT,module,max_error,tensor_digest

PARENT_SHA='007d0c74118715eacd03d49c7e6847d8349e55db2793dd9d461a4b9c311e0fcd'


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-preflight',action='store_true');a=p.parse_args()
    if not a.execute_parent_preflight:p.error('Parentlocalmeanfullpreflightrequired')
    if (ROOT/'PREFLIGHT.json').exists():raise FileExistsError(ROOT/'PREFLIGHT.json')
    start=time.monotonic()
    from learning import cpu1,rng_snapshot,rng_equal,restore_rng,adapter_anchor,objective,time_weights,resource_gate
    cpu1()
    import torch,numpy as np,random
    import common,model
    from data import prepare,dev_sources,production_sequence
    torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    b=read(ROOT/'INPUT_BINDINGS.json');verify(b['source_hashes']);verify(b['data_hashes']);setup=time.monotonic()
    m,parentpayload=common.initialize();scan=model.make_scan(m);data=prepare(m);setup_seconds=time.monotonic()-setup
    checks={};initial={k:v.detach().clone() for k,v in m.state_dict().items()};rng=rng_snapshot();olddigest=tensor_digest(m.old.state_dict())
    checks['12full1920_412_directoldmean']=data['features'].shape==(1920,12,412) and torch.equal(data['features'][:,:,-42:],data['oldmean']) and int(data['v1'].sum())==6
    weights=time_weights(1920,data['y'],data['valid'])
    base=data['y'].new_ones((1920,1));base[:20]=10.;independent=(base*data['valid']);mass=independent.sum(0);independent=independent/mass[None,:]
    checks['first20_masked_perseq_mass_normalized']=bool((mass>0).all()) and float((weights-independent).abs().max())<=1e-15 and float((weights.sum(0)-1.).abs().max())<=1e-12 and bool((weights[data['valid']==0]==0).all())
    floor_rows=[]
    with torch.no_grad():
        lo=(data['oldmean']-6.).tanh();hi=(data['oldmean']+6.).tanh();best=torch.minimum(torch.maximum(data['y'],lo),hi);sq=(best-data['y']).square()
        for i,raw in enumerate(data['raw']):
            keep=data['valid'][:,i].bool();floor=float(sq[:,i][keep].mean().sqrt());floor_rows.append(dict(id=raw['entry']['id'],valid_rows=int(keep.sum()),weighted_mass=float(mass[i]),valid_action_RMSE_floor=floor,out_of_range_fraction=float(((data['y'][:,i][keep]<lo[:,i][keep])|(data['y'][:,i][keep]>hi[:,i][keep])).double().mean())))
    checks['valid12_range6_RMSEfloor_perseq_le_p01']=all(r['valid_action_RMSE_floor']<=.01 for r in floor_rows)
    if not checks['valid12_range6_RMSEfloor_perseq_le_p01']:
        verify(b['source_hashes']);verify(b['data_hashes']);save(ROOT/'PREFLIGHT.json',dict(all_passed=False,checks=checks,timing_status='BLOCKED_BEFORE_TRAINING_REPRESENTABILITY',range_floor=floor_rows,source_hashes=b['source_hashes'],data_hashes=b['data_hashes'],training_updates=0,physical_steps=0));raise ValueError('Range6validtargetfloor exceedspreregistered.01, no mask/bound change')
    parentio=module('_localmean_preflight_parent_io',PARENT/'common.py');parent,_=parentio.load_checkpoint(PARENT/'TRAIN/checkpoint_000600.pt',PARENT_SHA,final=True)
    checks['migration_encoder_first370_bias_GRU_head_exact']=torch.equal(m.encoder.weight[:,:370],parent.encoder.weight) and not bool(m.encoder.weight[:,370:].abs().max()) and torch.equal(m.encoder.bias,parent.encoder.bias) and max_error(m.gru.state_dict(),parent.gru.state_dict())==0 and max_error(m.head.state_dict(),parent.head.state_dict())==0
    timer=time.monotonic();migration=[]
    with torch.no_grad():
        cachedres,cachedh=scan(data['features']);pyres,pyh=m.residual_sequence(data['features'])
        checks['full1920_Python_JIT_same_model']=float((cachedres-pyres).abs().max())<=1e-12 and float((cachedh-pyh).abs().max())<=1e-12;del pyres,pyh
        for i,raw in enumerate(data['raw']+dev_sources(b)):
            newmean,newstate=production_sequence(m,raw);oldmean,oldstate=production_sequence(parent,raw);me=float((newmean-oldmean).abs().max());ae=float((newmean.tanh()-oldmean.tanh()).abs().max());he=float((newstate[:,256:]-oldstate[:,256:]).abs().max());oe=torch.equal(newstate[:,:256],oldstate[:,:256])
            ns=m.initial_state(1);ps=parent.initial_state(1)
            for x in raw['warm']:_,ns=m.prime_old(x[None],x.new_zeros((1,2)),ns);_,ps=parent.prime_old(x[None],x.new_zeros((1,2)),ps)
            re=0.
            for x,u in zip(raw['x'],raw['u']):
                _,ns,nr=m.step_details(x[None],u[None],ns);_,ps,pr=parent.step_details(x[None],u[None],ps);re=max(re,float((nr-pr).abs().max()))
            tag=raw['entry']['id']+str(i);checks[tag+'_migration']=me<=1e-12 and ae<=1e-12 and he<=1e-12 and re<=1e-12 and oe
            if i<12:checks[tag+'_cache_production']=float((newmean-data['oldmean'][:,i]-cachedres[:,i]).abs().max())<=1e-12 and float((newstate[:,256:]-cachedh[:,i]).abs().max())<=1e-12
            migration.append(dict(id=raw['entry']['id'],training=i<12,actual_rows=len(raw['x']),mean_error=me,action_error=ae,new96_error=he,residual_error=re,old256exact=oe,labels_available=i<12))
            resource_gate(start)
    parity_seconds=time.monotonic()-timer;del cachedres,cachedh,newmean,newstate,oldmean,oldstate
    # Numeric fixed-input routing, not autograd through detachedfrozen cache.
    with torch.no_grad():
        x=data['x'][0,0:1].clone();u=data['u'][0,0:1];state=tuple(s[0:1].clone() for s in data['initial_old']);newstate=m.new_state_from_old(state);base,F,after=m.cached_features(x,u,state);mean0,s0,res0=m.step_details(x,u,newstate)
        for begin,end in ((42,84),(195,237)):
            perturbed=x.clone();perturbed[:,begin:end]+=.03125;bm,Fm,sm=m.cached_features(perturbed,u,state);mn,hn,rn=m.step_details(perturbed,u,newstate)
            checks['numeric_excluded_'+str(begin)+'_'+str(end)]=torch.equal(base,bm) and torch.equal(F,Fm) and all(torch.equal(a,b) for a,b in zip(after,sm)) and torch.equal(mean0,mn) and torch.equal(res0,rn)
        target=x.clone();target[:,126:168]+=.03125;changed,Fchanged,s_changed=m.cached_features(target,u,state)
        checks['target126_changes_oldmean_not_old256']=not torch.equal(changed,base) and all(torch.equal(a,b) for a,b in zip(after,s_changed))
        saved=m.encoder.weight.detach().clone();preactivation=m.encoder(F);j=int(preactivation.abs().argmin());col=int((changed-base).abs().argmax());m.encoder.weight[j,370+col]=.01
        encoded0=m.encoder(F).tanh();encoded1=m.encoder(Fchanged).tanh();route=float((encoded0-encoded1).abs().max());checks['causal_oldmean42_numeric_newencoder_route']=route>0.
        m.encoder.weight.copy_(saved)
    params=list(m.adapter_parameters());anchor=adapter_anchor(m);probe={k:v.detach().clone() for k,v in m.state_dict().items()};m.zero_grad(set_to_none=True)
    opt1=torch.optim.Adam(params,lr=3e-4);res,h=scan(data['features']);loss,_=objective(res,data['oldmean'],data['y'],data['v1'],m,anchor,valid=data['valid']);full=float(loss.detach());loss.backward();grads=[p.grad.detach().clone() for p in params];del res,h,loss;opt1.step();fullstate={k:v.detach().clone() for k,v in m.state_dict().items()};fulladam=copy.deepcopy(opt1.state_dict());del opt1
    m.load_state_dict(probe,strict=True);opt=torch.optim.Adam(params,lr=3e-4);opt.zero_grad(set_to_none=True);timer=time.monotonic();micro=0.
    for lo in (0,4,8):
        res,h=scan(data['features'][:,lo:lo+4]);loss,_=objective(res,data['oldmean'][:,lo:lo+4],data['y'][:,lo:lo+4],data['v1'][lo:lo+4],m,anchor,valid=data['valid'][:,lo:lo+4]);loss.backward();micro+=float(loss.detach());del res,h,loss
    ge=max(float((p.grad-g).abs().max()) for p,g in zip(params,grads));opt.step();update_seconds=time.monotonic()-timer;adam=max_error(fulladam,opt.state_dict());pe=max_error(fullstate,m.state_dict())
    checks['full1920_B12_3micro_loss']=abs(full-micro)<=1e-10;checks['full1920_all8grad']=ge<=1e-10;checks['full1920_Adam_andparams']=adam<=1e-10 and pe<=1e-10
    checks['frozenold_unchanged']=tensor_digest(m.old.state_dict())==olddigest and all(not p.requires_grad and p.grad is None for p in m.old.parameters())
    m.load_state_dict(initial,strict=True);m.zero_grad(set_to_none=True);restore_rng(rng);checks['initialmigration_andRNG_restored']=max_error(initial,m.state_dict())==0 and rng_equal(rng,rng_snapshot())
    checks['no_physics_or_teacherquery']=not any(k in sys.modules for k in ('mujoco','physics','flygym','static_teacher','student_wave'))
    allowance=3*(setup_seconds+parity_seconds)*1.3;estimate=update_seconds*600*1.3+allowance;timing=estimate<=3600;rss=resource_gate(start)
    verify(b['source_hashes']);verify(b['data_hashes'])
    save(ROOT/'PREFLIGHT.json',dict(all_passed=bool(all(checks.values()) and timing),all_correctness_checks_passed=bool(all(checks.values())),checks=checks,timing_status='READY_FIXED600_3600S' if timing else 'BLOCKED_BEFORE_TRAINING_TIMING',range_floor=floor_rows,migration20sequences=migration,numeric_causal_encoder_route_delta=route,full_history=1920,loss_error=abs(full-micro),gradient_error=ge,Adam_error=adam,parameter_error=pe,micro4x3_update_s=update_seconds,fresh3saveverify_allowance_s=allowance,predicted600_total_s_with_margin=estimate,peak_RSS_MiB=rss,source_hashes=b['source_hashes'],data_hashes=b['data_hashes'],training_updates=0,optimizer_probe_updates=2,physical_steps=0,scope='Numerical/coupling/mask/source/storage readiness only, no bodyPASS or unique causalclaim'))
    if not all(checks.values()):raise ValueError('Localmeanfullnumericalpreflightfailed')
    if not timing:raise ValueError('Timingblockedbeforetraining,noautobudgetextension')
    print('LOCALMEAN_PREFLIGHT_READY_FIXED600_3600S',flush=True)


if __name__=='__main__':main()
