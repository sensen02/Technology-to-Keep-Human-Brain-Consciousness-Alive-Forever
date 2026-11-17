"""Parent numericalqualification: migration12seq, full1920 gradients, timing only."""
import argparse
import ast
import copy
import time
import sys
from sources import ROOT,PARENT,read,save,verify,module,SHA

# The exact stable tensor/state comparator is stdlib-defined; nobackend import.
_ns={'__name__':'_fastpath_state_comparator','__file__':str(PARENT/'preflight.py')}
_tree=ast.parse((PARENT/'preflight.py').read_text())
exec(compile(ast.Module(body=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in ('max_error','tensor_digest')],type_ignores=[]),str(PARENT/'preflight.py')+'::exact_comparators','exec'),_ns)
# tensor_digest resolves hashlib when invoked below.
import hashlib
_ns['hashlib']=hashlib
max_error=_ns['max_error'];tensor_digest=_ns['tensor_digest']


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-preflight',action='store_true');a=p.parse_args()
    if not a.execute_parent_preflight:p.error('Parentfastpathnumericalpreflight required')
    if (ROOT/'PREFLIGHT.json').exists():raise FileExistsError(ROOT/'PREFLIGHT.json')
    from learning import cpu1,rng_snapshot,restore_rng,objective,adapter_anchor,time_weights,resource_gate
    cpu1()
    import torch,numpy as np,random
    import common,model
    from data import prepare,production_sequence,dev_sources
    binding=read(ROOT/'INPUT_BINDINGS.json');verify(binding['source_hashes']);verify(binding['data_hashes']);start=time.monotonic()
    torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    setup=time.monotonic();m,parentpayload=common.initialize();scan=model.make_scan(m);data=prepare(m);setup_seconds=time.monotonic()-setup
    parentio=module('_fastpreflight_parent_strict',PARENT/'common.py');parent,_=parentio.load_checkpoint(PARENT/'TRAIN/checkpoint_000600.pt',SHA,final=True)
    checks={};migration=[];parity_start=time.monotonic()
    olddigest=tensor_digest(m.old.state_dict());initial={k:v.detach().clone() for k,v in m.state_dict().items()};rng=rng_snapshot()
    checks['state352_head192_8params']=list(m.identity()['state_dims'])==[128,64,64,96] and tuple(m.head.weight.shape)==(42,192) and len(list(m.adapter_parameters()))==8
    checks['parent_encoder_GRU_exact']=max_error(m.encoder.state_dict(),parent.encoder.state_dict())==0 and max_error(m.gru.state_dict(),parent.gru.state_dict())==0
    checks['head_migration_exact']=torch.equal(m.head.weight[:,:96],parent.head.weight) and not bool(m.head.weight[:,96:].abs().max()) and torch.equal(m.head.bias,parent.head.bias)
    weight=time_weights(1920,data['y']);checks['first20weight10_normalized_exact']=int((weight>weight[-1]*1.01).sum())==20 and float((weight[:20]/weight[-1]-10.).abs().max())<1e-12 and abs(float(weight.sum())-1.)<1e-12
    with torch.no_grad():
        cachedres,cachedh=scan(data['features'])
        pythonres,pythonh=m.residual_sequence(data['features']);checks['full1920_python_JIT_parity']=float((pythonres-cachedres).abs().max())<=1e-12 and float((pythonh-cachedh).abs().max())<=1e-12
        del pythonres,pythonh
        for i,raw in enumerate(data['raw']+dev_sources(binding)):
            newmean,newstates=production_sequence(m,raw);oldmean,oldstates=production_sequence(parent,raw)
            # Residual from identicalstates/direct step is assessed independently below.
            err=float((newmean-oldmean).abs().max());actionerr=float((newmean.tanh()-oldmean.tanh()).abs().max())
            checks[raw['entry']['id']+('_train' if i<8 else '_DEV')+'_migration_mean_action_state']=err<=1e-12 and actionerr<=1e-12 and torch.equal(newstates,oldstates)
            residualerror=0.
            state=m.initial_state(1);ps=parent.initial_state(1)
            for row in raw['warm']:
                _,state=m.prime_old(row[None],row.new_zeros((1,2)),state);_,ps=parent.prime_old(row[None],row.new_zeros((1,2)),ps)
            for x,u in zip(raw['x'],raw['u']):
                _,state,r=m.step_details(x[None],u[None],state);_,ps,pr=parent.step_details(x[None],u[None],ps);residualerror=max(residualerror,float((r-pr).abs().max()))
            checks[raw['entry']['id']+str(i)+'_migration_residual']=residualerror<=1e-12
            if i<8:
                checks[raw['entry']['id']+'_cache_production']=float((newmean-data['oldmean'][:,i]-cachedres[:,i]).abs().max())<=1e-12 and float((newstates[:,256:]-cachedh[:,i]).abs().max())<=1e-12
            migration.append(dict(id=raw['entry']['id'],training=i<8,labels_available=i<8,maxmeanerror=err,maxactionerror=actionerr,maxresidualerror=residualerror,all352states_exact=torch.equal(newstates,oldstates)))
    parity_seconds=time.monotonic()-parity_start;del cachedres,cachedh,newmean,newstates,oldmean,oldstates
    # Direct encoderobservation influence excludes other126 rawchannels by construction.
    xx=data['x'][0,0:1].clone().requires_grad_(True);oldstates=tuple(s[0:1].detach() for s in data['initial_old']);base,f,after=m.cached_features(xx,data['u'][0,0:1],oldstates)
    physical=xx[:,m.old.base.physical_mask.bool()];direct=torch.cat((physical,*[v.detach() for v in after],data['u'][0,0:1]),-1);m.encoder(direct).sum().backward()
    checks['excluded_rawchannels_directgrad_zero']=bool((xx.grad[:,~m.old.base.physical_mask.bool()]==0.).all())
    m.zero_grad(set_to_none=True)
    # Full1920 effective8 vs two complete4 trajectories, newweightedjointloss.
    anchor=adapter_anchor(m);params=list(m.adapter_parameters());probe={k:v.detach().clone() for k,v in m.state_dict().items()}
    fullopt=torch.optim.Adam(params,lr=3e-4);fullopt.zero_grad(set_to_none=True);res,h=scan(data['features']);loss,_=objective(res,data['oldmean'],data['y'],data['v1'],m,anchor);fullvalue=float(loss.detach());loss.backward();grad=[p.grad.detach().clone() for p in params];del res,h,loss
    fullopt.step();fullstate={k:v.detach().clone() for k,v in m.state_dict().items()};fulladam=copy.deepcopy(fullopt.state_dict());del fullopt
    m.load_state_dict(probe,strict=True);opt=torch.optim.Adam(params,lr=3e-4);opt.zero_grad(set_to_none=True);timer=time.monotonic();microvalue=0.
    for lo in (0,4):
        res,h=scan(data['features'][:,lo:lo+4]);loss,_=objective(res,data['oldmean'][:,lo:lo+4],data['y'][:,lo:lo+4],data['v1'][lo:lo+4],m,anchor);loss.backward();microvalue+=float(loss.detach());del res,h,loss
    gradienterror=max(float((p.grad-g).abs().max()) for p,g in zip(params,grad));opt.step();update_seconds=time.monotonic()-timer
    adamerror=max_error(fulladam,opt.state_dict());parametererror=max_error(fullstate,m.state_dict())
    checks['full1920_jointloss_micro_exact']=abs(fullvalue-microvalue)<=1e-10
    checks['full1920_all8grad_micro_exact']=gradienterror<=1e-10
    checks['full1920_Adam_andparams_micro_exact']=adamerror<=1e-10 and parametererror<=1e-10
    checks['oldweights_frozen']=tensor_digest(m.old.state_dict())==olddigest and all(not p.requires_grad and p.grad is None for p in m.old.parameters())
    m.load_state_dict(initial,strict=True);m.zero_grad(set_to_none=True);restore_rng(rng)
    checks['migration_initialweights_restored']=max_error(initial,m.state_dict())==0
    checks['no_physical_or_teacherquery']=not any(k in sys.modules for k in ('physics','mujoco','flygym','student_wave'))
    allowance=3*(setup_seconds+parity_seconds)*1.3;estimate=600*update_seconds*1.3+allowance;timing=estimate<=1800;rss=resource_gate(start,1800)
    verify(binding['source_hashes']);verify(binding['data_hashes'])
    save(ROOT/'PREFLIGHT.json',dict(all_passed=bool(all(checks.values()) and timing),all_correctness_checks_passed=bool(all(checks.values())),checks=checks,timing_status='READY_FIXED600' if timing else 'BLOCKED_BEFORE_TRAINING_EXPECTED_WALL',migration12seq=migration,full_history=1920,grad_error=gradienterror,Adam_error=adamerror,parameter_error=parametererror,loss_error=abs(fullvalue-microvalue),micro4x2_full_update_s=update_seconds,fresh3savedverification_allowance_s=allowance,predicted600_wall_s_with1p3margin=estimate,peak_RSS_MiB=rss,source_hashes=binding['source_hashes'],data_hashes=binding['data_hashes'],training_updates=0,optimizer_probe_updates=2,physical_steps=0,scope='Migration/numerical/readiness only, no behaviorPASS'))
    if not all(checks.values()):raise ValueError('Fastpathnumericalpreflightfailed')
    if not timing:raise ValueError('Timingblockedbeforetraining, noautoshrink')
    print('FASTPATH_NUMERICAL_PREFLIGHT_PASS_READY_FIXED600',flush=True)


if __name__=='__main__':main()
