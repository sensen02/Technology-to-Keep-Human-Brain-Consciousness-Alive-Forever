"""Target existing RH motor neurons at release, with readout and full initial state frozen."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MUJOCO_GL','egl')
import sys,importlib.util,copy
from pathlib import Path
import numpy as np
SRC_CODE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-neural-joint-torque/20261105T153224Z/train_edge_readout.py')
spec=importlib.util.spec_from_file_location('edge',SRC_CODE);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T181843Z')
SRC=Path('/home/sensen/Desktop/engine修复验收/s5-neural-joint-torque/20261105T153657Z')
e.O=O;e.b.O=O
js=e.js;dump=e.dump;sha=e.sha

def inputs():
    best=js(SRC/'SUMMARY.json')['P0']['best']
    return e.b.sources()+[SRC/'SUMMARY.json',SRC/'COVERAGE.json',Path(best['matrix_file']),O/'CHAIN_AUDIT.json',SRC_CODE,e.D,Path(__file__)]
def prereg():
    audit=js(O/'CHAIN_AUDIT.json');rh=next(r for r in audit['per_leg'] if r['leg']=='RH')
    cfg=dict(stage='S5 RH loss-of-contact causal neural interventions',baseline='latest best P0 readout fixed, same complete warmed physical/neural/filter state',
        target_cells=rh['cells'],silent_target_cells=rh['silent_cells'],
        interventions=[dict(label='baseline'),*[dict(label='RH_threshold_'+str(v),threshold_shift_mV=v) for v in [-5.,-2.,2.,5.]],dict(label='RH_silent_threshold_-5',silent_only=True,threshold_shift_mV=-5.),dict(label='RH_incoming_exc_half',exc_scale=.5),dict(label='RH_incoming_inh_x1.5',inh_scale=1.5),*[dict(label='RH_readout_gain_'+str(v),readout_scale=v) for v in [.25,.5,2.]],dict(label='RH_readout_cut',readout_scale=0.)],
        interpretation='test excessive drive and underdrive symmetrically; firing increase is not success. Gain-only controls separate neural changes from mechanical readout amplitude.',
        integrity='threshold offset changes only selected existing cells, reset stays below threshold; We/Wi incoming rows scaled without changing sparsity/sign. No background current, spikes injected, new edges or force assistance.',
        timing='intervention at release; do not reset membrane/conductance/filter/pending or body; threshold/gain effects enter naturally through next neural step and completed command window',
        budget=dict(primary=12,validation=3,total=15,selection='lowest J among all primary including baseline; exact repeat, readout cut, sensory cut of chosen best'),
        metrics='existing J, D1-D8; separately RH spikes, foot load, first contact, foot trajectory, and per-joint torque saturation; no claim of walking from score',
        body_dt_s=e.b.DT,neural_dt_s=e.b.NDT,command_dt_s=e.b.CDT,warmup_s=1.3,release_s=.2,resource='one process',units='NOT_ESTABLISHED',importance='核心必需')
    dump(O/'PREREGISTRATION.json',cfg);dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in inputs()})
    dump(O/'CHANGE_LEDGER.json',dict(old_file=str(SRC/'SUMMARY.json'),old_sha256=sha(SRC/'SUMMARY.json'),new_file=str(O/'PREREGISTRATION.json'),new_sha256=sha(O/'PREREGISTRATION.json'),reason='Causal audit shows active RH neurons and strong drive but no contact; test existing neuron excitability and input balance without changing anatomy/readout topology.'))

def apply(t,basecache,cfg,cells,silent):
    t.cache['P0']=copy.deepcopy(basecache);net=t.cache['P0']['net'];idx=np.array(silent if cfg.get('silent_only') else cells,int)
    oldth=net.threshold_offset_mV.copy();newth=oldth.copy();newth[idx]+=cfg.get('threshold_shift_mV',0.);assert np.all(net.p.reset_mV<net.p.threshold_mV+newth);newth.flags.writeable=False;net.threshold_offset_mV=newth
    changes={}
    for key,scale in [('We',cfg.get('exc_scale',1.)),('Wi',cfg.get('inh_scale',1.))]:
        w=getattr(net,key);old=w.copy();new=w.copy();rows=np.repeat(np.arange(net.n),np.diff(w.indptr));new.data[np.isin(rows,idx)]*=scale
        assert np.array_equal(old.indices,new.indices) and np.array_equal(old.indptr,new.indptr) and np.all(new.data>=0)
        unchanged=~np.isin(rows,idx);assert np.array_equal(old.data[unchanged],new.data[unchanged]);setattr(net,key,new)
        changes[key]=dict(scale=scale,nnz=int(new.nnz),sparsity_unchanged=True,untargeted_weights_unchanged=True,changed_weights=int(np.count_nonzero(old.data!=new.data)))
    assert np.array_equal(net.v,basecache['net'].v) and np.array_equal(net.ge,basecache['net'].ge) and np.array_equal(net.gi,basecache['net'].gi)
    return dict(config=cfg,target_cells=idx,old_threshold=oldth[idx],new_threshold=newth[idx],synapses=changes,membrane_conductance_state_preserved=True)

def run():
    before=js(O/'INPUT_HASHES_BEFORE.json');assert before=={str(p):sha(p) for p in inputs()};cfg=js(O/'PREREGISTRATION.json')
    t=e.dmod.DiagnosticTrial();(O/'COVERAGE.json').rename(O/'DOMINANT_COVERAGE.json');cover=js(SRC/'COVERAGE.json');dump(O/'COVERAGE.json',cover)
    baseline=js(SRC/'SUMMARY.json')['P0']['best'];matrix=np.array(js(baseline['matrix_file'])['matrix']);t.warm('P0');cache=copy.deepcopy(t.cache['P0'])
    manifest=js(e.b.B/'RUN/MAPPING.json')['rows'];poollegs={tuple(r['local_indices']):r['leg'] for r in manifest if r['local_indices']};rhcols=[i for i,p in enumerate(t.pools) if poollegs[tuple(p)]=='RH']
    results=[];best=None;bestcfg=None
    def one(c,label,cut=False,sens=False):
        intervention=apply(t,cache,c,cfg['target_cells'],cfg['silent_target_cells']);M=matrix.copy();M[:,rhcols]*=c.get('readout_scale',1.)
        # exact intervention is recorded before the evaluation starts
        pre=O/(label+'_INTERVENTION.json');dump(pre,intervention)
        r=e.evaluate(t,'P0',M,label,sensory_cut=sens,cut=cut,previous=str(pre));dump(Path(r['folder'])/'NEURAL_INTERVENTION.json',dict(file=str(pre),sha256=sha(pre)))
        n=np.load(Path(r['folder'])/'NEURAL.npz');a=np.load(Path(r['folder'])/'NATIVE.npz')['rows'];bod=np.load(Path(r['folder'])/'BODY.npz');cells=cfg['target_cells'];force=a[:,6]
        first=np.flatnonzero(force>1e-9)
        r['RH']=dict(spikes=int(n['spikes'][:,cells].sum()),active_cells=int(np.count_nonzero(n['spikes'][:,cells].sum(0))),foot_load_mean=float(force.mean()/24.446354755878005),first_contact_s=None if len(first)==0 else float(a[first[0],0]-1.3),foot_z_60ms=float(bod['feet'][120,5,2]))
        dump(Path(r['folder'])/'RH_RESULT.json',r['RH']);return r
    try:
        for c in cfg['interventions']:
            r=one(c,c['label']);results.append(r)
            if best is None or r['J']<best['J']:best=r;bestcfg=c
            dump(O/'PROGRESS.json',dict(results=results,best=best))
            if c['label']=='baseline':
                old=np.load(Path(baseline['folder'])/'BODY.npz');new=np.load(Path(r['folder'])/'BODY.npz');assert r['J']==baseline['J'] and all(np.array_equal(old[k],new[k]) for k in old.files)
        repeat=one(bestcfg,'best_repeat');cut=one(bestcfg,'best_readout_cut',cut=True);sens=one(bestcfg,'best_sensory_cut',sens=True)
        a=np.load(Path(best['folder'])/'BODY.npz');bb=np.load(Path(repeat['folder'])/'BODY.npz');assert best['J']==repeat['J'] and all(np.array_equal(a[k],bb[k]) for k in a.files)
        dump(O/'SUMMARY.json',dict(results=results,baseline=results[0],best=best,best_intervention=bestcfg,repeat=repeat,cut=cut,sensory_cut=sens,exact_baseline=True,exact_repeat=True,total_evaluations=15,walking_certified=False))
        print('NEURAL PROBE COMPLETE',flush=True)
    finally:
        after={str(p):sha(p) for p in inputs()};dump(O/'INPUT_HASHES_AFTER.json',after);assert before==after
if __name__=='__main__':
    if sys.argv[1]=='prereg':prereg()
    elif sys.argv[1]=='run':run()
