"""S5 fixed-topology strength fit; bounded adaptive diagonal evolution strategy.
All online actions remain the original SNN + 5ms pool matrix readout.
"""
import os,sys,copy,pickle,time,json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee,connected_components
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=R/'experiments/s5-neural-translator/20261105T200021Z'
sys.path[:0]=[str(BASE),str(R),str(R/'tools')]
import runtime as rt
from engine.neural_cond import ConductanceNetwork,ConductanceParams
O=Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z')
SRC=Path(__file__).resolve().parent
rt.O=O;rt.b.O=O
js,sha,dump=rt.js,rt.sha,rt.dump
ARCH=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
AUD=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z/neural/SENSORY_CURRENTS.npz')
TRAIN=[1.2,1.25,1.35,1.4];GUARD=1.3;DEV=[1.225,1.275,1.325,1.375];FINAL=[1.219,1.269,1.319,1.369]
CAPTURE=sorted(set(rt.TRAIN+rt.DEV+rt.TEST+FINAL))
SEED=20261106

def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}

def cleanup():
    p=O/'COVERAGE.json'
    if p.exists():
        ledger=js(O/'CLEANUP_LEDGER.json') if (O/'CLEANUP_LEDGER.json').exists() else {'deleted':[]}
        ledger['deleted'].append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,importance='可再生临时',reason='Inspected legacy constructor writes only; runtime files() reads historical PRIOR/COVERAGE, never this output. Full pool identities retained.'))
        dump(O/'CLEANUP_LEDGER.json',ledger);p.unlink()

class RecordingRuntime(rt.Runtime):
    recording=False
    def observe_neural(self,d,net,enc,rates):
        result=super().observe_neural(d,net,enc,rates)
        if self.recording:
            self.true_currents.append(result[2].copy());self.true_times.append(float(d.time))
        return result

def graph():
    mats=[sparse.load_npz(rt.b.AB/'A'/('ACTUAL_'+k+'.npz')).tocsr() for k in ['We','Wi']]
    a=(mats[0]+mats[1]);a.data[:]=1;a=(a+a.T).tocsr();a.data[:]=1;a.sort_indices()
    order=reverse_cuthill_mckee(a,symmetric_mode=True);groups=np.empty(12000,int);groups[order]=np.arange(12000)//1200
    count,components=connected_components(a,directed=False)
    rows=[np.repeat(np.arange(12000),np.diff(m.indptr)) for m in mats]
    info=dict(method='reverse Cuthill-McKee ordering of symmetrized actual graph, consecutive equal-size blocks; structural only, not inferred anatomy or optimized communities',cells=12000,groups=10,dimensions=20,isolated_cells=np.flatnonzero(np.diff(a.indptr)==0),no_incoming_cells=np.flatnonzero((np.diff(mats[0].indptr)+np.diff(mats[1].indptr))==0),every_existing_edge_assigned_once=True,connected_components=int(count),component_sizes=np.bincount(components),group_counts=np.bincount(groups),edges=int(sum(m.nnz for m in mats)),group_details=[])
    for g in range(10):
        info['group_details'].append(dict(group=g,cells=int(sum(groups==g)),incoming_E=int(sum(groups[rows[0]]==g)),incoming_I=int(sum(groups[rows[1]]==g)),outgoing_edges=int(sum(np.sum(groups[m.indices]==g) for m in mats))))
    info['cross_group_edges']=int(sum(np.sum(groups[row]!=groups[m.indices]) for row,m in zip(rows,mats)))
    return mats,rows,groups,order,info

class Evolution:
    """Bounded diagonal Gaussian ES, elite recombination and variance adaptation.
    Not CMA-ES: no covariance evolution path or rank-one CMA update.
    Reflection handles bounds symmetrically; fixed three generations, no restarts.
    """
    def __init__(self,seed=SEED):
        self.rng=np.random.default_rng(seed);self.mean=np.zeros(20);self.var=np.full(20,.35**2)
    @staticmethod
    def reflect(x):
        return 1-np.abs((np.asarray(x)+1)%4-2)
    def ask(self):
        z=self.rng.normal(size=(4,20));z=np.concatenate([z,-z]);return self.reflect(self.mean+np.sqrt(self.var)*z)
    def tell(self,x,order):
        elite=np.asarray(x)[np.asarray(order[:4])];weights=np.array([.4,.3,.2,.1]);target=weights@elite
        spread=weights@((elite-self.mean)**2)
        self.mean=.5*self.mean+.5*target;self.var=np.clip(.7*self.var+.3*spread,.05**2,.6**2)
    def state(self):return dict(mean=self.mean,diagonal_variance=self.var)

def selftest():
    e=Evolution();a=e.ask();f=Evolution();assert np.array_equal(a,f.ask()) and np.max(abs(a))<=1
    grid=np.linspace(-15,15,10001);assert np.max(abs(Evolution.reflect(grid)))<=1
    assert np.allclose(Evolution.reflect(np.array([-.9,0,.9])),[-.9,0,.9])
    old=e.mean.copy();e.tell(a,list(range(8)));assert not np.array_equal(old,e.mean) and np.all(e.var>0)
    return dict(passed=True,determinism=True,bounds=True,adaptive_mean_and_variance=True,algorithm='bounded adaptive diagonal evolution strategy, not CMA-ES')

def prereg():
    O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
    mats,rows,groups,order,info=graph()
    np.savez_compressed(O/'GRAPH_GROUPS.npz',groups=groups,ordering=order,root_ids=arrays(rt.b.B/'IDENTITIES.npz')['root_ids'])
    dump(O/'GROUPING.json',info);dump(O/'OPTIMIZER_SELFTEST.json',selftest())
    cfg=dict(stage='S5 route S first bounded strength engineering fit, not biological calibration',train=TRAIN,guard=GUARD,dev=DEV,final=FINAL,final_withheld='New physical states captured during preregistered teacher run; no release scores until selection frozen and gates satisfied',groups=info['method'],dimensions=20,parameters='10 incoming excitation + 10 incoming inhibition multiplicative factors =1+.1*x; x in [-1,1]. All original edges and signs; RH incoming E .375 unchanged release-only.',immutable='topology, sign, neuron params, body, sensory, readout matrix, 5ms filter, limits; no new controller, body/time/phase inputs, PD or CPG in release',warm='ONE unchanged true 1.4s physical teacher warmup records actual float64 currents at .2+i*.001 for 1200 input windows. Replay each candidate from resting network for two .5ms steps per current; reconstruct all history/rates and every neural state field. Exact baseline all cached fields and spikes required before search. RH not applied during warm.',optimizer=dict(name='bounded adaptive diagonal evolution strategy (not CMA-ES)',seed=SEED,generations=3,offspring=8,initial_sd=.35,bounds=[-1,1],bound_method='reflection',elite_weights=[.4,.3,.2,.1],mean_update=.5,variance_update=.3,variance_bounds=[.05**2,.6**2],penalty='.05 * mean(x**2)',rank='guard-feasible first; infeasible guard_violation first; then descending full-D pass count on four train states, worst train J + penalty, mean train J'),guard_rule='1.3 J <= baseline and foot mean/COM min/first positive nonfoot >= baseline, 1e-9 tolerance; None treated duration right-censored',dev_selection='baseline plus one train-ranked finalist per generation only. Eligible requires guard and dev D passes >= baseline, dev worst J <= baseline worst J and mean J <= baseline mean J (1e-9). Rank dev D passes descending, worst J+penalty, mean J; baseline remains eligible. Freeze selection before final.',controls='Selected repeat, all motor cut, sensory cut at1.3. Conditional selected nonbaseline only: independent per-edge multiplicative lognormal sigma .001/.01 each seeds1/2/3, total multiplier clipped [.9,1.1] relative original baseline, clipped fraction reported. Six candidate-specific warm replays and four train+guard releases each; compare selected same states. Engineering sensitivity, not statistical robustness proof.',extended='Only nonbaseline selected and all final + perturbation D gates pass: paired baseline/selected .5s guard then1s guard only if selected .5s passes.',budget=dict(search_candidates=24,baseline=1,search_releases=125,dev_releases=16,controls=3,conditional_perturbations=30,conditional_final=8,conditional_extended=4,max_releases=186,max_physical_seconds=40.8,physical_teacher_runs=1,teacher_seconds=1.4,main_replay_networks=25,conditional_perturbation_replay_networks=6,neural_steps_per_replay=2400),stop='abort hash/timing/state/numerical invariant failures; retain all failures; no retries expanding search budget',units='Absolute force/SI and intent NOT_ESTABLISHED')
    dump(O/'PREREGISTRATION.json',cfg)
    paths=list(dict.fromkeys(rt.files()+[AUD,ARCH/'DATA/CACHE_IDENTITY.json',Path(js(ARCH/'DATA/CACHE_IDENTITY.json')['file']),ARCH/'DATA/train_offset_0/TEACHER.npz',ARCH/'DATA/train_offset_0/BODY.npz',SRC/'run.py',O/'PREREGISTRATION.json',O/'GRAPH_GROUPS.npz',O/'GROUPING.json']))
    paths+=list(SRC.glob('*.py'))
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in dict.fromkeys(paths)})
    dump(O/'CHANGE_LEDGER.json',dict(old=[dict(path=str(p),sha256=sha(p)) for p in paths if p not in list(SRC.glob('*.py')) and p.parent!=O],new_source=[dict(path=str(p),sha256=sha(p),reason='bounded fixed-topology strength fitting with candidate-specific warm state') for p in SRC.glob('*.py')]))
    print('PREREGISTERED',info['edges'],info['group_counts'],flush=True)

def exact(a,b):
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if sparse.issparse(a):return sparse.issparse(b) and a.shape==b.shape and np.array_equal(a.indptr,b.indptr) and np.array_equal(a.indices,b.indices) and np.array_equal(a.data,b.data)
    if isinstance(a,dict):return set(a)==set(b) and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return bool(a==b)

def checkhash():
    before=js(O/'INPUT_HASHES_BEFORE.json');after={p:sha(p) for p in before};assert before==after,'frozen source/input hash mismatch';return after

def prepare():
    checkhash();t=RecordingRuntime();cleanup();t.true_currents=[];t.true_times=[];t.recording=True
    t.warmup('true_teacher',capture=CAPTURE);t.recording=False
    cur=np.asarray(t.true_currents);times=np.asarray(t.true_times);assert cur.shape==(1200,12000) and cur.dtype==np.float64
    assert np.max(abs(times-(.2+np.arange(1200)*.001)))<1e-11
    np.savez_compressed(O/'TRUE_CURRENTS.npz',currents=cur,time=times)
    with (O/'DATA/PHYSICAL_CACHE.pkl').open('wb') as f:pickle.dump(t.cache,f,protocol=5)
    ident=js(ARCH/'DATA/CACHE_IDENTITY.json');assert sha(ident['file'])==ident['sha256']
    with Path(ident['file']).open('rb') as f:original=pickle.load(f)
    checks=[]
    for ph,c in original.items():
        fresh=t.cache[ph];assert exact(vars(c['net']),vars(fresh['net'])),f'old cache neural mismatch {ph}'
        for k in ['rates','history','integration','assist']:assert exact(c[k],fresh[k]),f'old cache {k} mismatch {ph}'
        checks.append(dict(phase=ph,all_neural_fields_exact=True,rates_history_physics_exact=True))
    old=arrays(ARCH/'DATA/train_offset_0/TEACHER.npz');new=arrays(O/'DATA/true_teacher/TEACHER.npz')
    assert np.array_equal(old['spikes204'],new['spikes204'])
    aud=arrays(AUD);diff=cur-aud['currents'][:1200].astype(float)
    dump(O/'WARM_PHYSICAL_VALIDATION.json',dict(passed=True,cache_checks=checks,teacher_spikes_exact=True,actual_float64=True,timing_max_error=float(np.max(abs(times-(.2+np.arange(1200)*.001)))),old_audit_current_max_abs_difference=float(np.max(abs(diff))),old_audit_current_rms_difference=float(np.sqrt(np.mean(diff**2))),old_currents_used_for_candidate_warm=False,reason='Archived audit has float32 and recomputed contact state; exact true teacher currents captured anew'))
    dump(O/'WARM_DATA_HASHES.json',{str(p):sha(p) for p in [O/'TRUE_CURRENTS.npz',O/'DATA/PHYSICAL_CACHE.pkl',O/'DATA/true_teacher/TEACHER.npz',O/'DATA/true_teacher/BODY.npz',O/'DATA/true_teacher/NATIVE.npz']})
    print('PREPARED EXACT PHYSICAL BASELINE',len(checks),flush=True)

def guard(r,base):
    p,b=r['parts'],base['parts'];first=lambda d:d['first_positive_nonfoot_s'] if d['first_positive_nonfoot_s'] is not None else .2
    differences=[r['J']-base['J'],b['foot_support_mean']-p['foot_support_mean'],b['com_z_min']-p['com_z_min'],first(base)-first(r)]
    return bool(max(differences)<=1e-9),float(sum(max(0,v) for v in differences))

def run():
    checkhash();assert all(sha(p)==h for p,h in js(O/'WARM_DATA_HASHES.json').items())
    t=RecordingRuntime();cleanup()
    with (O/'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:physical=pickle.load(f)
    currents=arrays(O/'TRUE_CURRENTS.npz')['currents'];teacher=arrays(O/'DATA/true_teacher/TEACHER.npz')['spikes204']
    mats,rows,groups,_,_=graph();assert np.array_equal(groups,arrays(O/'GRAPH_GROUPS.npz')['groups'])
    cache_steps={round((ph-.2)*2000):ph for ph in CAPTURE};all_results=[];configs={};candidate_metrics=[]
    actor=lambda x,stochastic:(np.zeros(42,np.float32),0.,0.)
    def warm(name,x,perturb=None):
        assert np.shape(x)==(20,) and np.max(abs(x))<=1
        folder=O/'CONFIGS'/name;folder.mkdir(parents=True,exist_ok=False);scaled=[];assertions=[]
        rng=np.random.default_rng(perturb['seed']) if perturb else None
        for i,m in enumerate(mats):
            n=m.copy();factor=1+.1*x[groups[rows[i]]+10*i];clipped_fraction=0.
            if perturb:
                raw=factor*np.exp(rng.normal(0,perturb['sigma'],m.nnz));factor=np.clip(raw,.9,1.1);clipped_fraction=float(np.mean(raw!=factor))
            n.data*=factor
            assert np.array_equal(n.indptr,m.indptr) and np.array_equal(n.indices,m.indices) and np.all(n.data>0)
            ratio=n.data/m.data;assert ratio.min()>=.9-1e-14 and ratio.max()<=1.1+1e-14
            p=folder/('We.npz' if i==0 else 'Wi.npz');sparse.save_npz(p,n);scaled.append(n)
            release=n.copy();baseline=m.copy()
            if i==0:
                mask=np.isin(rows[i],t.rh);release.data[mask]*=.375;baseline.data[mask]*=.375
            rr=release.data/baseline.data;assert rr.min()>=.9-1e-14 and rr.max()<=1.1+1e-14
            assertions.append(dict(matrix=str(p),sha256=sha(p),baseline=str(rt.b.AB/'A'/('ACTUAL_We.npz' if i==0 else 'ACTUAL_Wi.npz')),baseline_sha256=sha(rt.b.AB/'A'/('ACTUAL_We.npz' if i==0 else 'ACTUAL_Wi.npz')),edges=m.nnz,warm_ratio_min=float(ratio.min()),warm_ratio_max=float(ratio.max()),release_ratio_min=float(rr.min()),release_ratio_max=float(rr.max()),indices_indptr_exact=True,sign_preserved=True,clipped_fraction=clipped_fraction))
        config=folder/'PARAMETERS.json';dump(config,dict(name=name,x=x,multipliers=1+.1*x,penalty=.05*float(np.mean(x*x)),edge_assertions=assertions,RH_release_only=.375,perturbation=perturb))
        net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=.5);net.set_connectivity(*scaled)
        t.history.reset();rates=np.zeros(66);decay=np.exp(-.0005/.005);cache={};spikes=[];checks=[]
        for k,current in enumerate(currents):
            for j in range(2):
                sp=net.step(current_nA=current);spikes.append(sp[t.cells].copy());t.history.consume(sp)
                rates=decay*rates+(1-decay)*np.asarray(t.poolmat@sp,float)/.0005
                count=k*2+j+1
                if count in cache_steps:
                    ph=cache_steps[count];c=dict(physical[ph]);c.update(net=copy.deepcopy(net),rates=rates.copy(),history=t.history.state_dict());cache[ph]=c
                    if name=='baseline':
                        assert exact(vars(net),vars(physical[ph]['net'])),f'baseline replay neural mismatch {ph}'
                        assert exact(rates,c['rates']) and exact(rates,physical[ph]['rates']) and exact(c['history'],physical[ph]['history'])
                        checks.append(dict(phase=ph,steps=count,all_net_fields_exact=True,rates_history_exact=True))
        spikes=np.asarray(spikes)
        if name=='baseline':assert np.array_equal(spikes,teacher)
        np.savez_compressed(folder/'WARM_SPIKES.npz',spikes204=spikes,cells=t.cells)
        with (folder/'WARM_CACHE.pkl').open('wb') as f:pickle.dump(cache,f,protocol=5)
        dump(folder/'WARM_IDENTITY.json',dict(file=str(folder/'WARM_CACHE.pkl'),sha256=sha(folder/'WARM_CACHE.pkl'),current_source=str(O/'TRUE_CURRENTS.npz'),current_sha256=sha(O/'TRUE_CURRENTS.npz'),candidate_parameters_sha256=sha(config),neural_steps=2400,baseline_exact_checks=checks))
        configs[name]=dict(file=config,x=np.array(x),cache_file=folder/'WARM_CACHE.pkl');t.cache=cache
        print('WARM REPLAY',name,flush=True);return cache
    def load(name):
        with configs[name]['cache_file'].open('rb') as f:t.cache=pickle.load(f)
    def evaluate(name,ph,label,**kw):
        r,_=t.episode(label,ph,actor,configs[name]['file'],**kw);r['candidate']=name
        native=arrays(Path(r['folder'])/'NATIVE.npz')['rows'];positive=np.flatnonzero(native[:,7]>1e-9)
        r['first_positive_nonfoot_s']=float(native[positive[0],0]-ph) if len(positive) else None
        dump(Path(r['folder'])/'RESULT.json',r);all_results.append(r);return r
    def metrics(name,results,base):
        gr=next(r for r in results if r['release_time_s']==GUARD);ok,violation=guard(gr,base)
        tr=[r for r in results if r['release_time_s'] in TRAIN];pen=.05*float(np.mean(configs[name]['x']**2))
        return dict(candidate=name,guard=ok,guard_violation=violation,passes=sum(r['validation']['passed'] for r in tr),worst_J=max(r['J'] for r in tr),mean_J=float(np.mean([r['J'] for r in tr])),penalty=pen,fitness=max(r['J'] for r in tr)+pen)
    key=lambda m:(not m['guard'],m['guard_violation'] if not m['guard'] else 0,-m['passes'],m['fitness'],m['mean_J'])
    warm('baseline',np.zeros(20));base_results=[evaluate('baseline',ph,f'baseline_train_guard_{ph}') for ph in TRAIN+[GUARD]]
    baseguard=base_results[-1];assert abs(baseguard['J']-1.5903704331878943)<1e-9
    baseline_metric=metrics('baseline',base_results,baseguard);candidate_metrics.append(baseline_metric)
    base_dev=[evaluate('baseline',ph,f'baseline_dev_{ph}') for ph in DEV];dev_rows={'baseline':base_dev};finalists=[];evo=Evolution();generations=[]
    for gen in range(3):
        population=evo.ask();generation=dict(generation=gen,optimizer_before=evo.state(),population=population)
        dump(O/f'GENERATION_{gen}_BEFORE.json',generation);generation_metrics=[]
        for i,x in enumerate(population):
            name=f'g{gen}_c{i}';warm(name,x)
            rr=[evaluate(name,ph,f'{name}_train_guard_{ph}') for ph in TRAIN+[GUARD]];m=metrics(name,rr,baseguard);candidate_metrics.append(m);generation_metrics.append(m)
            dump(O/'CANDIDATE_METRICS.json',candidate_metrics)
        order=sorted(range(8),key=lambda i:key(generation_metrics[i]));best=generation_metrics[order[0]]['candidate'];finalists.append(best);load(best)
        dev_rows[best]=[evaluate(best,ph,f'{best}_dev_{ph}') for ph in DEV]
        evo.tell(population,order);generation.update(ranking=order,metrics=generation_metrics,finalist=best,optimizer_after=evo.state());generations.append(generation);dump(O/f'GENERATION_{gen}_AFTER.json',generation)
    dev_metrics=[]
    for name,rr in dev_rows.items():
        cm=next(m for m in candidate_metrics if m['candidate']==name);dev_metrics.append(dict(candidate=name,guard=cm['guard'],passes=sum(r['validation']['passed'] for r in rr),worst_J=max(r['J'] for r in rr),mean_J=float(np.mean([r['J'] for r in rr])),penalty=cm['penalty'],fitness=max(r['J'] for r in rr)+cm['penalty']))
    bm=dev_metrics[0];eligible=[m for m in dev_metrics if m['guard'] and m['passes']>=bm['passes'] and m['worst_J']<=bm['worst_J']+1e-9 and m['mean_J']<=bm['mean_J']+1e-9]
    selected=min(eligible,key=lambda m:(-m['passes'],m['fitness'],m['mean_J']))['candidate']
    selection=dict(selected=selected,dev_metrics=dev_metrics,finalists=finalists,final_scores_not_accessed=True,rule=js(O/'PREREGISTRATION.json')['dev_selection']);dump(O/'SELECTION.json',selection);selection_hash=sha(O/'SELECTION.json')
    final_results=[]
    if selected!='baseline':
        for name in ['baseline',selected]:
            load(name)
            for ph in FINAL:final_results.append(evaluate(name,ph,f'{name}_final_{ph}'))
    load(selected);evaluate(selected,GUARD,'selected_repeat');evaluate(selected,GUARD,'selected_motor_cut',cut=True);evaluate(selected,GUARD,'selected_sensory_cut',sensory_cut=True)
    pert_results=[]
    if selected!='baseline':
        for sigma in [.001,.01]:
            for seed in [1,2,3]:
                name=f'perturb_sigma{sigma}_seed{seed}';warm(name,configs[selected]['x'],dict(sigma=sigma,seed=seed,kind='per-edge multiplicative lognormal clipped relative original baseline'))
                for ph in TRAIN+[GUARD]:pert_results.append(evaluate(name,ph,f'{name}_{ph}'))
    extended=[]
    if selected!='baseline' and all(r['validation']['passed'] for r in [v for v in final_results if v['candidate']==selected]+pert_results):
        for duration in [.5,1.]:
            pair=[]
            for name in ['baseline',selected]:
                load(name);pair.append(evaluate(name,GUARD,f'{name}_extended_{duration}',duration=duration))
            extended.extend(pair)
            if not pair[-1]['validation']['passed']:break
    assert sha(O/'SELECTION.json')==selection_hash and len(all_results)<=186
    summary=dict(selected=selected,selection=selection,candidate_metrics=candidate_metrics,generations=generations,results=all_results,physical_releases=len(all_results),physical_seconds=1.4+sum(r['duration_s'] for r in all_results),teacher_runs=1,neural_replay_networks=len(configs),neural_replay_steps=2400*len(configs),final_skipped=selected=='baseline',extended_runs=len(extended),all_D_passes=sum(r['validation']['passed'] for r in all_results),walking_certified=False,biological_validity='NOT_ESTABLISHED',intent='NOT_ESTABLISHED')
    dump(O/'SUMMARY.json',summary);dump(O/'INPUT_HASHES_AFTER.json',checkhash());print('DONE',selected,len(all_results),flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    try:{'prereg':prereg,'prepare':prepare,'run':run}[action]()
    except BaseException:
        import traceback
        if O.exists():dump(O/('FAILURE_'+action+'_'+str(time.time_ns())+'.json'),dict(action=action,traceback=traceback.format_exc()))
        raise
