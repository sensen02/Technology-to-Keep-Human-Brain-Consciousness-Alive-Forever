"""S5 bounded offline optimizer comparison. Frozen physical helpers imported normally."""
import os,sys,copy,pickle,time,importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
SRC=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD_SRC=ROOT/'experiments/s5-strength-fit/20261106T051337Z'
OLD=Path('/home/sensen/Desktop/engine修复验收/s5-strength-fit/20261106T051337Z')
O=Path('/home/sensen/Desktop/engine修复验收/s5-optimizer-comparison/20261106T053754Z')
spec=importlib.util.spec_from_file_location('frozen_strength_helper',OLD_SRC/'run.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
s.O=O;s.rt.O=O;s.rt.b.O=O
rt=s.rt;js,sha,dump=s.js,s.sha,s.dump;arrays=s.arrays;TRAIN=s.TRAIN;GUARD=s.GUARD;DEV=s.DEV;FINAL=s.FINAL
from optimizers import CLASSES,SEED,selftest

class Experiment:
    def __init__(self):
        self.t=s.RecordingRuntime();s.cleanup()
        with (OLD/'DATA/PHYSICAL_CACHE.pkl').open('rb') as f:self.physical=pickle.load(f)
        self.currents=arrays(OLD/'TRUE_CURRENTS.npz')['currents'];self.teacher=arrays(OLD/'DATA/true_teacher/TEACHER.npz')['spikes204']
        assert self.currents.dtype==np.float64 and self.currents.shape==(1200,12000)
        self.mats,self.rows,self.groups,_,_=s.graph();assert np.array_equal(self.groups,arrays(OLD/'GRAPH_GROUPS.npz')['groups'])
        self.steps={round((ph-.2)*2000):ph for ph in self.physical};self.configs={};self.results=[];self.metrics=[];self.actor=lambda x,stochastic:(np.zeros(42,np.float32),0.,0.)
    def warm(self,name,x,perturb=None):
        x=np.asarray(x);assert x.shape==(20,) and np.max(abs(x))<=1
        folder=O/'CONFIGS'/name;folder.mkdir(parents=True,exist_ok=False);scaled=[];edges=[];rng=np.random.default_rng(perturb['seed']) if perturb else None
        for i,m in enumerate(self.mats):
            factor=1+.1*x[self.groups[self.rows[i]]+10*i];clipped=0.
            if perturb:
                raw=factor*np.exp(rng.normal(0,perturb['sigma'],m.nnz));factor=np.clip(raw,.9,1.1);clipped=float(np.mean(raw!=factor))
            n=m.copy();n.data*=factor;ratio=n.data/m.data
            assert np.array_equal(n.indptr,m.indptr) and np.array_equal(n.indices,m.indices) and np.all(n.data>0) and ratio.min()>=.9-1e-14 and ratio.max()<=1.1+1e-14
            p=folder/('We.npz' if i==0 else 'Wi.npz');sparse.save_npz(p,n);scaled.append(n)
            edges.append(dict(path=str(p),sha256=sha(p),baseline=str(rt.b.AB/'A'/('ACTUAL_We.npz' if i==0 else 'ACTUAL_Wi.npz')),min=float(ratio.min()),max=float(ratio.max()),clipped_fraction=clipped))
        cfg=folder/'PARAMETERS.json';dump(cfg,dict(name=name,x=x,multipliers=1+.1*x,penalty=.05*float(np.mean(x*x)),edge_assertions=edges,RH_release_only=.375,perturbation=perturb))
        net=s.ConductanceNetwork(12000,s.ConductanceParams(),dt_ms=.5);net.set_connectivity(*scaled)
        self.t.history.reset();rates=np.zeros(66);decay=np.exp(-.0005/.005);cache={};spikes=[];checks=[]
        for k,current in enumerate(self.currents):
            for j in range(2):
                sp=net.step(current_nA=current);spikes.append(sp[self.t.cells].copy());self.t.history.consume(sp)
                rates=decay*rates+(1-decay)*np.asarray(self.t.poolmat@sp,float)/.0005;count=k*2+j+1
                if count in self.steps:
                    ph=self.steps[count];c=dict(self.physical[ph]);c.update(net=copy.deepcopy(net),rates=rates.copy(),history=self.t.history.state_dict());cache[ph]=c
                    if name=='baseline':
                        assert s.exact(vars(net),vars(self.physical[ph]['net'])) and s.exact(rates,self.physical[ph]['rates']) and s.exact(c['history'],self.physical[ph]['history'])
                        checks.append(dict(phase=ph,all_net_fields_exact=True,rates_history_exact=True))
        if name=='baseline':assert np.array_equal(np.asarray(spikes),self.teacher)
        np.savez_compressed(folder/'WARM_SPIKES.npz',spikes204=np.asarray(spikes),cells=self.t.cells)
        with (folder/'WARM_CACHE.pkl').open('wb') as f:pickle.dump(cache,f,protocol=5)
        dump(folder/'WARM_IDENTITY.json',dict(file=str(folder/'WARM_CACHE.pkl'),sha256=sha(folder/'WARM_CACHE.pkl'),current_source=str(OLD/'TRUE_CURRENTS.npz'),current_sha256=sha(OLD/'TRUE_CURRENTS.npz'),candidate_parameters_sha256=sha(cfg),neural_steps=2400,baseline_exact_checks=checks))
        self.configs[name]=dict(file=cfg,x=x.copy(),cache_file=folder/'WARM_CACHE.pkl');self.t.cache=cache
        print('WARM',name,flush=True)
    def load(self,name):
        with self.configs[name]['cache_file'].open('rb') as f:self.t.cache=pickle.load(f)
    def evaluate(self,name,ph,label,**kw):
        r,_=self.t.episode(label,ph,self.actor,self.configs[name]['file'],**kw);r['candidate']=name
        native=arrays(Path(r['folder'])/'NATIVE.npz')['rows'];positive=np.flatnonzero(native[:,7]>1e-9);r['first_positive_nonfoot_s']=float(native[positive[0],0]-ph) if len(positive) else None
        dump(Path(r['folder'])/'RESULT.json',r);self.results.append(r);return r
    def metric(self,name,rr,baseline):
        gr=next(r for r in rr if r['release_time_s']==GUARD);ok,violation=s.guard(gr,baseline);tr=[r for r in rr if r['release_time_s'] in TRAIN];pen=.05*float(np.mean(self.configs[name]['x']**2))
        return dict(candidate=name,guard=ok,guard_violation=violation,passes=sum(r['validation']['passed'] for r in tr),worst_J=max(r['J'] for r in tr),mean_J=float(np.mean([r['J'] for r in tr])),penalty=pen,fitness=max(r['J'] for r in tr)+pen)

def key(m):return (not m['guard'],m['guard_violation'] if not m['guard'] else 0,-m['passes'],m['fitness'],m['mean_J'])
def checkhash():
    before=js(O/'INPUT_HASHES_BEFORE.json');after={p:sha(p) for p in before};assert before==after;return after

def prereg():
    O.mkdir(parents=True,exist_ok=True);assert not (O/'PREREGISTRATION.json').exists()
    dump(O/'OPTIMIZER_SELFTEST.json',selftest())
    cfg=dict(stage='S5 offline strength optimizer comparison; engineering fit, not biological calibration',arms=['diagonal','cma','de'],seed=SEED,dimensions=20,initial_normalized_sd=.05,initial_strength_sd=.005,population=8,batches=3,candidates_per_arm=24,bounds=[-1,1],strength_multiplier='1+.1*x',same_graph=str(OLD/'GRAPH_GROUPS.npz'),train=TRAIN,guard=GUARD,dev=DEV,final=FINAL,immutable='topology/sign/electrical params/body/sensory/5ms readout; RH .375 release only; no body/time controller',warm='reuse archived actual float64 currents and exact 17 physical states read-only; candidate-specific full neural replay; exact baseline all fields/spikes prerequisite; zero new teacher',algorithms=dict(diagonal='previous ES formula unchanged except initial variance .05**2; min SD .05 retained',cma='pycma 4.5.0 active full covariance, native sampling/bound transform, rank objective; full state transcript',de='explicit DE/rand/1/bin NP8 F.5 CR.9 deferred pairwise selection, reflected donors; first eight paid initialization, two paid trial generations'),initialization='same normalized marginal SD .05 and seed; ES native antithetic; DE native iid Gaussian; CMA native Gaussian with tiny library axis scaling; samples differ, same nominal scale only. DE population8 in20D starts in affine rank at most7, two trial generations only; CMA three updates only; bounded algorithm-package pilot, not tuned optimizer comparison',objective='guard-feasible first, then guard violation if infeasible, train D pass descending, worst J+.05mean(x^2), mean J. Same objective for every arm.',dev_selection='one train-ranked evaluated batch representative per arm per8; baseline +9 representatives eligible only guard and dev D nondecrease and worst/mean J nonincrease within1e-9; rank D descending,worst J+penalty,mean J; ties retain baseline by insertion order',DE_representative='best newly evaluated batch point, even if trial rejected from DE population; preserves equal dev access to all evaluated candidates',guard_rule=js(OLD/'PREREGISTRATION.json')['guard_rule'],conditional='only selected nonbaseline gets final4 paired8 and six edge perturbations x5=30; only selected final+perturb complete D pass allows .5/1s paired extension4',budget=dict(shared_baseline=9,objective=360,dev=36,controls=3,main_max=408,conditional_final=8,conditional_perturb=30,conditional_extension=4,total_max=450),final_withheld='17 states generated in previous S; four final states never release-scored there; withheld until selection here, not newly generated or never-seen physical records',historical_reference='previous .35 initial SD diagonal ES 24 candidates context only; not matched scale; one seed cannot establish randomized significance',walking_certified=False)
    dump(O/'PREREGISTRATION.json',cfg)
    historical=js(OLD/'MANIFEST_SHA256.json');dump(O/'HISTORICAL_MANIFEST_IDENTITY.json',dict(path=str(OLD/'MANIFEST_SHA256.json'),sha256=sha(OLD/'MANIFEST_SHA256.json')))
    paths=list(dict.fromkeys(rt.files()+list(OLD_SRC.glob('*.py'))+list(SRC.glob('*.py'))+[OLD/'TRUE_CURRENTS.npz',OLD/'DATA/PHYSICAL_CACHE.pkl',OLD/'DATA/true_teacher/TEACHER.npz',OLD/'DATA/true_teacher/BODY.npz',OLD/'GRAPH_GROUPS.npz',OLD/'PREREGISTRATION.json',OLD/'SUMMARY.json',O/'PREREGISTRATION.json',SRC/'.venv/lib/python3.12/site-packages/readonly_dependencies.pth']))
    import cma
    paths+=list(Path(cma.__file__).parent.rglob('*.py'))
    paths+=[ROOT/'run_integrated_repaired_closure.py',SRC/'.venv/pyvenv.cfg',Path(cma.__file__).parent.parent/'cma-4.5.0.dist-info/METADATA']
    dump(O/'INPUT_HASHES_BEFORE.json',{str(p):sha(p) for p in paths});dump(O/'CHANGE_LEDGER.json',dict(reason='matched small-scale offline algorithm comparison; no original inputs modified',new_source=[dict(path=str(p),sha256=sha(p)) for p in SRC.glob('*.py')],original={str(p):sha(p) for p in paths if p.is_relative_to(OLD) or p.is_relative_to(OLD_SRC)},environment='first system python3.14 venv failed missing ensurepip before simulation; new partial environment recreated using existing3.12 interpreter; cma4.5.0 installed only in new isolated environment; old site packages read-only .pth'))
    print('PREREGISTERED',flush=True)

def run():
    checkhash()
    assert all(sha(p)==h for p,h in js(OLD/'WARM_DATA_HASHES.json').items())
    ex=Experiment();ex.warm('baseline',np.zeros(20));rr=[ex.evaluate('baseline',ph,f'baseline_train_guard_{ph}') for ph in TRAIN+[GUARD]];bg=rr[-1];assert abs(bg['J']-1.5903704331878943)<1e-9
    ex.metrics.append(ex.metric('baseline',rr,bg));dev={'baseline':[ex.evaluate('baseline',ph,f'baseline_dev_{ph}') for ph in DEV]};finalists=[];batches=[]
    for arm,cls in CLASSES.items():
        opt=cls()
        for batch in range(3):
            before=opt.state();pop=opt.ask();record=dict(arm=arm,batch=batch,before=before,population=pop,proposal_state=opt.state());dump(O/f'{arm}_BATCH_{batch}_BEFORE.json',record);mm=[]
            for i,x in enumerate(pop):
                name=f'{arm}_g{batch}_c{i}';ex.warm(name,x);rr=[ex.evaluate(name,ph,f'{name}_train_guard_{ph}') for ph in TRAIN+[GUARD]];m=ex.metric(name,rr,bg);mm.append(m);ex.metrics.append(m);dump(O/'CANDIDATE_METRICS.json',ex.metrics)
            order=sorted(range(8),key=lambda i:key(mm[i]));best=mm[order[0]]['candidate'];finalists.append(best);ex.load(best);dev[best]=[ex.evaluate(best,ph,f'{best}_dev_{ph}') for ph in DEV]
            opt.tell(pop,[key(m) for m in mm]);record.update(ranking=order,metrics=mm,finalist=best,after=opt.state());dump(O/f'{arm}_BATCH_{batch}_AFTER.json',record);batches.append(record)
            print('BATCH DONE',arm,batch,'guard passes',sum(m['guard'] for m in mm),'best',best,flush=True)
    dm=[]
    for name,rr in dev.items():
        cm=next(m for m in ex.metrics if m['candidate']==name);dm.append(dict(candidate=name,guard=cm['guard'],passes=sum(r['validation']['passed'] for r in rr),worst_J=max(r['J'] for r in rr),mean_J=float(np.mean([r['J'] for r in rr])),penalty=cm['penalty'],fitness=max(r['J'] for r in rr)+cm['penalty']))
    base=dm[0];eligible=[m for m in dm if m['guard'] and m['passes']>=base['passes'] and m['worst_J']<=base['worst_J']+1e-9 and m['mean_J']<=base['mean_J']+1e-9];selected=min(eligible,key=lambda m:(-m['passes'],m['fitness'],m['mean_J']))['candidate']
    selection=dict(selected=selected,dev_metrics=dm,finalists=finalists,final_scores_not_accessed=True);dump(O/'SELECTION.json',selection);selhash=sha(O/'SELECTION.json');final=[];pert=[];extended=[]
    if selected!='baseline':
        for name in ['baseline',selected]:
            ex.load(name)
            for ph in FINAL:final.append(ex.evaluate(name,ph,f'{name}_final_{ph}'))
    ex.load(selected);ex.evaluate(selected,GUARD,'selected_repeat');ex.evaluate(selected,GUARD,'selected_motor_cut',cut=True);ex.evaluate(selected,GUARD,'selected_sensory_cut',sensory_cut=True)
    if selected!='baseline':
        for sigma in [.001,.01]:
            for seed in [1,2,3]:
                name=f'perturb_sigma{sigma}_seed{seed}';ex.warm(name,ex.configs[selected]['x'],dict(sigma=sigma,seed=seed));pert.extend(ex.evaluate(name,ph,f'{name}_{ph}') for ph in TRAIN+[GUARD])
    if selected!='baseline' and all(r['validation']['passed'] for r in [v for v in final if v['candidate']==selected]+pert):
        for duration in [.5,1.]:
            pair=[]
            for name in ['baseline',selected]:
                ex.load(name);pair.append(ex.evaluate(name,GUARD,f'{name}_extended_{duration}',duration=duration))
            extended.extend(pair)
            if not pair[-1]['validation']['passed']:break
    assert sha(O/'SELECTION.json')==selhash and len(ex.results)<=450
    dump(O/'SUMMARY.json',dict(selected=selected,selection=selection,candidate_metrics=ex.metrics,batches=batches,results=ex.results,physical_releases=len(ex.results),physical_seconds=sum(r['duration_s'] for r in ex.results),teacher_runs=0,neural_replay_networks=len(ex.configs),neural_replay_steps=2400*len(ex.configs),final_skipped=selected=='baseline',extended_runs=len(extended),all_D_passes=sum(r['validation']['passed'] for r in ex.results),walking_certified=False,biological_validity='NOT_ESTABLISHED',intent='NOT_ESTABLISHED'))
    dump(O/'INPUT_HASHES_AFTER.json',checkhash());print('DONE',selected,len(ex.results),flush=True)

if __name__=='__main__':
    action=sys.argv[1]
    try:{'prereg':prereg,'run':run}[action]()
    except BaseException:
        import traceback
        if O.exists():dump(O/f'FAILURE_{action}_{time.time_ns()}.json',dict(action=action,traceback=traceback.format_exc()))
        raise
