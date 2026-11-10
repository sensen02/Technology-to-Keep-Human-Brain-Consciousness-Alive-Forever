"""Parent independent optimizer transcript audit; no imports of experiment optimizers.
Reads only completed search summaries; writes one new verification artifact.
"""
from pathlib import Path
import json
import numpy as np
import cma
O=Path('/home/sensen/Desktop/engine修复验收/s5-optimizer-comparison/20261106T053754Z')

def js(p):return json.loads(Path(p).read_text())
def reflect(x):return 1-np.abs((np.asarray(x)+1)%4-2)
def key(m):return (not m['guard'],m['guard_violation'] if not m['guard'] else 0,-m['passes'],m['fitness'],m['mean_J'])
def check_population(arm,batch,pop):
    log=js(O/f'{arm}_BATCH_{batch}_AFTER.json')
    assert np.array_equal(pop,np.asarray(log['population']))
    assert np.max(abs(pop))<=1
    keys=[key(m) for m in log['metrics']]
    order=sorted(range(8),key=lambda i:keys[i]);assert order==log['ranking']
    assert log['finalist']==log['metrics'][order[0]]['candidate']
    return log,keys,order

def main():
    summary=js(O/'SUMMARY.json');rows=[]
    rng=np.random.default_rng(20261106);mean=np.zeros(20);var=np.full(20,.05**2)
    for b in range(3):
        z=rng.normal(size=(4,20));pop=reflect(mean+np.sqrt(var)*np.concatenate([z,-z]));log,keys,order=check_population('diagonal',b,pop)
        assert np.array_equal(mean,np.asarray(log['before']['mean'])) and np.array_equal(var,np.asarray(log['before']['variance']))
        elite=pop[order[:4]];w=np.array([.4,.3,.2,.1]);spread=w@((elite-mean)**2)
        mean=.5*mean+.5*(w@elite);var=np.clip(.7*var+.3*spread,.05**2,.6**2)
        assert np.array_equal(mean,np.asarray(log['after']['mean'])) and np.array_equal(var,np.asarray(log['after']['variance']))
        rows.append(dict(arm='diagonal',batch=b,independent_formula=True))
    rng=np.random.default_rng(20261106);pop=None;retained_keys=None
    for b in range(3):
        donors_log=[]
        if b==0:trial=reflect(.05*rng.normal(size=(8,20)))
        else:
            trial=[]
            for i in range(8):
                donors=rng.choice([j for j in range(8) if j!=i],3,replace=False);a,bb,c=donors
                mutant=reflect(pop[a]+.5*(pop[bb]-pop[c]));mask=rng.random(20)<.9;j=int(rng.integers(20));mask[j]=True
                trial.append(np.where(mask,mutant,pop[i]));donors_log.append(dict(target=i,donors=donors.tolist(),mask=mask.tolist(),jrand=j))
            trial=np.asarray(trial)
        log,trial_keys,order=check_population('de',b,trial)
        assert log['proposal_state']['donor_transcript']==donors_log
        if b==0:pop=trial.copy();retained_keys=list(trial_keys)
        else:
            assert np.array_equal(pop,np.asarray(log['before']['population']))
            assert [list(k) for k in retained_keys]==log['before']['keys']
            accept=[trial_keys[i]<retained_keys[i] for i in range(8)]
            pop=np.where(np.asarray(accept)[:,None],trial,pop)
            retained_keys=[trial_keys[i] if accept[i] else retained_keys[i] for i in range(8)]
        assert np.array_equal(pop,np.asarray(log['after']['population']))
        assert [list(k) for k in retained_keys]==log['after']['keys']
        rows.append(dict(arm='de',batch=b,independent_deferred_formula=True))
    rng=np.random.RandomState(20261106)
    es=cma.CMAEvolutionStrategy(np.zeros(20),.05,dict(seed=20261106,randn=rng.randn,popsize=8,bounds=[-1,1],CMA_diagonal=0,verbose=-9,verb_log=0,CMA_active=True))
    for b in range(3):
        pop=np.asarray(es.ask());log,keys,order=check_population('cma',b,pop)
        ranks=np.empty(8);ranks[order]=np.arange(8);es.tell(pop.tolist(),ranks.tolist())
        assert np.array_equal(es.mean,np.asarray(log['after']['mean']))
        assert np.array_equal(es.sm.C,np.asarray(log['after']['covariance']))
        assert np.array_equal(es.pc,np.asarray(log['after']['pc']))
        assert np.array_equal(es.adapt_sigma.ps,np.asarray(log['after']['ps']))
        assert es.sigma==log['after']['sigma']
        eigenvalues=np.linalg.eigvalsh(es.sm.C)
        assert eigenvalues.min()>0
        rows.append(dict(arm='cma',batch=b,canonical_library_direct_replay=True,minimum_covariance_eigenvalue=float(eigenvalues.min()),offdiagonal_norm=float(np.linalg.norm(es.sm.C-np.diag(np.diag(es.sm.C)))),sigma=float(es.sigma)))
    assert len(summary['candidate_metrics'])==73
    output=dict(importance='重要辅助',passed=True,method='ES/DE independently expanded formulas and native CMA library direct invocation without experiment optimizer wrappers; checks immutable before/after snapshots and equal24candidate budget',cma_version=cma.__version__,batches=rows)
    p=O/'PARENT_OPTIMIZER_VERIFICATION.json';assert not p.exists()
    p.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False))
if __name__=='__main__':main()
