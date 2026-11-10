"""Offline bounded optimizers; no body/controller inputs. Rank-based objective."""
import copy
import numpy as np
import cma
SEED=20261106

def reflect(x):return 1-np.abs((np.asarray(x)+1)%4-2)

class Diagonal:
    def __init__(self,seed=SEED,n=20):
        self.rng=np.random.default_rng(seed);self.mean=np.zeros(n);self.var=np.full(n,.05**2)
    def ask(self):
        z=self.rng.normal(size=(4,len(self.mean)));return reflect(self.mean+np.sqrt(self.var)*np.concatenate([z,-z]))
    def tell(self,x,keys):
        order=sorted(range(8),key=lambda i:keys[i]);elite=x[order[:4]];w=np.array([.4,.3,.2,.1]);spread=w@((elite-self.mean)**2)
        self.mean=.5*self.mean+.5*(w@elite);self.var=np.clip(.7*self.var+.3*spread,.05**2,.6**2)
    def state(self):return dict(mean=self.mean.copy(),variance=self.var.copy())

class CMA:
    def __init__(self,seed=SEED,n=20):
        self.rng=np.random.RandomState(seed)
        self.es=cma.CMAEvolutionStrategy(np.zeros(n),.05,dict(seed=seed,randn=self.rng.randn,popsize=8,bounds=[-1,1],CMA_diagonal=0,verbose=-9,verb_log=0,CMA_active=True))
    def ask(self):return np.asarray(self.es.ask())
    def tell(self,x,keys):
        order=sorted(range(8),key=lambda i:keys[i]);ranks=np.empty(8);ranks[order]=np.arange(8);self.es.tell(x.tolist(),ranks.tolist())
    def state(self):return dict(mean=self.es.mean.copy(),sigma=self.es.sigma,covariance=self.es.sm.C.copy(),pc=self.es.pc.copy(),ps=getattr(self.es.adapt_sigma,'ps',np.zeros(len(self.es.mean))).copy(),countiter=self.es.countiter)

class DE:
    """DE/rand/1/bin, NP8, F.5, CR.9, deferred pairwise selection.
    Batch0 initializes eight paid points; batches1,2 each evaluate eight trials.
    Donors distinct and exclude target; forced jrand guarantees crossover.
    """
    def __init__(self,seed=SEED,n=20):
        self.rng=np.random.default_rng(seed);self.n=n;self.population=None;self.keys=None;self.proposals=[]
    def ask(self):
        if self.population is None:
            self.proposals=[];return reflect(.05*self.rng.normal(size=(8,self.n)))
        out=[];self.proposals=[]
        for i in range(8):
            donors=self.rng.choice([j for j in range(8) if j!=i],3,replace=False);a,b,c=donors
            mutant=reflect(self.population[a]+.5*(self.population[b]-self.population[c]));mask=self.rng.random(self.n)<.9;jrand=int(self.rng.integers(self.n));mask[jrand]=True
            out.append(np.where(mask,mutant,self.population[i]));self.proposals.append(dict(target=i,donors=donors.tolist(),mask=mask.tolist(),jrand=jrand))
        return np.asarray(out)
    def tell(self,x,keys):
        if self.population is None:self.population=x.copy();self.keys=list(keys)
        else:
            for i in range(8):
                if keys[i]<self.keys[i]:self.population[i]=x[i];self.keys[i]=keys[i]
    def state(self):return dict(population=None if self.population is None else self.population.copy(),keys=copy.deepcopy(self.keys),donor_transcript=copy.deepcopy(self.proposals))

CLASSES={'diagonal':Diagonal,'cma':CMA,'de':DE}

def selftest():
    result={}
    for name,cls in CLASSES.items():
        a=cls(n=6);b=cls(n=6);first=None;best=float('inf');target=np.full(6,.15)
        for batch in range(150):
            x=a.ask();y=b.ask();assert np.array_equal(x,y) and x.shape==(8,6) and np.max(abs(x))<=1
            values=np.sum((x-target)**2,axis=1);keys=[(float(v),) for v in values]
            if first is None:first=float(values.min())
            best=min(best,float(values.min()));a.tell(x,keys);b.tell(y,keys)
            if name=='de' and batch:
                for d in a.proposals:assert len(set(d['donors']))==3 and d['target'] not in d['donors'] and d['mask'][d['jrand']]
        assert best<first*.5,(name,first,best)
        if name=='cma':assert np.max(abs(a.es.sm.C-np.diag(np.diag(a.es.sm.C))))>1e-8
        result[name]=dict(initial_best=first,best=best,bounded=True,deterministic=True,analytic_improvement=True)
    q,_=np.linalg.qr(np.random.default_rng(91).normal(size=(6,6)));A=q@np.diag(np.geomspace(1,30,6))@q.T
    rotated={}
    for name,cls in CLASSES.items():
        opt=cls(n=6);best=float('inf');first=None;target=np.full(6,.10)
        for batch in range(200):
            x=opt.ask();delta=x-target;values=np.einsum('bi,ij,bj->b',delta,A,delta)
            if first is None:first=float(values.min())
            best=min(best,float(values.min()));opt.tell(x,[(float(v),) for v in values])
            assert np.max(abs(x))<=1
        assert best<first*.5,(name,first,best)
        if name=='cma':
            st=opt.state();assert np.linalg.eigvalsh(st['covariance']).min()>0 and np.linalg.norm(st['pc'])>0 and np.linalg.norm(st['ps'])>0 and st['sigma']>0
        rotated[name]=dict(initial_best=first,best=best)
    # Independently check one DE generation against the donor/mask transcript,
    # and all pairwise decisions against a frozen pre-update population.
    de=DE(n=6);initial=de.ask();keys=[(float(i),) for i in range(8)];de.tell(initial,keys);trial=de.ask();old=de.population.copy()
    for i,d in enumerate(de.proposals):
        a,b,c=d['donors'];expected=np.where(d['mask'],reflect(old[a]+.5*(old[b]-old[c])),old[i]);assert np.array_equal(expected,trial[i])
    newkeys=[(float(i-1 if i%2 else i),) for i in range(8)];de.tell(trial,newkeys)
    for i in range(8):assert np.array_equal(de.population[i],trial[i] if i%2 else old[i])
    return dict(passed=True,cma_version=cma.__version__,analytic='bounded6D shifted sphere1200 and rotated anisotropic quadratic1600 evaluations per arm; implementation checks only, no physics success implication',results=result,rotated=rotated,DE_deferred_formula_and_tie_retention=True)
