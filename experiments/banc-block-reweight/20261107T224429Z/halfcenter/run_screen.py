"""Round12 half-centre screen runner: neural open loop only, zero physics."""
from pathlib import Path
import hashlib,importlib.util,itertools,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FROZEN=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py')
RUNTIME=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation/runtime.py')
INPUTS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/descending/INPUTS.npz')
BASE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation/BASE.npz')
LEGS=['LF','LH','LM','RF','RH','RM']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def dump(p,o):
    def conv(x):
        if isinstance(x,np.bool_):return bool(x)
        if isinstance(x,np.integer):return int(x)
        if isinstance(x,np.floating):return float(x)
        raise TypeError(type(x).__name__)
    Path(p).write_text(json.dumps(o,indent=2,allow_nan=False,default=conv)+'\n')

def analyse(rates,leg_ids):
    series=np.stack([rates[:,leg_ids==L.lower()].mean(1) for L in LEGS],1)
    win=series[100:]-series[100:].mean(0)
    U,S,Vt=np.linalg.svd(win,full_matrices=False);frac=S**2/np.sum(S**2)
    C=np.corrcoef(series.T);A=series[:,[0,5,1]].mean(1);B=series[:,[3,2,4]].mean(1)
    return dict(per_leg_mean_hz={L:float(series[:,i].mean()) for i,L in enumerate(LEGS)},
                tripod_alternation_score=float(-np.corrcoef(A,B)[0,1]),
                ipsilateral_score=float(np.corrcoef(series[:,[0,1,2]].mean(1),series[:,[3,4,5]].mean(1))[0,1]),
                lf_rh_correlation=float(C[0,4]),rf_lh_correlation=float(C[3,1]),
                group_mean_hz={'A_LF_RM_LH':float(A.mean()),'B_RF_LM_RH':float(B.mean())},
                PC1_variance_fraction=float(frac[0]),PC1_loadings={L:float(v) for L,v in zip(LEGS,Vt[0])})

def main():
    prereg=json.loads((HERE/'PREREG.json').read_text())
    sys.path.insert(0,str(ROOT))
    frozen=module('_hc_frozen',FROZEN)
    mb=module('_hc_blocks',ROOT/'module_blocks.py')
    hc=module('_hc_halfcenter',HERE/'module_halfcenter.py')
    rt=module('_hc_runtime',RUNTIME)
    inp=np.load(INPUTS);base=np.load(BASE)
    sensory=np.asarray(inp['sensory_local']);recorded=np.asarray(inp['sensory_recorded_nA'])
    mn_local=np.asarray(base['mn_local']);leg_ids=np.asarray(base['mn_leg']);zero=np.zeros(20000)
    # guard: all scalars 1 must reproduce the frozen matrices and behaviour exactly
    We,Wi,det=hc.build(dict(ab_inh=1.,ab_exc=1.,ba_inh=1.,ba_exc=1.))
    ref=frozen.LocalCircuit()
    matrix_exact=bool(abs(We-ref._We).max()==0 and abs(Wi-ref._Wi).max()==0)
    assert matrix_exact,'half-centre rebuild differs from frozen at scalars=1'
    assert float(np.max(np.abs(det['row_sums'][det['row_sums']>0]-.02)))<1e-12
    class_counts={k:int(v) for k,v in det['class_edge_counts'].items()}
    class_uS={k:float(v) for k,v in det['class_conductance_uS'].items()}
    def run(circuit):
        rt.restore_circuit(circuit);rows=[]
        for b in range(400):
            s=zero.copy();s[sensory]=recorded[b%100]
            circuit.advance(s,zero,5.)
            rows.append(circuit.filtered_rates_hz[mn_local].copy())
        return np.asarray(rows)
    ref_rates=run(ref)
    class HC(mb.BlockLocalCircuit):
        def __init__(self,scalars):
            self.sc = dict(scalars)
            self.det = None
            super().__init__(scalars=None)
        def reset(self):
            We,Wi,self.det = hc.build(self.sc)
            self._We,self._Wi = We,Wi
            super().reset()
    one=HC(dict(ab_inh=1.,ab_exc=1.,ba_inh=1.,ba_exc=1.))
    behaviour_exact=bool(np.array_equal(ref_rates,run(one)))
    assert behaviour_exact,'half-centre scalars=1 behaviour differs from frozen core'
    results={'baseline':dict(analyse(ref_rates,leg_ids),scalars=dict(ab_inh=1.,ab_exc=1.,ba_inh=1.,ba_exc=1.),neural_steps=0,source='reused_guard_run')}
    total=8000;started=time.monotonic();n=0
    for ia,ib,ea,eb in itertools.product(prereg['grid']['inh_values'],prereg['grid']['inh_values'],prereg['grid']['exc_values'],prereg['grid']['exc_values']):
        scalars=dict(ab_inh=ia,ba_inh=ib,ab_exc=ea,ba_exc=eb)
        if scalars==dict(ab_inh=1.,ab_exc=1.,ba_inh=1.,ba_exc=1.):continue
        c=HC(scalars);rates=run(c);total+=4000;n+=1
        key='abInh%g_baInh%g_abExc%g_baExc%g'%(ia,ib,ea,eb)
        res=dict(analyse(rates,leg_ids),scalars=scalars,neural_steps=4000,
                 class_conductance_uS=c.det['class_conductance_uS'],rates_sha256=hashlib.sha256(rates.astype(np.float64).tobytes()).hexdigest())
        results[key]=res
        print(key,'tripod %+.3f ipsi %+.3f lf_rh %+.3f rf_lh %+.3f'%(res['tripod_alternation_score'],res['ipsilateral_score'],res['lf_rh_correlation'],res['rf_lh_correlation']),flush=True)
    positive=[k for k,v in results.items() if k!='baseline' and v['tripod_alternation_score']>=.30
              and min(v['group_mean_hz'].values())>2. and v['lf_rh_correlation']<0 and v['rf_lh_correlation']<0]
    ranked=sorted(((k,v['tripod_alternation_score']) for k,v in results.items() if k!='baseline'),key=lambda kv:-kv[1])[:6]
    summary={'status':'HALFCENTER_SCREEN_COMPLETE_NOT_WALKING','guard_matrix_exact':matrix_exact,'guard_behaviour_bit_exact':behaviour_exact,
             'class_edge_counts':class_counts,'class_conductance_uS':class_uS,
             'results':results,'alternation_positive':positive,'best_six_by_tripod_score':ranked,
             'decision_rule':prereg['decision_rule_fixed_in_advance'],'combinations_run':n,
             'total_neural_steps':total,'physical_steps':0,'training_updates':0,'new_edges':0,'new_synapses':0,'wall_s':time.monotonic()-started,
             'limits':prereg['limits'],'source_hashes':{str(p):sha(p) for p in [FROZEN,RUNTIME,prereg['frozen_sources']['canonical_neural_state'],
                prereg['frozen_sources']['pools_and_reach'],str(HERE/'module_halfcenter.py'),str(ROOT/'module_blocks.py'),str(INPUTS)]}}
    dump(HERE/'RESULT.json',summary)
    print('positive:',positive);print('best:',ranked);print('neural',total,'physical 0')

if __name__=='__main__':main()
