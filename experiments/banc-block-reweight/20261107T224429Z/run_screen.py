"""Round12 preregistered block-reweight screen: neural open loop only, no physics."""
from pathlib import Path
import hashlib,importlib.util,itertools,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
FOUND=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation')
INPUTS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-descending-learning/20261107T203049Z/descending/INPUTS.npz')
LEGS=['LF','LH','LM','RF','RH','RM']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
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
    C=np.corrcoef(series.T)
    A=series[:,[0,5,1]].mean(1);B=series[:,[3,2,4]].mean(1)
    return dict(per_leg_mean_hz={L:float(series[:,i].mean()) for i,L in enumerate(LEGS)},
                tripod_alternation_score=float(-np.corrcoef(A,B)[0,1]),
                ipsilateral_score=float(np.corrcoef(series[:,[0,1,2]].mean(1),series[:,[3,4,5]].mean(1))[0,1]),
                lf_rh_correlation=float(C[0,4]),rf_lh_correlation=float(C[3,1]),
                lf_rf_correlation=float(C[0,3]),
                group_mean_hz={'A_LF_RM_LH':float(A.mean()),'B_RF_LM_RH':float(B.mean())},
                PC1_variance_fraction=float(frac[0]),PC1_loadings={L:float(v) for L,v in zip(LEGS,Vt[0])},
                all_leg_mean_std_hz=float(series.mean(1).std()))

def main():
    prereg=json.loads((HERE/'PREREG.json').read_text())
    frozen=module('_rw_frozen',prereg['frozen_sources']['neural_core'])
    blocks=module('_rw_blocks',HERE/'module_blocks.py')
    rt=module('_rw_runtime',prereg['frozen_sources']['runtime'])
    inp=np.load(INPUTS);base=np.load(prereg['frozen_sources']['leg_mapping'])
    sensory=np.asarray(inp['sensory_local']);recorded=np.asarray(inp['sensory_recorded_nA'])
    mn_local=np.asarray(base['mn_local']);leg_ids=np.asarray(base['mn_leg'])
    zero=np.zeros(20000)
    # guard 1: rebuild with all scalars 1 must equal the frozen matrices
    We,Wi,stats1=blocks.build_matrices(dict(contra_same_row=1.,front_hind=1.,tripod_diagonal=1.))
    ref=frozen.LocalCircuit()
    matrix_exact=bool((abs(We-ref._We).max()==0) and (abs(Wi-ref._Wi).max()==0))
    assert matrix_exact,'rebuilt matrices differ from frozen at scalars=1'
    def run(circuit):
        rt.restore_circuit(circuit);rows=[]
        for b in range(400):
            s=zero.copy();s[sensory]=recorded[b%100]
            circuit.advance(s,zero,5.)
            rows.append(circuit.filtered_rates_hz[mn_local].copy())
        return np.asarray(rows)
    # guard 2: rebuilt circuit at scalars 1 must reproduce frozen behaviour bit-exactly
    ref_rates=run(ref);one_rates=run(blocks.BlockLocalCircuit())
    behaviour_exact=bool(np.array_equal(ref_rates,one_rates))
    assert behaviour_exact,'scalars=1 behaviour differs from frozen core'
    results={};total=4000*2
    def key(vals):return 'contra%s_front%s_tripod%s'%tuple(('%g'%v).replace('.','p') for v in vals)
    baseline=analyse(ref_rates,leg_ids);baseline.update(scalars={'contra_same_row':1.,'front_hind':1.,'tripod_diagonal':1.},neural_steps=0,source='reused_guard_run')
    results['baseline']=baseline
    values=prereg['grid']['values'];started=time.monotonic()
    for combo in itertools.product(values,repeat=3):
        if combo==(1.,1.,1.):continue
        scalars=dict(contra_same_row=combo[0],front_hind=combo[1],tripod_diagonal=combo[2])
        c=blocks.BlockLocalCircuit();st=blocks.apply_scalars(c,scalars)
        rates=run(c);total+=4000
        res=analyse(rates,leg_ids);res.update(scalars=scalars,neural_steps=4000,class_conductance_uS=st['class_conductance_uS'],
                                               row_sum_max_abs_deviation_uS=st['row_sum_max_abs_deviation_uS'],total_conductance_uS=st['total_conductance_uS'],
                                               rates_sha256=hashlib.sha256(rates.astype(np.float64).tobytes()).hexdigest())
        results[key(combo)]=res
        print(key(combo),'tripod %.3f ipsi %.3f lf_rh %.3f rf_lh %.3f'%(res['tripod_alternation_score'],res['ipsilateral_score'],res['lf_rh_correlation'],res['rf_lh_correlation']),flush=True)
    positive=[k for k,v in results.items() if k!='baseline' and v['tripod_alternation_score']>=.30
              and min(v['group_mean_hz'].values())>2. and v['lf_rh_correlation']<0 and v['rf_lh_correlation']<0]
    ranked=sorted(((k,v['tripod_alternation_score']) for k,v in results.items() if k!='baseline'),key=lambda kv:-kv[1])[:5]
    summary={'status':'BLOCK_REWEIGHT_SCREEN_COMPLETE_NOT_WALKING','guard_matrix_exact':matrix_exact,'guard_behaviour_bit_exact':behaviour_exact,
             'results':results,'alternation_positive':positive,'best_five_by_tripod_score':ranked,
             'decision_rule':prereg['decision_rule_fixed_in_advance'],'expected_headroom':prereg['expected_headroom_from_structure'],
             'total_neural_steps':total,'physical_steps':0,'training_updates':0,'new_edges':0,'new_synapses':0,
             'wall_s':time.monotonic()-started,'limits':prereg['limits'],
             'source_hashes':{str(p):sha(p) for p in [prereg['frozen_sources']['neural_core'],prereg['frozen_sources']['runtime'],
                prereg['frozen_sources']['canonical_neural_state'],prereg['frozen_sources']['pools'],str(HERE/'module_blocks.py'),str(INPUTS)]}}
    dump(HERE/'RESULT.json',summary)
    print('positive:',positive);print('best:',ranked);print('total neural steps',total,'physical 0')

if __name__=='__main__':main()
