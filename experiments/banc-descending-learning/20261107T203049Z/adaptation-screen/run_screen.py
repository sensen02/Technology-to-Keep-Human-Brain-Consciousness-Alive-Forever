"""Preregistered adaptation-asymmetry screen: neural-only, no physics, no learning."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
FOUND=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation')
INPUTS=HERE.parent/'descending/INPUTS.npz'
LEGS=['LF','LH','LM','RF','RH','RM']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')

def leg_group_scale(circuit,side_mult,mult):
    """scale vector: mult on motors of the named group, 1 elsewhere."""
    scale=np.ones(20000)
    legs=np.char.lower(LEGS); idx=np.array([list(legs).index(x.lower()) for x in circuit.mn_leg])
    member=np.isin(idx,side_mult)
    scale[circuit.mn_local[member]]=mult
    return scale, int(member.sum())

def analyse(rates,leg_ids,orders):
    series=np.stack([rates[:,leg_ids==L.lower()].mean(1) for L in LEGS],1)
    win=series[100:]-series[100:].mean(0)
    freqs=np.fft.rfftfreq(win.shape[0],d=.005);spec=np.abs(np.fft.rfft(win*np.hanning(win.shape[0])[:,None],axis=0))**2
    band=(freqs>=.05)&(freqs<=20)
    dominant={L:float(freqs[np.argmax(np.where(band,spec[:,i],-1))]) for i,L in enumerate(LEGS)}
    C=np.corrcoef(series.T)
    A=series[:,[0,5,1]].mean(1);B=series[:,[3,2,4]].mean(1)
    return dict(per_leg_mean_hz={L:float(series[:,i].mean()) for i,L in enumerate(LEGS)},
                per_leg_cv={L:float(series[:,i].std()/series[:,i].mean()) for i,L in enumerate(LEGS)},
                tripod_alternation_score=float(-np.corrcoef(A,B)[0,1]),
                ipsilateral_score=float(np.corrcoef(series[:,[0,1,2]].mean(1),series[:,[3,4,5]].mean(1))[0,1]),
                pair_correlations={LEGS[i]+'_'+LEGS[j]:float(C[i,j]) for i in range(6) for j in range(i+1,6)},
                dominant_hz=dominant,group_mean_hz={'A_LF_RM_LH':float(A.mean()),'B_RF_LM_RH':float(B.mean())},
                all_leg_mean_std_hz=float(series.mean(1).std()))

def main():
    prereg=json.loads((HERE/'PREREG.json').read_text())
    frozen=module('_screen_frozen',prereg['frozen_sources']['neural_core'])
    scaled=module('_screen_scaled',HERE/'module_scale.py')
    rt=module('_screen_runtime',prereg['frozen_sources']['runtime'])
    inp=np.load(INPUTS);base=np.load(prereg['frozen_sources']['leg_mapping'])
    sensory=np.asarray(inp['sensory_local']);recorded=np.asarray(inp['sensory_recorded_nA'])
    mn_local=np.asarray(base['mn_local']);leg_ids=np.asarray(base['mn_leg'])
    zero=np.zeros(20000)
    def run(circuit):
        c=circuit;rt.restore_circuit(c);rows=[]
        for b in range(400):
            s=zero.copy();s[sensory]=recorded[b%100]
            c.advance(s,zero,5.)
            rows.append(c.filtered_rates_hz[mn_local].copy())
        return np.asarray(rows)
    # equivalence guard first: scale==1 must be bit-identical to the frozen core
    ref=run(frozen.LocalCircuit())
    one=run(scaled.ScaledLocalCircuit())
    equiv=bool(np.array_equal(ref,one))
    assert equiv,'scale==1 does not reproduce the frozen core'
    schemes=[('baseline',None,1.)]
    for mult in (2.,4.):
        schemes.append((f'ipsilateral_x{int(mult)}',[0,1,2],mult))
        schemes.append((f'tripod_x{int(mult)}',[0,5,1],mult))
    results={};total=4000  # equivalence guard run: 400 bins x 10 substeps
    for name,group,mult in schemes:
        if group is None:
            c=scaled.ScaledLocalCircuit()
            scaled_motors=0
        else:
            probe=scaled.ScaledLocalCircuit()
            vec,scaled_motors=leg_group_scale(probe,group,mult)
            c=scaled.ScaledLocalCircuit(increment_scale=vec)
        if name=='baseline':
            rates=ref.copy()
        else:
            rates=run(c);total+=4000
        res=analyse(rates,leg_ids,LEGS)
        res.update(neural_steps=4000,physical_steps=0,training_updates=0,scaled_motors=scaled_motors,multiplier=mult,rates_sha256=hashlib.sha256(rates.tobytes()).hexdigest())
        results[name]=res
        np.savez_compressed(HERE/f'RATES_{name}.npz',rates_hz=rates.astype(np.float32),time_s=1.0+np.arange(400)*.005,leg=leg_ids,leg_order=np.asarray(LEGS))
        print(name,json.dumps({k:res[k] for k in ('tripod_alternation_score','ipsilateral_score','group_mean_hz','scaled_motors','neural_steps')}),flush=True)
    positive=[k for k,v in results.items() if k!='baseline' and v['tripod_alternation_score']>=.3 and min(v['group_mean_hz'].values())>2.]
    summary={'status':'ADAPTATION_ASYMMETRY_SCREEN_COMPLETE_NOT_WALKING','equivalence_scale1_bit_exact':equiv,
             'schemes':results,'alternation_positive':positive,'decision_rule':prereg['decision_rule_fixed_in_advance'],
             'total_neural_steps':total,'physical_steps':0,'training_updates':0,'new_synapses':0,
             'limits':prereg['limits'],'source_hashes':{str(p):sha(p) for p in [prereg['frozen_sources']['neural_core'],prereg['frozen_sources']['runtime'],prereg['frozen_sources']['canonical_neural_state'],str(HERE/'module_scale.py'),str(INPUTS)]}}
    dump(HERE/'RESULT.json',summary)
    print('positive:',positive)
    print('total neural steps',total,'physical 0')

if __name__=='__main__':main()
