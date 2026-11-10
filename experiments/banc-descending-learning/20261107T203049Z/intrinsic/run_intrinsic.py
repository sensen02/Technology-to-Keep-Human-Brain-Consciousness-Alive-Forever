"""Preregistered intrinsic-dynamics replay of the frozen local module; no physics, no learning."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FOUND=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation')
INPUTS=ROOT/'descending/INPUTS.npz'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')

def main():
    prereg=json.loads((HERE/'PREREG.json').read_text())
    for label,path in prereg['frozen_sources'].items():
        assert Path(path).exists(),(label,path)
    rt=module('_intrinsic_runtime',prereg['frozen_sources']['runtime'])
    inp=np.load(INPUTS);base=np.load(prereg['frozen_sources']['leg_mapping'])
    sensory=np.asarray(inp['sensory_local']);allmn=np.asarray(inp['all_mn_local'])
    mn_local=np.asarray(base['mn_local']);leg=np.asarray(base['mn_leg']);legs=['LF','LH','LM','RF','RH','RM']
    pos=np.searchsorted(allmn,mn_local);assert np.array_equal(allmn[pos],mn_local)
    zero=np.zeros(20000);results={};total=0
    held_row=np.asarray(inp['sensory_held_nA'])[0];assert np.array_equal(np.asarray(inp['sensory_held_nA']),held_row[None,:].repeat(100,0))
    for condition in ['held','replay']:
        c=rt.load_circuit();start=np.array(rt.snapshot(c)['spike_count'],copy=True)
        rows=[];dn=[]
        for b in range(1000):
            s=zero.copy()
            s[sensory]=held_row if condition=='held' else inp['sensory_recorded_nA'][b%100]
            total+=10
            for sub in range(10):
                c.advance(s,zero,.5)
            rows.append(c.filtered_rates_hz[mn_local].astype(np.float32));dn.append(c.filtered_rates_hz[[int(x) for x in inp['dn_local']]].copy())
            assert np.isfinite(rows[-1]).all() and np.isfinite(c.net.v).all()
        rates=np.asarray(rows,float)             # (1000,323) 5ms bins
        series=np.stack([rates[:,leg==L.lower()].mean(1) for L in legs],1)
        win=series[100:]-series[100:].mean(0)    # 0.5-5 s
        freqs=np.fft.rfftfreq(win.shape[0],d=.005);hann=np.hanning(win.shape[0])[:,None]
        spec=np.abs(np.fft.rfft(win*hann,axis=0))**2
        band=(freqs>=.05)&(freqs<=20)
        dominant=[float(freqs[np.argmax(np.where(band,spec[:,i],-1))]) for i in range(6)]
        C=np.corrcoef(series.T)
        pairs={'LF_RH':(0,4),'RF_LH':(3,1),'LM_LM':(2,2)}
        results[condition]={
          'neural_steps':10000,'physical_steps':0,'training_updates':0,
          'per_leg_mean_hz':{L:float(series[:,i].mean()) for i,L in enumerate(legs)},
          'per_leg_mean_rate_cv':{L:float(series[:,i].std()/series[:,i].mean()) for i,L in enumerate(legs)},
          'cross_correlation':{legs[i]+'_'+legs[j]:float(C[i,j]) for i in range(6) for j in range(6) if i<j},
          'dominant_frequency_hz':{L:v for L,v in zip(legs,dominant)},
          'alternation_score_minus_corr':{k:(-C[a,b] if a!=b else float(C[a,b])) for k,(a,b) in pairs.items()},
          'all_leg_mean_rate_std_hz':float(series.mean(1).std()),
          'DN_spikes':[int(x) for x in (c.net.spike_count[[int(y) for y in inp['dn_local']]]-start[[int(y) for y in inp['dn_local']]])],
          'rates_sha256':hashlib.sha256(rates.astype(np.float64).tobytes()).hexdigest(),
          'voltage_min_mV':float(c.net.v.min()),'voltage_max_mV':float(c.net.v.max())}
        np.savez_compressed(HERE/f'RATES_{condition}.npz',rates_hz=rates,time_s=1.0+np.arange(1000)*.005,leg=leg,leg_order=np.asarray(legs),dn_rates_hz=np.asarray(dn))
    summary={'status':'INTRINSIC_DYNAMICS_AUDIT_COMPLETE_NOT_WALKING','conditions':results,'total_neural_steps':total,'physical_steps':0,
             'limits':prereg['limits'],'source_hashes':{str(p):sha(p) for p in [prereg['frozen_sources'][k] for k in ('runtime','neural_core','canonical_neural_state','sensory_currents')]}}
    dump(HERE/'RESULT.json',summary)
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('cross_correlation',)} for k,v in results.items()},indent=2))
    print(json.dumps(results['held']['cross_correlation'],indent=2))

if __name__=='__main__':main()
