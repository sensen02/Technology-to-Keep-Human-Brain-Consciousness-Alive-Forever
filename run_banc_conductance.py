"""BANC coarse conductance feasibility, not calibrated physiology or isolation.
Explicit receptor assumptions: ACH->E=0; GABA/GLUT->E=-70. Modulators and
unknown NT excluded rather than made excitatory. Counts are NOT measured weights.
"""
from pathlib import Path
from dataclasses import replace
import json,time,resource
import numpy as np
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.neural_cond import ConductanceNetwork,ConductanceParams
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'brain_isolation'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with np.load(ROOT/'data/flywire/banc_connectome.npz',allow_pickle=False) as d:
        pre=d['pre'];post=d['post'];syn=d['syn'];nt=d['nt_pair'];n=len(d['root_ids'])
        classes=d['ann_class']; verified=d['ann_nt_verified']
    scale=1e-5 # uS per synapse, arbitrary hypothesis
    # Edge NT field is entirely empty in this downloaded BANC resource.
    # Use separately verified neuron annotation ONLY for unambiguous pure labels.
    # Mixed labels and unknown labels remain explicitly excluded.
    labels=verified[pre]
    exc=labels=='acetylcholine'; inh=np.isin(labels,['gaba','glutamate'])
    if not exc.any() or not inh.any():
        raise RuntimeError('insufficient explicit transmitter annotation; no network claim')
    We=sparse.csr_matrix((syn[exc]*scale,(post[exc],pre[exc])),shape=(n,n))
    Wi=sparse.csr_matrix((syn[inh]*scale,(post[inh],pre[inh])),shape=(n,n))
    visual=classes=='photoreceptor_neuron'
    counts={str(int(code)):int(np.count_nonzero(nt==code)) for code in np.unique(nt)}
    visual_out=visual[pre]
    visual_edges_retained=int(np.count_nonzero(visual_out & (exc|inh)))
    visual_labels={str(label):int(np.count_nonzero(verified[visual]==label)) for label in np.unique(verified[visual])}
    results=[];dt=.2;dur=100
    configs=[('silent',0,True),('visual_low',.02,True),('visual_high',.04,True),('visual_high_no_adapt',.04,False)]
    for name,drive,adapt in configs:
        p=ConductanceParams() if adapt else replace(ConductanceParams(),adaptation_increment_nA=0)
        net=ConductanceNetwork(n,p,dt_ms=dt).set_connectivity(We,Wi)
        stimulus=np.zeros(n);stimulus[visual]=drive
        t0=time.perf_counter();rates=[];sensory=[];other=[]
        for _ in range(round(dur/dt)):
            spikes=net.step(stimulus)
            rates.append(float(spikes.mean()*1000/dt))
            sensory.append(float(spikes[visual].mean()*1000/dt))
            other.append(float(spikes[~visual].mean()*1000/dt))
        elapsed=time.perf_counter()-t0
        results.append({'name':name,'visual_drive_nA':drive,'adaptation':adapt,'wall_seconds':elapsed,
              'simulated_ms':dur,'wall_seconds_per_simulated_second':elapsed/(dur/1000),
              'mean_rate_Hz':float(np.mean(rates)),'visual_mean_Hz':float(np.mean(sensory)),
              'nonvisual_mean_Hz':float(np.mean(other)),'spike_count':int(net.spike_count.sum()),
              'rate_series_Hz':rates})
        print(name,elapsed,results[-1]['mean_rate_Hz'],flush=True)
    metadata={'scope':'100ms coarse BANC benchmark, no fine cells or cut; initial transients not steady-state firing rates',
       'dataset':'BANC cached unique pairs; receptor mapping uses ann_nt_verified for presynaptic neuron',
       'critical_data_gap':'all 3990039 source connection rows have empty nt_type; cached nt_pair all -1',
       'excluded_label_policy':'unknown, mixed-transmitter, histamine and modulatory labels excluded; not inferred excitatory',
       'neurons':n,'unique_pairs':int(len(pre)),'nt_pair_counts':counts,
       'included_exc_pairs':int(exc.sum()),'included_inh_pairs':int(inh.sum()),
       'excluded_modulator_or_unknown_pairs':int((~(exc|inh)).sum()),
       'nt_verification_nonempty_neurons':int(np.count_nonzero(verified!='')),
       'visual_count':int(visual.sum()),'visual_transmitter_labels':visual_labels,
       'visual_outgoing_pairs':int(visual_out.sum()),'visual_outgoing_pairs_retained':visual_edges_retained,
       'visual_coverage_warning':'zero retained photoreceptor outgoing pairs: histamine/mixed/unknown all excluded. Zero downstream response is structurally forced, NOT a physiological prediction.',
       'weight_scale_uS_per_synapse':scale,
       'assumptions':['NT annotation does not measure receptor sign; ACH/GABA/GLUT mapping illustrative',
           'all point parameters illustrative','visual injection is an interface test NOT phototransduction or real visual current',
           '100ms different dt/current units from legacy runs; no claim of solving 245Hz regime'],
       'dt_ms':dt,'params':ConductanceParams().provenance(),'runs':results,
       'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'total_wall_seconds':time.perf_counter()-start}
    (OUT/'banc_conductance_metrics.json').write_text(json.dumps(metadata,indent=2))
    fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for r in results:
        # fixed 5ms bins; no hidden smoothing across scenarios
        values=np.array(r['rate_series_Hz']).reshape(-1,25).mean(axis=1)
        ax[0].plot(np.arange(1,len(values)+1)*5,values,label=r['name'])
    ax[0].set(xlabel='ms',ylabel='all-neuron mean Hz (5ms bins)',title='Initial response: not steady state');ax[0].legend(fontsize=7)
    ax[1].bar(np.arange(len(results)),[r['nonvisual_mean_Hz'] for r in results]);ax[1].set_xticks(np.arange(len(results)),[r['name'] for r in results],rotation=20)
    ax[1].set(ylabel='nonvisual mean Hz',title='Downstream response (partial annotated edges)')
    for i,r in enumerate(results):
        ax[1].text(i,r['nonvisual_mean_Hz'],f"{r['nonvisual_mean_Hz']:.3g}",ha='center',va='bottom')
    fig.suptitle('BANC PARTIAL-GRAPH BENCHMARK — visual output edges excluded; NOT visual-pathway validation')
    ax[1].text(.5,.8,'0 retained photoreceptor output edges\nDownstream silence is forced by exclusions',
               transform=ax[1].transAxes,ha='center',fontsize=8,color='darkred')
    fig.savefig(OUT/'banc_conductance_demo.png',dpi=150);plt.close(fig)
    print('Wrote',OUT/'banc_conductance_metrics.json',flush=True)

if __name__=='__main__': main()
