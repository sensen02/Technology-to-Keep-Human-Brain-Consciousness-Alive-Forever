"""Full BANC sensory-projection benchmark; recurrence absent, parameters illustrative."""
from pathlib import Path
import json,time
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from engine.graded_vision import GradedVision,pure_histamine_mask
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'/'brain_isolation'
def main():
 OUT.mkdir(parents=True,exist_ok=True); t0=time.perf_counter()
 with np.load(ROOT/'data/flywire/banc_connectome.npz',allow_pickle=False) as d:
  pre,post,syn=d['pre'],d['post'],d['syn']; nt=d['nt_pair']; cls=d['ann_class']; ver=d['ann_nt_verified']; n=len(d['root_ids'])
 visual=np.flatnonzero(cls=='photoreceptor_neuron'); hist=pure_histamine_mask(ver,pre)
 gv=GradedVision(n,visual,pre,post,syn,hist,dt_ms=.2,scale_uS_per_synapse=1e-5,e_hist_mV=-70,tau_release_ms=8)
 params=ConductanceParams(adaptation_increment_nA=0.0,inh_reversal_mV=-70,inh_tau_ms=8); net=ConductanceNetwork(n,params,dt_ms=.2)
 probe=np.unique(post[np.isin(pre,visual)&hist]); scenarios=[('dark',0.0,100),('light',1.0,100),('washout',0.0,100)]
 traces=[]; target_traces=[]; means=[]; gi_traces=[]; inc_traces=[]
 for name,level,dur in scenarios:
  vals=[]; targets=[]; conductances=[]; increments=[]
  for _ in range(round(dur/.2)):
   inc=gv.apply_to(net,level); vals.append(float(net.v.mean())); targets.append(float(net.v[probe].mean())); conductances.append(float(net.gi.sum())); increments.append(float(inc.sum()))
  traces.extend(vals); target_traces.extend(targets); gi_traces.extend(conductances); inc_traces.extend(increments)
  means.append({'name':name,'whole_network_mean_Vm_mV':float(np.mean(vals)),'target_mean_Vm_mV':float(np.mean(targets)),'target_final_Vm_mV':targets[-1],'mean_post_decay_histamine_conductance_uS':float(np.mean(conductances)),'mean_injected_increment_uS_per_step':float(np.mean(increments))})
 t=np.arange(len(traces))*.2+.2
 metadata={'status':'completed','scope':'full neuron count, actual BANC sensory projection only; We/Wi zero, recurrent network absent','neurons':n,'unique_pairs':int(len(pre)),'nt_pair_all_minus_one':bool(np.all(nt==-1)),'visual_receptors':int(len(visual)),'pure_histamine_presynaptic_neurons':int(np.sum(ver=='histamine')),'pure_histamine_edges_all':int(hist.sum()),'visual_pure_histamine_edges':int(np.sum(hist & np.isin(pre,visual))),'reachable_posts_visual_histamine':int(probe.size),'excluded_mixed_ACh_histamine_edges':int(np.sum(ver[pre]=='acetylcholine,histamine')),'excluded_unknown_edges':int(np.sum(ver[pre]=='')),'excluded_non_histamine_edges':int(np.sum(~hist)),'E_hist_mV':-70.0,'tau_release_ms':8.0,'mapping':'ann_nt_verified exact pure histamine only; no nt_pair fallback; E_hist explicit chloride hypothesis','params_status':'illustrative, not biologically calibrated','scenarios':means,'wall_seconds':time.perf_counter()-t0}
 (OUT/'graded_vision_banc.json').write_text(json.dumps(metadata,indent=2)); np.savez_compressed(OUT/'graded_vision_banc.npz',times_ms=t,whole_mean_vm_mV=np.array(traces),target_mean_vm_mV=np.array(target_traces),target_indices=probe,post_decay_histamine_conductance_uS=np.array(gi_traces),injected_increment_uS_per_step=np.array(inc_traces))
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for ax,values,title in zip(axes,[target_traces,traces],['Reachable target mean Vm','Whole-network mean Vm']):
  ax.plot(t,values,lw=1); ax.axvspan(0,100,color='gray',alpha=.12,label='dark'); ax.axvspan(100,200,color='gold',alpha=.16,label='light'); ax.axvspan(200,300,color='steelblue',alpha=.10,label='washout'); ax.set(xlabel='time (ms)',ylabel='Vm (mV)',title=title); ax.legend()
  ax.ticklabel_format(axis='y',useOffset=False)  # avoid misleading -6.5e1 offset label
 fig.suptitle('BANC sensory projection ONLY (no recurrence) — illustrative E_hist=-70 mV'); fig.tight_layout(); fig.savefig(OUT/'graded_vision_banc.png',dpi=150); plt.close(fig)
 print(json.dumps(metadata,indent=2))
if __name__=='__main__': main()
