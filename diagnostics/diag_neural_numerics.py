"""Isolated implementation/numerics audit. No behavior target or parameter fit."""
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib,collections
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from engine.neural_cond import ConductanceNetwork,ConductanceParams
OUT=ROOT/'outputs/neural_numerics_round11'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 checks={};constant=[];pulse=[];two=[];strong=[]
 for dt in (.5,.25,.1):
  p=replace(ConductanceParams(),threshold_mV=100.,adaptation_increment_nA=0.)
  for ge,gi,I in ((0.,0.,.01),(.003,0.,0.),(0.,.003,0.),(.03,.02,.025)):
   n=ConductanceNetwork(1,p,dt);G=p.leak_uS+ge+gi
   eq=(p.leak_uS*p.rest_mV+ge*p.exc_reversal_mV+gi*p.inh_reversal_mV+I)/G
   v=[];exact=[]
   for k in range(round(20/dt)):
    # External increments exactly restore known constant interval conductance.
    n.step(I,ge-n.ge[0],gi-n.gi[0]);v.append(float(n.v[0]));exact.append(float(eq+(p.rest_mV-eq)*np.exp(-G*n.t_ms/p.capacitance_nF)))
   err=np.max(np.abs(np.array(v)-exact))
   discrete=eq+(p.rest_mV-eq)*(1+dt*G/p.capacitance_nF)**(-round(20/dt))
   checks[f'backward_euler_formula_{dt}_{ge}_{gi}']=bool(abs(n.v[0]-discrete)<1e-10)
   constant.append({'dt_ms':dt,'ge_uS':ge,'gi_uS':gi,'I_nA':I,'max_error_mV':float(err),'final_mV':float(n.v[0]),'times_ms':(np.arange(len(v))+1).tolist() if False else (dt*(np.arange(len(v))+1)).tolist(),'voltage_mV':v,'analytic_mV':exact})
  for I in (.025,.05):
   n=ConductanceNetwork(1,dt_ms=dt);ts=[];vol=[]
   for k in range(round(200/dt)):
    if n.step(I if k*dt<50 else 0)[0]:ts.append(n.t_ms)
    vol.append(float(n.v[0]))
   pulse.append({'dt_ms':dt,'I_nA':I,'spike_times_ms':ts,'release_spikes':sum(t>50 for t in ts),'voltage_mV':vol})
  # One directed synapse row1,col0; force exactly one threshold crossing of cell0.
  for sign in ('exc','inh'):
   n=ConductanceNetwork(2,dt_ms=dt);w=sparse.csr_matrix(([.01],([1],[0])),shape=(2,2));z=sparse.csr_matrix((2,2))
   n.set_connectivity(w if sign=='exc' else z,w if sign=='inh' else z)
   first=n.step([10,0]);assert first.tolist()==[True,False]
   emitted=n.t_ms;arrival=None;target_voltage=None
   for k in range(round(4/dt)):
    start=n.t_ms;n.step()
    g=n.ge if sign=='exc' else n.gi
    if g[1]>0 and arrival is None:arrival=start;target_voltage=float(n.v[1])
    assert g[0]==0
   checks[f'delay_orientation_{sign}_{dt}']=bool(np.isclose(arrival-emitted,2))
   checks[f'sign_{sign}_{dt}']=bool(target_voltage>-65 if sign=='exc' else target_voltage<-65)
   two.append({'dt_ms':dt,'sign':sign,'emission_ms':emitted,'arrival_start_ms':arrival,'first_target_voltage_mV':target_voltage})
  n=ConductanceNetwork(1,dt_ms=dt);times=[]
  for k in range(round(20/dt)):
   if n.step(10)[0]:times.append(n.t_ms)
  checks[f'refractory_{dt}']=bool(np.min(np.diff(times))>=2+dt-1e-8)
  # Deliberate strong excitatory recurrent pair, fixed .01uS per edge =20 default-count synapses.
  n=ConductanceNetwork(2,dt_ms=dt);n.set_connectivity([[0,.01],[.01,0]],np.zeros((2,2)));counts=np.zeros(2,int)
  for k in range(round(100/dt)):
   counts+=n.step([.05,0] if k*dt<20 else 0)
   assert np.all(np.isfinite(n.v))
  strong.append({'dt_ms':dt,'edge_uS':.01,'counts':counts.tolist(),'final_v_mV':n.v.tolist()})
 from engine.embodied.adapters import TierConfig,build_neural_tier
 b=build_neural_tier(TierConfig(background_mean_nA=0,background_sd_nA=0));a=b['data'];gi=b['global_index'];edges=b['edges']
 from engine.neck_cut_data import TouchTierConfig
 cap=TouchTierConfig().max_synapses_per_edge;scale=.0005
 for key,rowskey in (('We','exc_rows'),('Wi','inh_rows')):
  rows=edges[rowskey];loc=b['local_of_global'];ref=sparse.csr_matrix((np.minimum(a['syn'][rows].astype(float),cap)*scale,(loc[a['post'][rows]],loc[a['pre'][rows]])),shape=(b['n'],b['n']))
  delta=ref-edges[key];checks[key+'_matches_post_pre_capped_weights']=bool(delta.nnz==0 or np.max(abs(delta.data))<1e-14)
 motor=np.asarray(a['ann_super_class'])[gi]=='motor';nt=np.asarray(a['ann_nt_verified'])[gi]
 motor_nt=dict(collections.Counter(map(str,nt[motor])))
 from engine.embodied.feco_identity import resolve_feco
 from engine.embodied.adapters import NeuralTier,OwnershipLedger
 groups=resolve_feco(ROOT/'data/flywire/banc_neurons.csv.gz',b)
 club=np.array([r['local_index'] for r in groups['club'] if r['local_index'] is not None],int)
 full=[]
 for dt in (.5,.25,.1):
  tier=NeuralTier(b,TierConfig(dt_ms=dt,background_mean_nA=0,background_sd_nA=0),OwnershipLedger())
  current=np.zeros(b['n']);current[club]=.05
  on=tier.advance(round(50/dt),current);off=tier.advance(round(100/dt),np.zeros(b['n']))
  full.append({'dt_ms':dt,'pulse_spikes':int(on.sum()),'release_spikes':int(off.sum()),'MN_pulse_spikes':int(on[:,motor].sum()),'MN_release_spikes':int(off[:,motor].sum()),'finite':bool(np.all(np.isfinite(tier.net.v)))})
  print('full network',full[-1],flush=True)
 sources=[ROOT/'engine/neural_cond.py',ROOT/'engine/embodied/adapters.py',ROOT/'engine/neck_cut_data.py',Path(__file__)]
 report={'status':'NUMERICS_AUDIT_NOT_BIOLOGICAL_VALIDATION','checks':checks,'constant_conductance':constant,'single_neuron_pulse':pulse,'two_cell_orientation_delay':two,'strong_recurrent_pair':strong,'motor_NT_counts':motor_nt,'full_network_fixed_input_dt_convergence':full,
 'unit_identity':'1uS*1mV=1nA; 1nF*1mV/1ms=1nA; C/g gives ms',
 'source_evidence':{'voltage':'neural_cond.py:121-125 backward Euler; conductances held at START then exact decay at135-136; first-order splitting not exact time-varying solution','delay':'neural_cond.py:112-140 end-spike to start-arrival ring; tested 2ms','sign':'neck_cut_data.py:1163-1189 pure presynaptic ACh excitatory, GABA/glutamate inhibitory CNS receptor assumption; not NMJ sign','orientation':'neck_cut_data.py:1200 row post col pre; neural_cond.py:116 matvec'},
 'conclusion':'No arithmetic/orientation/sign-flip/refractory/delay defect proven in tested core. Finite step integration errors are measurable; illustrative thresholds/receptor assumptions remain unvalidated. Motor glutamate inhibitory CNS outgoing edges does NOT imply inhibitory NMJ: muscle spike readout is separate and positive.',
 'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(12,5))
 for ge,gi,I in ((0.,0.,.01),(.003,0.,0.),(0.,.003,0.),(.03,.02,.025)):
  r=[x for x in constant if (x['ge_uS'],x['gi_uS'],x['I_nA'])==(ge,gi,I)]
  ax[0].plot([x['dt_ms'] for x in r],[x['max_error_mV'] for x in r],'o-',label=f'ge={ge},gi={gi},I={I}')
 for r in pulse:
  ax[1].scatter(r['spike_times_ms'],[r['I_nA']*100+ r['dt_ms']]*len(r['spike_times_ms']),label=f"I={r['I_nA']}nA dt={r['dt_ms']}ms")
 ax[0].set(xlabel='dt (ms)',ylabel='Max voltage error vs exact (mV)',title='Constant conductance: first-order voltage integration');ax[1].set(xlabel='Spike time (ms)',ylabel='Condition offset',title='Single-cell50ms pulse,150ms release');ax[1].axvline(50,color='gray',ls='--')
 for aa in ax:aa.legend(fontsize=7);aa.grid(alpha=.2)
 fig.suptitle('Implementation tests, NOT physiological calibration');fig.tight_layout();fig.savefig(OUT/'numerics.png',dpi=140)
 print('checks',len(checks),all(checks.values()),'motorNT',motor_nt,'recurrent',strong)
 for r in constant:print('constant',r['dt_ms'],r['ge_uS'],r['gi_uS'],r['max_error_mV'])
 for r in pulse:print('pulse',r['dt_ms'],r['I_nA'],r['spike_times_ms'])
 return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
