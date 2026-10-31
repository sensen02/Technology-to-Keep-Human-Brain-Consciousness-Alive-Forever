"""S3/P3 one fixed historical-input neural replay; no body/model imports or steps."""
from pathlib import Path
import inspect,json,sys,hashlib,textwrap
import numpy as np
from engine.neural_cond import ConductanceNetwork,ConductanceParams
from engine.embodied.adapters import build_neural_tier,TierConfig
O=Path('/home/sensen/Desktop/engine修复验收/s3-silent-motor-drive/20261027T204100Z')
BASE=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z')
COMP=Path('/home/sensen/Desktop/engine修复验收/s2-muscle-components/20261027T175000Z')
MAP=Path('/home/sensen/Desktop/engine修复验收/s2-postclip-half-command/20261027T181000Z/MAPPING.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item(),allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);ivpath=BASE/'normal/INTERVALS.npz';recpath=BASE/'normal/RECORDS.json';ledger=COMP/'RECRUITMENT_COMPONENT_SUMMARY.json';code=Path(inspect.getfile(ConductanceNetwork));sources=[Path(__file__),code,ivpath,recpath,ledger,MAP,Path('/run/media/sensen/Data2/cell_wound_prototype/data/flywire/banc_connectome.npz')];hashes={str(p):sha(p) for p in sources};dump(O/'PREREGISTRATION.json',dict(source_hashes=hashes,scope='one40ms fixed historical neural input replay, no body',verification='full neuron counts per5ms and exact historical probe events per500us; historical full-network per-tick raster unavailable, cannot certify all-node spike timing',mechanism_gate='no attribution to old run unless exact full-network node/time reference available; preserve descriptive replay data and mark verification gap',timing='arrivals at start; ge/gi before and after arrivals; v_before and v_trial before reset; postdecay separate',predictive_transmitter='model hypothesis, not physiological truth'))
 for root,key in [(BASE,'normal/INTERVALS.npz'),(BASE,'normal/RECORDS.json'),(COMP,'RECRUITMENT_COMPONENT_SUMMARY.json'),(MAP.parent,'MAPPING.json')]:
  assert sha(root/key)==json.loads((root/'MANIFEST_SHA256.json').read_text())[key]
 original=json.loads(ledger.read_text())['states'][0]['zero_event_components'];names=[r['actuator'] for r in original];assert len(names)==11;mapping=json.loads(MAP.read_text());selected=[r for r in mapping['rows'] if r['actuator'] in names];assert len(selected)==11;nodes=sorted({i for r in selected for i in r['local_indices']});dump(O/'TARGETS.json',dict(source_11_muscles=selected,unique_local_nodes=nodes,shared_pools_preserved=True))
 built=build_neural_tier(TierConfig(background_mean_nA=0,background_sd_nA=0));net=ConductanceNetwork(built['n'],ConductanceParams(),dt_ms=.5);net.set_connectivity(built['edges']['We']*.2,built['edges']['Wi']*.2);history=np.load(ivpath);inputs=history['sensory_current_nA'][:8];counts=history['spikes'][:8];rec=json.loads(recpath.read_text());probe=rec['probe_local_indices'];records=[]
 src=inspect.getsource(ConductanceNetwork.step);src=src.replace('    def step(', 'def replay_step(',1)
 # Dedent original method exactly one class level.
 src=textwrap.dedent(inspect.getsource(ConductanceNetwork.step));src='def replay_step'+src[len('def step'):]
 src=src.replace('    self.ge += self.We @ arrivals + ee','    drive_e=self.We @ arrivals + ee\n    drive_i=self.Wi @ arrivals + ei\n    v_before=self.v.copy()\n    ge_before=self.ge.copy();gi_before=self.gi.copy()\n    adaptation_before=self.adaptation.copy()\n    self.ge += drive_e').replace('    self.gi += self.Wi @ arrivals + ei','    self.gi += drive_i')
 src=src.replace('    spiked = active & (self.v >= p.threshold_mV)', '''    v_trial=self.v.copy()
    target=np.asarray(target_nodes)
    entries.append(dict(start_ms=float(self.t_ms),end_ms=float(self.t_ms+dt),v_before=v_before[target],v_trial=v_trial[target],threshold_mV=p.threshold_mV,active=active[target],refractory_until_ms=self.refractory_until_ms[target].copy(),adaptation_before=adaptation_before[target],ge_before=ge_before[target],gi_before=gi_before[target],exc_event_conductance=drive_e[target],inh_event_conductance=drive_i[target],exc_arrival_count=np.asarray((self.We[target,:]!=0)@arrivals.astype(int)).ravel(),inh_arrival_count=np.asarray((self.Wi[target,:]!=0)@arrivals.astype(int)).ravel(),ge_solve=self.ge[target].copy(),gi_solve=self.gi[target].copy(),external_current=I[target].copy(),exc_current_at_v_before=self.ge[target]*(p.exc_reversal_mV-v_before[target]),inh_current_at_v_before=self.gi[target]*(p.inh_reversal_mV-v_before[target]),exc_current_at_v_trial=self.ge[target]*(p.exc_reversal_mV-v_trial[target]),inh_current_at_v_trial=self.gi[target]*(p.inh_reversal_mV-v_trial[target])))
    spiked = active & (self.v >= p.threshold_mV)''')
 src=src.replace('    return spiked','    entries[-1].update(v_post=self.v[target].copy(),ge_post_decay=self.ge[target].copy(),gi_post_decay=self.gi[target].copy(),adaptation_post=self.adaptation[target].copy(),spiked=spiked[target].copy())\n    return spiked')
 ns=dict(np=np,target_nodes=nodes,entries=records);exec(compile(src,__file__,'exec'),ns);events=[];allsp=[]
 for k,current in enumerate(inputs):
  sp=np.array([ns['replay_step'](net,current_nA=current) for _ in range(10)]);allsp.append(sp);events.append(np.argwhere(sp[:,probe]).tolist())
 sp=np.concatenate(allsp);checks=dict(full_neuron_5ms_counts_exact=bool(np.array_equal(np.array(allsp).sum(1),counts)),probe_node_tick_events_exact=all(events[k]==rec['records'][k]['probe_raw_spike_events_step_column'] for k in range(8)),full_network_node_tick_reference_available=False)
 dump(O/'REPLAY_VERIFICATION.json',checks);np.savez_compressed(O/'REPLAY_ALL_NODE_TICK_SPIKES.npz',spikes=sp,step_end_ms=np.arange(1,81)*.5);dump(O/'TARGET_DRIVE_TIME_SERIES.json',records);(O/'INSTRUMENTED_STEP.py').write_text(src)
 assert checks['full_neuron_5ms_counts_exact'] and checks['probe_node_tick_events_exact'],'replay differs: do not interpret'
 first_nonfoot=.0387375;usable=[r for r in records if r['end_ms']/1000<first_nonfoot];summary=[]
 for j,node in enumerate(nodes):
  gap=np.array([r['threshold_mV']-r['v_trial'][j] for r in usable]);best=int(gap.argmin());summary.append(dict(local_node=node,root_id=int(built['tier']['root_ids'][node]),muscles=[r['actuator'] for r in selected if node in r['local_indices']],exc_arrival_events=int(sum(r['exc_arrival_count'][j] for r in usable)),inh_arrival_events=int(sum(r['inh_arrival_count'][j] for r in usable)),minimum_threshold_distance_mV=float(gap[best]),closest_step_end_ms=usable[best]['end_ms'],max_voltage_change_mV=float(max(r['v_trial'][j]-r['v_before'][j] for r in usable)),emitted_spikes=int(sp[:len(usable),node].sum()),status='REPLAY_DESCRIPTION_ONLY_FULL_NETWORK_TIMING_REFERENCE_MISSING'))
 dump(O/'NODE_SUMMARY.json',summary);dump(O/'MUSCLE_SUMMARY.json',[dict(actuator=r['actuator'],local_indices=r['local_indices'],nodes=[v for v in summary if v['local_node'] in r['local_indices']],mechanism='NOT_ATTRIBUTED_TO_HISTORY: full-network node/time reference unavailable') for r in selected]);assert all(sha(Path(p))==v for p,v in hashes.items());dump(O/'EXECUTION.json',dict(command=sys.executable+' '+str(Path(__file__)),exit_code=0,neural_steps=80,body_model_load=0,mj_forward=0,mj_step=0,replays=1,mechanism_gate='FULL_NETWORK_TIMING_REFERENCE_MISSING',failures=[]));(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(checks));print('targets',len(nodes))
if __name__=='__main__':main()
