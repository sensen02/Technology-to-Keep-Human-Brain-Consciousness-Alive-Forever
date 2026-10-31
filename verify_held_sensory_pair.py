"""S164 independent algebraic audit of recorded pair; never calls neural.step."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.sparse import csr_matrix
REF=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z/init80ms')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',required=True,type=Path);p=pa.parse_args().output;i=load(REF/'INITIALIZATION.npz');m=json.loads((REF/'NEURAL_RELEASE_META.json').read_text());cfg=m['neural_configs']['p'];con=load(REF/'CONNECTIVITY.npz');W={k:csr_matrix((con[k+'_data'],con[k+'_indices'],con[k+'_indptr']),shape=tuple(con[k+'_shape'])) for k in ['We','Wi']};keys=['v','ge','gi','adaptation','refractory_until_ms','spike_count','_ring'];allarms={};errs={}
 for arm in ['A','B']:
  x=load(p/f'{arm}_STATES.npz');allarms[arm]=x
  assert len(x['input'])==41
  for k in keys:assert np.array_equal(x['pre_'+k][0],i[k]),k
  assert x['pre_t_ms'][0]==80 and x['pre_steps'][0]==160 and x['pre__ring_i'][0]==0
  h=i['history'].copy();ap=load(REF/'TRAJECTORY.npz')['actuator_pool'];rows={}
  for tick in range(41):
   for k in keys:
    if tick:assert np.array_equal(x['pre_'+k][tick],x['post_'+k][tick-1]),k
   pre=lambda k:x['pre_'+k][tick];post=lambda k:x['post_'+k][tick];tm=x['pre_t_ms'][tick];end=x['post_t_ms'][tick];ri=int(x['pre__ring_i'][tick]);assert tm==80+tick*.5 and end==80+(tick+1)*.5
   arrivals=pre('_ring')[ri];assert np.array_equal(arrivals,x['arrivals'][tick]);ge=pre('ge')+W['We']@arrivals;gi=pre('gi')+W['Wi']@arrivals;active=tm+1e-10>=pre('refractory_until_ms');assert np.array_equal(active,x['active'][tick]);candidate=(cfg['capacitance_nF']/.5*pre('v')+cfg['leak_uS']*cfg['rest_mV']+ge*cfg['exc_reversal_mV']+gi*cfg['inh_reversal_mV']+x['input'][tick]-pre('adaptation'))/(cfg['capacitance_nF']/.5+cfg['leak_uS']+ge+gi);assert abs(candidate-x['candidate_v_BE'][tick]).max()<1e-12
   v=np.where(active,candidate,cfg['reset_mV']);spike=active&(v>=cfg['threshold_mV']);assert np.array_equal(spike,x['spikes'][tick]);v[spike]=cfg['reset_mV'];refr=pre('refractory_until_ms').copy();refr[spike]=end+cfg['refractory_ms'];adapt=pre('adaptation')*np.exp(-.5/cfg['adaptation_tau_ms']);adapt[spike]+=cfg['adaptation_increment_nA'];ring=pre('_ring').copy();ring[ri]=spike
   expected=dict(v=v,ge=ge*np.exp(-.5/cfg['exc_tau_ms']),gi=gi*np.exp(-.5/cfg['inh_tau_ms']),adaptation=adapt,refractory_until_ms=refr,spike_count=pre('spike_count')+spike,_ring=ring)
   for k,val in expected.items():assert np.allclose(val,post(k),rtol=0,atol=1e-12),k
   assert x['post__ring_i'][tick]==(ri+1)%5
  # Independent FIR from saved actual events, retaining PRE history, never fake spikes.
  import csv
  poolrows=list(csv.DictReader(Path('/home/sensen/Desktop/engine修复验收/s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_66_POOL_MEMBERSHIP.csv').open()));pools=[list(map(int,r['local_indices'].split(';'))) for r in poolrows]
  for tick in range(41):
   activity=np.array([x['spikes'][tick,ids].mean()/.0005/100 for ids in pools]);h=np.concatenate((activity[None,:],h),axis=0)[:80];features=np.stack([h[lo:hi].mean(0) for lo,hi in [(0,5),(5,10),(10,20),(20,40),(40,80)]],axis=1);unclipped=(features*i['weights']).sum(1);decoded=np.zeros(90);mapped=ap>=0;decoded[mapped]=np.clip(unclipped,0,1)[ap[mapped]]
   for got,expected in [(x['history'][tick],h),(x['features'][tick],features),(x['unclipped_pool'][tick],unclipped),(x['decoded'][tick],decoded),(x['applied'][tick],.5*decoded)]:assert abs(got-expected).max()<1e-14
  errs[arm]=dict(ticks=41,all_recorded_state_recurrences=True,ring_arrival_and_spike_timing=True,FIR_and_gain_independently_reconstructed=True)
 a,b=allarms['A'],allarms['B']
 for k in a:assert np.array_equal(a[k][0],b[k][0]),k
 assert np.array_equal(a['input'],np.repeat(i['current'][None,:],41,axis=0));assert np.array_equal(b['input'][0],i['current']);assert np.array_equal(b['input'][1:],np.repeat(b['input'][1:2],40,axis=0))
 result=dict(passed=True,arms=errs,actual_new_neural_ticks=82,new_body_calls=0,complete_restore_and_shared_tick_equal=True,held_inputs_verified=True,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(p/'PARENT_STATE_RECURRENCE_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
