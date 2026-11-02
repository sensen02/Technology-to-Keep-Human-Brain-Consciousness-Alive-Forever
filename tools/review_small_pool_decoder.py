"""Independent saved-output preclip/physicalsupport review. No integrations."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s4-small-pool-decoder/20261102T064123Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
C=ROOT/'s4-bending-closedloop-training/20261102T042848Z'
O=ROOT/'s4-small-pool-parent-review/20261102T064123Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
 O.mkdir(parents=True,exist_ok=False);arm=S/'FIXED_K5_NEUTRAL'
 paths=[Path(__file__),S/'PREREGISTRATION.json',S/'FIXED_GAIN_VECTOR.npz',S/'POOL_GAIN_SCHEMA.json',S/'RESULT.json',arm/'ACTUAL_DECODER_CONFIG.json',arm/'PARAMETERS.npz',arm/'RASTER.npz',arm/'BODYLOSS_NATIVE.npz',arm/'ACTUAL_COMPILED_INITIAL.npz',arm/'ACTUAL_COMPILED_FINAL.npz',arm/'MATRIX_FINAL_HASHES.json',arm/'SELECTED_MUSCLE_MOMENT.npz',B/'RUN/MAPPING.json',C/'FROZEN_REFERENCE.json']
 hashes={str(p):sha(p) for p in paths};dump(O/'PREREGISTRATION.json',dict(scope='Postrun independent actualdecoder/roots/matrix/bodymetrics review,0modelintegration or optimization',source_sha256=hashes))
 mapping=json.loads((B/'RUN/MAPPING.json').read_text())['rows'];vector=arrays(S/'FIXED_GAIN_VECTOR.npz');n=np.array([len(r['local_indices']) for r in mapping]);gain=np.minimum(1,n/5.);assert np.array_equal(vector['actual_present_n'],n) and np.array_equal(vector['per_actuator_gain'],gain)
 config=json.loads((arm/'ACTUAL_DECODER_CONFIG.json').read_text());assert config==dict(filter_tau_s=.005,rate_for_full_control_hz=100.,max_control=1.,per_actuator_gain=gain.tolist())
 seen={}
 for row in mapping:
  key=tuple(row['local_indices']);i=row['actuator_index']
  if key:
   if key in seen:assert gain[i]==gain[seen[key]]
   else:seen[key]=i
 assert len(seen)==66 and (gain==0).sum()==16
 p=arrays(arm/'PARAMETERS.npz');assert not p['threshold_offset'].any() and not p['z'].any()
 final=json.loads((arm/'MATRIX_FINAL_HASHES.json').read_text())
 def csrhash(w):
  h=hashlib.sha256()
  for k in ['data','indices','indptr']:h.update(getattr(w,k).tobytes())
  return h.hexdigest()
 for key in ['We','Wi']:
  base=sparse.load_npz(B/('ACTUAL_'+key+'.npz'))
  for prefix in ['ACTUAL_','FINAL_']:
   a=sparse.load_npz(arm/(prefix+key+'.npz'));assert all(np.array_equal(x,y) for x,y in [(a.indptr,base.indptr),(a.indices,base.indices),(a.data,base.data)])
  assert csrhash(base)==final[key]
 ci=arrays(arm/'ACTUAL_COMPILED_INITIAL.npz');cf=arrays(arm/'ACTUAL_COMPILED_FINAL.npz');assert set(ci)==set(cf) and len(ci)==469 and all(np.array_equal(ci[k],cf[k],equal_nan=True) for k in ci)
 body=arrays(arm/'RUN/BODY.npz');old=arrays(C/'TRAIN_000_BASELINE/RUN/BODY.npz')
 for key in body:assert np.array_equal(body[key][:400],old[key][:400]),key
 assert not body['assist'][2600:].any() and not body['external'][2600:].any() and not body['warning'].any()
 records=json.loads((arm/'RUN/PREWARM_NEURAL_RECORDS.json').read_text())+json.loads((arm/'RUN/NEURAL_RECORDS.json').read_text());events=np.unpackbits(arrays(arm/'RASTER.npz')['packed'],axis=1,count=12000).astype(bool);assert len(events)==2300 and len(records)==1150
 rate=np.zeros(90);prev=np.zeros(90);decay=math.exp(-.0005/.005);maxrate=0.;saturation=[0,0]
 for tick,sp in enumerate(events):
  inst=np.array([math.fsum(float(sp[i]) for i in r['local_indices'])/len(r['local_indices'])/.0005 if r['local_indices'] else 0. for r in mapping]);rate=decay*rate+(1-decay)*inst
  if tick%2:
   window=tick//2;r=records[window];assert np.array_equal(prev,r['applied_command']);maxrate=max(maxrate,float(abs(rate-np.array(r['decoder']['rates_hz'])).max()));expected=np.clip((rate/100)*gain,0,1);assert abs(expected-np.array(r['decoded_next_command'])).max()<1e-12
   saturation[0 if window<1100 else 1]+=int((prev>=1).sum());prev=np.array(r['decoded_next_command'])
 assert maxrate<1e-12
 ref=json.loads((C/'FROZEN_REFERENCE.json').read_text());result=json.loads((S/'RESULT.json').read_text());metrics=[]
 for record,folder in zip(result['records'],[C/'TRAIN_000_BASELINE',arm]):
  a=arrays(folder/'BODYLOSS_NATIVE.npz')['rows'][1:4001];t=np.arange(1,4001)*.0000125;pos=a[:,7]/ref['weight']>1e-8;indices=np.flatnonzero(pos);first=None if not len(indices) else float(t[indices[0]]);foot=float(a[:,1:7].sum(1).mean()/ref['weight']);height=float(a[:,10].min()/ref['h0']);gates=dict(no_positive_nonfoot=first is None,mean_foot_load=foot>=.8,minimum_height=height>=.8)
  assert gates==record['support']['feasibility_gates'] and first==record['support']['first_nonfoot_s'] and not all(gates.values())
  assert not a[:,19].any();metrics.append(dict(label=record['label'],firstNF=first,positiveNF_endpoints=int(pos.sum()),meanfoot=foot,minheightfraction=height,motion_loss=record['motion_loss']))
 assert result['selection']['admitted_label'] is None
 force=arrays(arm/'SELECTED_MUSCLE_MOMENT.npz');forceerror=float(abs(force['force']-force['saved_force']).max());torque=np.einsum('tju,tu->tj',force['moment'],force['force']);assert forceerror<1e-10 and abs(torque-force['qfrc_actuator'][:,6:]).max()<1e-10
 out=dict(passed=True,source_sha256=hashes,all90gainformula_exact=True,samepoolbinding_preserved=True,unmapped_gainzero16=True,oldneutralgraph_andzerooffset_preserved=True,actualmatrixinitialfinalexact=True,actualcompiledinitialfinal469exact=True,old400bodyprefix_exact=True,all2300ticks90pool_preclipformula_max_rate_error=maxrate,new_applied_saturated=dict(prewarm=saturation[0],release50=saturation[1]),metrics=metrics,admitted_label=None,offlineforce_saved_max_error=forceerror,calls=dict(model_compile=0,body_step=0,network=0,neural_step=0,optimizer=0),limits='Onlyoneengineeringconfiguration.Fullbodyresponsechangesfeedback/history,not comparable tofixedoldspikesasaprediction. Reducedclipdoesnotestablishsupport or biologicalcorrectness.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',out);print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
