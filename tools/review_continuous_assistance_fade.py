"""Independent raw/scaled torque/phase/state and support review; no integration."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s5-continuous-assistance-fade/20261102T074033Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
C=ROOT/'s4-bending-closedloop-training/20261102T042848Z'
O=ROOT/'s5-fade-parent-review/20261102T074033Z'
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);f=S/'CONTINUOUS_FADE_200MS'
 paths=[Path(__file__),S/'PREREGISTRATION.json',S/'RESULT.json',S/'INDEPENDENT_ARRAY_VERIFICATION.json',f/'NATIVE.npz',f/'ASSIST_RAW_AND_SCALED.npz',f/'ACTUAL_ASSIST_CALLBACK.npz',f/'RASTER.npz',f/'RUN/BODY.npz',f/'RUN/MODEL.xml',f/'BOUNDARY_CLOCKS.json',B/'RUN/BODY.npz',B/'NEURAL_RASTER.npz']
 hashes={str(p):sha(p) for p in paths};dump(O/'PREREGISTRATION.json',dict(scope='Postrun rawtorque/time/force/decoder review,0models or integration',source_sha256=hashes,autonomy='Only1.3..1.4solequalifies;fadeendnonfoot disqualifies evenifafterbodyprone.'))
 result=json.loads((S/'RESULT.json').read_text());assert json.loads((S/'INDEPENDENT_ARRAY_VERIFICATION.json').read_text())['passed']
 native=arrays(f/'NATIVE.npz')['rows'];raw=arrays(f/'ASSIST_RAW_AND_SCALED.npz');tap=arrays(f/'ACTUAL_ASSIST_CALLBACK.npz')['rows'];assert native.shape==(32002,69) and tap.shape==(24001,46)
 expected=np.clip((1.3-raw['clock'])/.2,0,1);assert np.array_equal(raw['multiplier'],expected) and np.array_equal(raw['scaled'],raw['raw_clipped']*expected[:,None]) and np.array_equal(raw['scaled'],tap[:,2:44])
 assisted=native[:,26]<2;idx=np.rint((native[assisted,22]-tap[0,0])/.0000125).astype(int);assert np.array_equal(native[assisted,27:],tap[idx,2:44]) and np.array_equal(native[assisted,23],tap[idx,1])
 assert not native[:,24:26].any();released=native[:,26]==2;assert not native[released,21].any() and not native[released,27:].any()
 body=arrays(f/'RUN/BODY.npz');base=arrays(B/'RUN/BODY.npz');prefix=np.flatnonzero(np.rint(body['time']/.0000125)<=88000)
 for key in body:assert np.array_equal(body[key][prefix],base[key][prefix]),key
 raster=np.unpackbits(arrays(f/'RASTER.npz')['packed'],axis=1,count=12000).astype(bool);assert raster.shape==(2400,12000) and np.array_equal(raster[:1800],arrays(B/'NEURAL_RASTER.npz')['spikes'][:1800])
 records=json.loads((f/'RUN/PREWARM_NEURAL_RECORDS.json').read_text())+json.loads((f/'RUN/NEURAL_RECORDS.json').read_text());mapping=json.loads((B/'RUN/MAPPING.json').read_text())['rows'];rate=np.zeros(90);prev=np.zeros(90);decay=math.exp(-.0005/.005);error=0.
 for tick,event in enumerate(raster):
  rate=decay*rate+(1-decay)*np.array([math.fsum(float(event[i]) for i in row['local_indices'])/len(row['local_indices'])/.0005 if row['local_indices'] else 0 for row in mapping])
  if tick%2:
   r=records[tick//2];assert np.array_equal(prev,r['applied_command']);error=max(error,float(abs(rate-r['decoder']['rates_hz']).max()));assert abs(np.clip(rate/100,0,1)-r['decoded_next_command']).max()<1e-12;prev=np.array(r['decoded_next_command'])
 assert error<1e-12
 warm=arrays(f/'PREWARM_STATE.npz');entry=arrays(f/'RELEASE_ENTRY_STATE.npz');assert all(np.array_equal(warm[k],entry[k]) for k in warm)
 for key in ['We','Wi']:
  basew=sparse.load_npz(B/('ACTUAL_'+key+'.npz'))
  for prefix in ['ACTUAL_','FINAL_']:
   w=sparse.load_npz(f/(prefix+key+'.npz'));assert all(np.array_equal(a,b) for a,b in [(w.data,basew.data),(w.indices,basew.indices),(w.indptr,basew.indptr)])
 ci=arrays(f/'ACTUAL_COMPILED_INITIAL.npz');cf=arrays(f/'ACTUAL_COMPILED_FINAL.npz');assert len(ci)==469 and set(ci)==set(cf) and all(np.array_equal(ci[k],cf[k],equal_nan=True) for k in ci)
 W=result['weight'];release=native[released&(np.rint(native[:,0]/.0000125)>104000)];assert len(release)==8000;hits=np.flatnonzero(release[:,7]>1e-8*W);first=None if not len(hits) else float(release[hits[0],0]-1.3);mean=float(release[:,1:7].sum(1).mean()/W);minh=float(release[:,12].min()/result['h0']);checks=dict(no_nonfoot_gate=first is None,foot_gate=mean>=.8,height_gate=minh>=.8)
 saved=result['continuous_fade'];assert first==saved['first_positive_nonfoot_s'] and abs(mean-saved['mean_foot_weight'])<1e-12 and abs(minh-saved['min_height_ratio'])<1e-12
 assert all(checks[k]==saved[k] for k in checks)
 fade=native[native[:,26]==1];end=fade[np.rint(fade[:,0]/.0000125)==104000][0];fade_end_bad=bool(end[7]>1e-8*W);assert fade_end_bad==result['fade']['end_nonfoot_admission_failure']
 allowed=all(checks.values()) and not fade_end_bad;assert allowed==saved['admission_passed']
 out=dict(passed=True,source_sha256=hashes,actualraw_times_multiplier_exact=True,actualpoststep_assist_callbackclock_exact=True,noassist_release_exact=True,original_beforefade_BODY_and_raster_exact=True,compiled469_sameobject_initialfinal_exact=True,originalgraph_and_neuralstate_continuity_exact=True,decoder_max_error=error,fade_first_events=result['fade']['first_events'],fade_end_positive_nonfoot=fade_end_bad,release_support=dict(first_nonfoot_s=first,meanfoot=mean,minheightfraction=minh,gates=checks,admission_passed=allowed),calls=dict(model_compile=0,network=0,body_step=0,neural_step=0,training=0),limits='Read-only evidence review,not second bodyrun. Fadeisassisted andmayalreadyfail;stableproneafterrelease cannot qualify. No STDP or plasticity enabled.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',out);print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
