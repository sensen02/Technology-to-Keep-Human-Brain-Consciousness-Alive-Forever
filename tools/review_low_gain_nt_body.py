"""Independent saved body/matrix/native-force audit; never re-integrates physics."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收')
P=ROOT/'s5-low-gain-nt-physical/20261102T024247Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
G=ROOT/'s3-sourced-nt-candidate-graph/20261102T004326Z'
O=ROOT/'s5-low-gain-nt-body-review/20261102T024247Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def aggregate(folder,weight):
 foot=nonfoot=0.;count=0;nfcount=0;positivecount=0;first=None;first_candidate=None;minz=float('inf');maxfoot=-float('inf');last=None
 for record in json.loads((folder/'CHUNKS.json').read_text()):
  z=np.load(record['file'],allow_pickle=False);mask=(z['kind']==0)&(z['step']>104000)&(z['step']<=160000);steps=z['step'][mask];n=len(steps)
  if not n:continue
  fz=z['foot_force'][mask,:,2].sum(1);nf=z['nonfoot_force'][mask,2];candidate=z['nonfoot_count'][mask]>0
  foot+=float(fz.sum());nonfoot+=float(np.maximum(nf,0).sum());nfcount+=int(candidate.sum());positivecount+=int((nf>0).sum());count+=n;minz=min(minz,float(z['com'][mask,2].min()));maxfoot=max(maxfoot,float(fz.max()))
  hits=steps[nf>0];hitsc=steps[candidate]
  if len(hits):first=int(hits[0]) if first is None else min(first,int(hits[0]))
  if len(hitsc):first_candidate=int(hitsc[0]) if first_candidate is None else min(first_candidate,int(hitsc[0]))
  assert not z['assist_root_max'][mask].any() and not z['assist_leg_norm'][mask].any() and not z['external_max'][mask].any()
  last=dict(COMz=float(z['com'][mask][-1,2]),foot_Fz_W=float(fz[-1]/weight))
 assert count==56000
 return dict(endpoint_bins=count,first_nonfoot_candidate_s=None if first_candidate is None else (first_candidate-104000)*.0000125,first_positive_aggregate_nonfoot_s=None if first is None else (first-104000)*.0000125,nonfoot_candidate_bins=nfcount,positive_aggregate_nonfoot_bins=positivecount,foot_Fz_W_mean=foot/count/weight,positive_aggregate_nonfoot_Fz_W_mean=nonfoot/count/weight,min_COMz=minz,max_foot_Fz_W=maxfoot/weight,last=last)
def main():
 O.mkdir(parents=True,exist_ok=True)
 assert not (O/'RESULT.json').exists(), 'Never overwrite a completed review'
 paths=[Path(__file__),P/'RUN/BODY.npz',B/'RUN/BODY.npz',P/'RUN/GAINS.json',P/'RUN/MODEL.xml',P/'ACTUAL_RUNTIME_AUDIT.json',P/'CHUNKS.json',B/'CHUNKS.json',G/'ADDED_We.npz']
 paths += [P/f'{prefix}{kind}.npz' for prefix in ['BASE_','CANDIDATE_','FINAL_'] for kind in ['We','Wi']]
 paths += [P/'COMPILED_0.npz',P/'COMPILED_1.npz',ROOT/'s4-banc888-closedloop-parameter-fit/20261101T222509Z/COMPILED_MODEL_ARRAYS.npz']
 identities={str(p):sha(p) for p in paths}
 dump(O/('COMPILE_STAGE_CORRECTION.json' if (O/'PREREGISTRATION.json').exists() else 'PREREGISTRATION.json'),dict(task='Independent post-run arrays/matrix/nativeforce review only; no model or neural integration',source_sha256=identities,comparators='Original neutral round13 and unique lowgaincandidate, equal native56000endpoints/.5ms1400bins'))
 assert sha(P/'RUN/MODEL.xml')=='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
 for k in ['We','Wi']:
  base=sparse.load_npz(P/f'BASE_{k}.npz');expected=sparse.load_npz(B/f'ACTUAL_{k}.npz');add=sparse.load_npz(G/f'ADDED_{k}.npz')
  assert all(np.array_equal(a,b) for a,b in [(base.indptr,expected.indptr),(base.indices,expected.indices),(base.data,expected.data)])
  expected=(expected+.03125*add).tocsr()
  for prefix in ['CANDIDATE_','FINAL_']:
   actual=sparse.load_npz(P/f'{prefix}{k}.npz');assert all(np.array_equal(a,b) for a,b in [(actual.indptr,expected.indptr),(actual.indices,expected.indices),(actual.data,expected.data)])
 new=np.load(P/'RUN/BODY.npz');old=np.load(B/'RUN/BODY.npz');assert len(new['time'])==4001
 for k in old.files:assert np.array_equal(new[k][:400],old[k][:400]),k
 assert not new['assist'][2600:].any() and not new['external'][2600:].any() and not new['warning'].any()
 W=float(json.loads((P/'RUN/GAINS.json').read_text())['weight']);metrics={}
 for label,folder,z in [('candidate',P,new),('neutral',B,old)]:
  native=aggregate(folder,W);idx=np.arange(2601,4001);nf=z['nonfoot_count'][idx]>0
  shared=dict(endpoint_bins=1400,nonfoot_bins=int(nf.sum()),first_nonfoot_sample_s=(float(z['time'][idx[np.flatnonzero(nf)[0]]])-1.3) if nf.any() else None,mean_foot_Fz_W=float(z['foot_force'][idx,:,2].sum(1).mean()/W),last_COMz=float(z['com'][-1,2]))
  metrics[label]=dict(native=native,shared=shared)
 compiled=np.load(P/'COMPILED_1.npz');preparation=np.load(P/'COMPILED_0.npz');reference_path=ROOT/'s4-banc888-closedloop-parameter-fit/20261101T222509Z/COMPILED_MODEL_ARRAYS.npz';reference=np.load(reference_path)
 assert set(reference.files)-set(compiled.files)=={'gravity'} and set(compiled.files)<=set(reference.files)
 assert all(np.array_equal(compiled[k],reference[k],equal_nan=True) for k in compiled.files)
 gravity=__import__('xml.etree.ElementTree',fromlist=['parse']).parse(P/'RUN/MODEL.xml').getroot().find('option').get('gravity')
 assert np.array_equal(np.fromstring(gravity,sep=' '),reference['gravity'])
 stage_differences=[k for k in compiled.files if not np.array_equal(compiled[k],preparation[k],equal_nan=True)]
 count=json.loads((P/'ACTUAL_RUNTIME_AUDIT.json').read_text());assert count['base_steps']==0 and count['active_steps']==3600 and count['factory_calls']==1
 result=dict(passed=True,source_sha256=identities,actual_matrix_constant_from_first_prewarm=True,original_body_prefix_all400frames_exact=True,actual_compiled_arrays_equal_original_reference=len(compiled.files),preparation_vs_actual_expected_different_arrays=stage_differences,post_run_compiled_model_capture_available=False,same_active_network_continuous=True,metrics=metrics,calls=dict(model_compile=0,network=0,body_step=0,neural_step=0,training=0),limits='Positive aggregate nonfoot force is not sum of positive per-contact forces. This reviewer reconstructs recorded force statistics, not a second physics run. Support/walking require physical success, not engineering pass.')
 assert all(sha(Path(p))==h for p,h in identities.items());dump(O/'RESULT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
