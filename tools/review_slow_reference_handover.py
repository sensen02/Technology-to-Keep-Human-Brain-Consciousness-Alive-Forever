"""Independent saved-output review of slower-reference physical handover; no steps."""
from pathlib import Path
import math,json,hashlib
import numpy as np
from scipy import sparse
ROOT=Path('/home/sensen/Desktop/engine修复验收')
S=ROOT/'s5-slow-reference-handover/20261102T165227Z'
B=ROOT/'s5-banc888-physical-handover/20261101T220018Z'
O=ROOT/'s5-slow-reference-parent-review/20261102T165227Z'
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 O.mkdir(parents=True,exist_ok=False);f=S/'SLOW_2P5HZ_PHASE_MATCHED'
 paths=[Path(__file__),S/'PREREGISTRATION.json',S/'RESULT.json',S/'MECHANISM.npz',S/'INDEPENDENT_ARRAY_VERIFICATION.json',f/'RASTER.npz',f/'NATIVE.npz',f/'RUN/BODY.npz',f/'RUNNER_ADAPTER.py',f/'PARAMETERS.npz',f/'ACTUAL_COMPILED_INITIAL.npz',f/'ACTUAL_COMPILED_FINAL.npz'];hashes={str(p):sha(p) for p in paths};dump(O/'PREREGISTRATION.json',dict(scope='Independent completedtrialtime/decoder/mechanismformula/physicalsupport audit,0integration',source_sha256=hashes))
 result=json.loads((S/'RESULT.json').read_text());assert json.loads((S/'INDEPENDENT_ARRAY_VERIFICATION.json').read_text())['passed']
 source=(f/'RUNNER_ADAPTER.py').read_text();assert source.count('*2*np.pi*2.5')==4 and '*2*np.pi*5' not in source and 'frequency_hz=2.5' in source
 pr=json.loads((S/'PREREGISTRATION.json').read_text());assert pr['config']['duration']==1.25 and pr['config']['neural_prewarm_s']==1.05 and pr['release_s']==.1 and not pr['plasticity']
 body=arrays(f/'RUN/BODY.npz');old=arrays(B/'RUN/BODY.npz')
 for key in body:assert np.array_equal(body[key][:301],old[key][:301]),key
 assert not body['warning'].any() and not body['external'].any() and not body['assist'][:,:6].any()
 native=arrays(f/'NATIVE.npz');rows=native['rows'];assert rows.shape==(92002,69) and native['joint_velocity'].shape==(92002,42)
 release=rows[:,26]==2;assert release.sum()==8001 and not rows[release,21].any() and not rows[release,27:].any() and not rows[:,24:26].any()
 params=arrays(f/'PARAMETERS.npz');assert not params['z'].any() and not params['threshold_offset'].any() and np.array_equal(params['root_ids'],arrays(B/'IDENTITIES.npz')['root_ids'])
 for key in ['We','Wi']:
  base=sparse.load_npz(B/('ACTUAL_'+key+'.npz'))
  for prefix in ['ACTUAL_','FINAL_']:
   w=sparse.load_npz(f/(prefix+key+'.npz'));assert all(np.array_equal(x,y) for x,y in [(w.data,base.data),(w.indices,base.indices),(w.indptr,base.indptr)])
 ci=arrays(f/'ACTUAL_COMPILED_INITIAL.npz');cf=arrays(f/'ACTUAL_COMPILED_FINAL.npz');assert len(ci)==469 and set(ci)==set(cf) and all(np.array_equal(ci[k],cf[k],equal_nan=True) for k in ci)
 events=np.unpackbits(arrays(f/'RASTER.npz')['packed'],axis=1,count=12000).astype(bool);assert events.shape==(2300,12000)
 records=json.loads((f/'RUN/PREWARM_NEURAL_RECORDS.json').read_text())+json.loads((f/'RUN/NEURAL_RECORDS.json').read_text());assert len(records)==1150
 mapping=json.loads((B/'RUN/MAPPING.json').read_text())['rows'];rate=np.zeros(90);pending=np.zeros(90);decay=math.exp(-.0005/.005);error=0.
 for tick,sp in enumerate(events):
  rate=decay*rate+(1-decay)*np.array([math.fsum(float(sp[i]) for i in r['local_indices'])/len(r['local_indices'])/.0005 if r['local_indices'] else 0. for r in mapping])
  if tick%2:
   record=records[tick//2];assert np.array_equal(pending,record['applied_command']);error=max(error,float(abs(rate-record['decoder']['rates_hz']).max()));expected=np.clip(rate/100,0,1);assert abs(expected-record['decoded_next_command']).max()<1e-12;pending=np.array(record['decoded_next_command'])
 assert error<1e-12
 warm=arrays(f/'PREWARM_STATE.npz');entry=arrays(f/'RELEASE_ENTRY_STATE.npz');assert all(np.array_equal(warm[k],entry[k]) for k in warm)
 initial=arrays(f/'NETWORK_INITIAL.npz');assert np.all(initial['v']==-65) and all(not initial[k].any() for k in ['ge','gi','adaptation','refractory_until_ms','spike_count','_ring','decoder_rates','pending'])
 for role,ticks in [('PREWARM_STATE',2100),('RELEASE_STATE',2300)]:
  state=arrays(f/(role+'.npz'));assert not state['threshold_offset_mV'].any() and np.array_equal(state['spike_count'],events[:ticks].sum(0));ring=np.zeros_like(state['_ring']);position=0
  for ev in events[:ticks]:ring[position]=ev;position=(position+1)%len(ring)
  assert np.array_equal(ring,state['_ring']) and position==int(state['ring_pos'])
 before=arrays(f/'RUN/HANDOVER_BEFORE.npz');after=arrays(f/'RUN/HANDOVER_AFTER.npz')
 for key in before:
  if key not in ['ctrl','qfrc_applied','xfrc_applied']:assert np.array_equal(before[key],after[key]),key
 assert not after['qfrc_applied'].any() and not after['xfrc_applied'].any() and np.array_equal(after['ctrl'],warm['pending'])
 W=result['weight'];h0=result['h0'];r=rows[release][1:];assert len(r)==8000;nf=r[:,7]>W*1e-8;idx=np.flatnonzero(nf);first=None if not len(idx) else float(r[idx[0],0]-1.25);mean=float(r[:,1:7].sum(1).mean()/W);height=float(r[:,12].min()/h0);gates=dict(no_nonfoot=first is None,mean_foot=mean>=.8,min_height=height>=.8);boundaryNF=bool(rows[release][0,7]>W*1e-8)
 saved=result['candidate'];assert gates==saved['gates'] and first==saved['first_positive_nonfoot_s'] and abs(mean-saved['mean_foot_weight'])<1e-12 and abs(height-saved['min_height_ratio'])<1e-12 and boundaryNF==saved['boundary_already_NF']
 assert saved['admission_passed']==(all(gates.values()) and not boundaryNF)
 mech=arrays(S/'MECHANISM.npz');assert np.array_equal(mech['source_indices'],np.arange(1700,2501,8)) and np.array_equal(mech['baseline_AUDIT_indices'],np.arange(0,401,4))
 assert np.max(abs(np.exp(1j*mech['phase_rad'])-np.exp(1j*mech['baseline_phase_rad'])))<1e-9
 force_error=float(abs(mech['force']-mech['muscle_force']).max());assert force_error==0
 maxformula=0.;counts=dict(FLzero=0,FVzero=0,relativecrit=0)
 for t in range(101):
  for u in range(90):
   p=ci['actuator_gainprm'][u];r0,r1,F,scale,lo,hi,vmax,fp,fvmax=map(float,p[:9]);lr=ci['actuator_lengthrange'][u];L0=(lr[1]-lr[0])/(r1-r0);L=r0+(mech['length'][t,u]-lr[0])/L0;V=mech['velocity'][t,u]/(L0*vmax);left=(lo+1)/2;right=(1+hi)/2
   if L<lo or L>hi:fl=0.
   elif L<=left:fl=.5*((L-lo)/(left-lo))**2
   elif L<=1:fl=1-.5*((1-L)/(1-left))**2
   elif L<=right:fl=1-.5*((L-1)/(right-1))**2
   else:fl=.5*((hi-L)/(hi-right))**2
   y=fvmax-1;fv=0. if V<=-1 else (1+V)**2 if V<=0 else fvmax-(y-V)**2/y if V<=y else fvmax
   F=scale/ci['actuator_acc0'][u] if F<0 else F;gain=-F*fl*fv;stationary=-F*fl;maxformula=max(maxformula,abs(gain-mech['gain'][t,u]),abs(stationary-mech['gain_zero_velocity'][t,u]));valid=abs(stationary)>1e-12
   counts['FLzero']+=int(not valid);counts['FVzero']+=int(valid and abs(gain)<=1e-12);counts['relativecrit']+=int(valid and abs(gain)<=.001*abs(stationary))
 assert maxformula<1e-9
 for key in counts:assert counts[key]==result['mechanism_candidate'][key]
 out=dict(passed=True,source_sha256=hashes,source_all4_frequencyfactors_andmetadata_exact=True,phase101_matched=True,neural_history_duration_difference_s=.05,original301BODYprefix_exact=True,zeroassistrelease_and_originalweights_exact=True,samemodel469initialfinal_exact=True,decoder2300ticks90pool_maxerror=error,native_endpoint_support=dict(firstNF_s=first,meanfoot=mean,minheight=height,boundaryNF=boundaryNF,gates=gates,admission_passed=saved['admission_passed']),native_gain_formula_max_error=maxformula,savedforce_error=force_error,mechanism_candidate=counts,mechanism_baseline=result['mechanism_baseline'],calls=dict(model_compile=0,forward=0,body_step=0,neural_step=0,training=0),limits='Frequencyandsamedterminalphasecost50msprewarmdifference,notexactneuralhistorypair. LowerFVzerosdoesnotimplysupport. NoSTDP,newanatomyornewphysiologicalcalibration.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',out);print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
