"""Independent group/threshold/support/force review, no dynamics or optimizer."""
from pathlib import Path
import json,hashlib,math,csv
import numpy as np
ROOT=Path('/home/sensen/Desktop/engine修复验收')
P=ROOT/'s4-antagonist-mn-support/20261102T061756Z'
S=ROOT/'s4-bending-closedloop-training/20261102T042848Z'
G=ROOT/'s4-antagonist-pool-coupling/20261102T061756Z'
O=ROOT/'s4-antagonist-support-parent-review/20261102T061756Z'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arrays(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
 O.mkdir(parents=True,exist_ok=False)
 arm=P/'ARM_LF_tibia_pair';paths=[Path(__file__),P/'PREREGISTRATION.json',P/'RESULT.json',P/'ADJUDICATED_PLAN.json',P/'STATIC_POOL_SIGN_AND_OVERLAP.json',G/'RESULT.json',arm/'PARAMETERS.npz',arm/'BODYLOSS_NATIVE.npz',arm/'GEOMETRY_SAMPLES.npz',arm/'RASTER.npz',arm/'MATRIX_FINAL_HASHES.json',arm/'ACTUAL_COMPILED_INITIAL.npz',arm/'ACTUAL_COMPILED_FINAL.npz']
 paths += [arm/'SELECTED_MUSCLE_MOMENT.npz',ROOT/'s2-early-muscle-support/20261102T053639Z/TRAIN_000_BASELINE_AUDIT.npz']
 hashes={str(p):sha(p) for p in paths};dump(O/'PREREGISTRATION.json',dict(scope='Independent pool/threshold/physicalresponse arithmetic,not refit or new integration',source_sha256=hashes,checks='Traceable signedpools/sevenrootthresholds/matrixunchangedinitialfinal/actualcompiledsameobjectcapture/allfilter/earlyfootmetrics/strictsupportnoadmission.'))
 report=json.loads((P/'RESULT.json').read_text());plans=json.loads((P/'ADJUDICATED_PLAN.json').read_text())['plans'];chosen=[p for p in plans if p['valid']];assert len(chosen)==1 and chosen[0]['label']=='ARM_LF_tibia_pair'
 independent=json.loads((G/'RESULT.json').read_text());pair=independent['targets']['joint_LFTibia_pitch'];assert pair['strongest_positive']['pool_index']==10 and pair['strongest_negative']['pool_index']==11 and pair['antagonist_MN_overlap']==[]
 assert independent['targets']['joint_LMCoxa_yaw']['strongest_positive'] is None
 params=arrays(arm/'PARAMETERS.npz');expected=np.zeros(12000);expected[[2955,3271,3707,6343,11441]]=-1;expected[[2756,10765]]=1
 assert np.array_equal(params['threshold_offset'],expected) and np.array_equal(params['theta_mV'],expected) and not params['source_offset'].any()
 rootref=arrays(ROOT/'s5-banc888-physical-handover/20261101T220018Z/IDENTITIES.npz')['root_ids'];assert np.array_equal(params['root_ids'],rootref)
 from scipy import sparse
 def csrhash(w):
  h=hashlib.sha256()
  for k in ['data','indices','indptr']:h.update(getattr(w,k).tobytes())
  return h.hexdigest()
 final=json.loads((arm/'MATRIX_FINAL_HASHES.json').read_text())
 for key in ['We','Wi']:
  w=sparse.load_npz(arm/('ACTUAL_'+key+'.npz'));base=sparse.load_npz(ROOT/('s5-banc888-physical-handover/20261101T220018Z/ACTUAL_'+key+'.npz'))
  assert np.array_equal(w.indptr,base.indptr) and np.array_equal(w.indices,base.indices) and np.array_equal(w.data,base.data) and csrhash(w)==final[key]
  actualfinal=sparse.load_npz(arm/('FINAL_'+key+'.npz'));assert all(np.array_equal(a,b) for a,b in [(w.indptr,actualfinal.indptr),(w.indices,actualfinal.indices),(w.data,actualfinal.data)])
 initial=arrays(arm/'ACTUAL_COMPILED_INITIAL.npz');end=arrays(arm/'ACTUAL_COMPILED_FINAL.npz');assert set(initial)==set(end) and len(initial)==469 and all(np.array_equal(initial[k],end[k],equal_nan=True) for k in initial)
 baseline=arrays(S/'TRAIN_000_BASELINE/RUN/BODY.npz');body=arrays(arm/'RUN/BODY.npz')
 for key in body:assert np.array_equal(body[key][:400],baseline[key][:400]),key
 assert not body['assist'][2600:].any() and not body['external'][2600:].any() and not body['warning'].any()
 roster=json.loads((ROOT/'s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json').read_text())['rows'];raster=arrays(arm/'RASTER.npz');events=np.unpackbits(raster['packed'],axis=1,count=12000).astype(bool);assert events.shape==(2300,12000)
 records=json.loads((arm/'RUN/PREWARM_NEURAL_RECORDS.json').read_text())+json.loads((arm/'RUN/NEURAL_RECORDS.json').read_text());assert len(records)==1150
 rate=np.zeros(90);decay=math.exp(-.0005/.005);filtererror=0.;prev=np.zeros(90)
 for tick,event in enumerate(events):
  instant=np.array([math.fsum(float(event[i]) for i in row['local_indices'])/len(row['local_indices'])/.0005 if row['local_indices'] else 0. for row in roster]);rate=decay*rate+(1-decay)*instant
  if tick%2:
   r=records[tick//2];filtererror=max(filtererror,float(abs(rate-np.asarray(r['decoder']['rates_hz'])).max()));assert np.array_equal(prev,r['applied_command']);pending=np.clip(rate/100,0,1);assert abs(pending-np.asarray(r['decoded_next_command'])).max()<1e-12;prev=np.array(r['decoded_next_command'])
 assert filtererror<1e-12
 for name,ticks in [('PREWARM',2200),('RELEASE',2300)]:
  state=arrays(arm/(name+'_STATE.npz'));assert np.array_equal(state['threshold_offset_mV'],expected) and np.array_equal(state['spike_count'],events[:ticks].sum(0))
 rawinit=arrays(arm/'NETWORK_INITIAL.npz');assert not rawinit['_ring'].any() and not rawinit['spike_count'].any() and np.array_equal(rawinit['threshold_offset_mV'],expected)
 ref=json.loads((S/'FROZEN_REFERENCE.json').read_text());metrics={}
 for label,folder in [('BASELINE_CACHE',S/'TRAIN_000_BASELINE'),('ARM_LF_tibia_pair',arm)]:
  rows=arrays(folder/'BODYLOSS_NATIVE.npz')['rows'][1:4001];assert rows.shape==(4000,20);assert not rows[:,19].any();time=np.arange(1,4001)*.0000125;nf=rows[:,7]/ref['weight']>1e-8;idx=np.flatnonzero(nf);first=None if not len(idx) else float(time[idx[0]])
  meanfoot=float(rows[:,1:7].sum(1).mean()/ref['weight']);minheight=float(rows[:,10].min()/ref['h0']);gates=dict(no_positive_nonfoot=first is None,mean_foot_load=meanfoot>=.8,minimum_height=minheight>=.8)
  saved=next(r for r in report['records'] if r['label']==label);assert gates==saved['support']['feasibility_gates'] and first==saved['support']['first_nonfoot_s'] and not all(gates.values())
  metrics[label]=dict(first_nonfoot_over_deadband_s=first,positiveNF_endpoints=int(nf.sum()),meanfoot=meanfoot,meanpositiveNF=float(rows[:,7].mean()/ref['weight']),minheightfraction=minheight,motion_loss=saved['motion_loss'])
 assert report['selection']['admitted_label'] is None
 # Compare actual releases and selected pools, separate full prewarm history from release.
 oldraster=arrays(S/'TRAIN_000_BASELINE/RASTER.npz');old=np.unpackbits(oldraster['packed'],axis=1,count=12000)[:2300].astype(bool)
 poolcounts=[]
 for row in [pair['strongest_positive'],pair['strongest_negative']]:
  ids=row['MN_local_indices'];counts={}
  for window,lo,hi in [('prewarm',0,2200),('release0_18ms',2200,2236),('release50ms',2200,2300)]:counts[window]=dict(baseline=int(old[lo:hi][:,ids].sum()),intervention=int(events[lo:hi][:,ids].sum()))
  poolcounts.append(dict(pool=row['pool_index'],muscles=row['muscles'],counts=counts))
 force=arrays(arm/'SELECTED_MUSCLE_MOMENT.npz');force_error=float(abs(force['force']-force['saved_force']).max());assert force_error<1e-10
 torque=np.einsum('tju,tu->tj',force['moment'],force['force']);assert abs(torque-force['qfrc_actuator'][:,6:]).max()<1e-10
 j=list(force['joint_names']).index('joint_LFTibia_pitch');mapping=json.loads((ROOT/'s5-banc888-physical-handover/20261101T220018Z/RUN/MAPPING.json').read_text())['rows'];actnames=[r['actuator'] for r in mapping];positive=actnames.index('LFTibia_flex_93434');negative=actnames.index('LFTibia_extensor_93932')
 selectedforce=[]
 oldaudit=arrays(ROOT/'s2-early-muscle-support/20261102T053639Z/TRAIN_000_BASELINE_AUDIT.npz')
 for t,when in enumerate(force['time']):
  relative=float(when-1.3);indices=np.flatnonzero(abs(oldaudit['relative_time_s']-relative)<1e-10);oldindex=int(indices[0]) if len(indices) else None
  selectedforce.append(dict(relative_s=relative,candidate_joint_muscle_torque=float(torque[t,j]),baseline_joint_muscle_torque=None if oldindex is None else float(oldaudit['qfrc_actuator'][oldindex,j+6]),positive_muscle_activation=float(force['activation'][t,positive]),negative_muscle_activation=float(force['activation'][t,negative]),positive_active_contribution=float(force['moment'][t,j,positive]*force['active_force'][t,positive]),negative_active_contribution=float(force['moment'][t,j,negative]*force['active_force'][t,negative])))
 dump(O/'SELECTED_JOINT_FORCE_RESPONSE.json',dict(records=selectedforce,force_saved_maxerror=force_error,scope='Offlineforwardreconstruction is samephysicalstate,moment/activationdependent;not newbodyintervention. Baselineavailable only original−20..30ms audit,50msbaseline null.'))
 out=dict(passed=True,source_sha256=hashes,registered_trials=1,seven_neuron_offsets_exact=True,unmodified_base_matrices_initialfinal=True,actual_compiled_arrays_initialfinal=len(initial),body400prefix_exact=True,full_decoder_max_error=filtererror,strict_support_selection=report['selection'],metrics=metrics,selected_pool_spikes=poolcounts,calls=dict(model_compile=0,body_step=0,network=0,neural_step=0,optimizer=0),limits='InterventionisconstantMNthresholdfromfirstprewarm,not musclecommand or policy;poolspikechanges can couple throughbodyfeedback. Physicalfailure overridesgeometry or localtorque improvement.')
 assert all(sha(Path(p))==h for p,h in hashes.items());dump(O/'RESULT.json',out);print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
