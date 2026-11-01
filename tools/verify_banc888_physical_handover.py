"""Independent array-only physical handover audit; no engine imports or integration."""
from pathlib import Path
import json,hashlib,math
import numpy as np
P=Path('/home/sensen/Desktop/engine修复验收/s5-banc888-physical-handover/20261101T220018Z')
B=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
N=Path('/home/sensen/Desktop/engine修复验收/s5-muscle-native-recording/20261101T191552Z')
LEGS=['LF','LM','LH','RF','RM','RH']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def dump(name,x):
 (P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
def load(p):
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def episodes(bits,steps,times):
 cuts=np.r_[0,np.flatnonzero(bits[1:]!=bits[:-1])+1,len(bits)]
 rows=[dict(state=bool(bits[a]),first_step=int(steps[a]),last_step=int(steps[b-1]),first_s=float(times[a]),last_s=float(times[b-1]),sample_count=int(b-a),observed_span_s=float(times[b-1]-times[a]),left_censored=bool(a==0),right_censored=bool(b==len(bits))) for a,b in zip(cuts[:-1],cuts[1:])]
 assert np.array_equal(np.concatenate([np.full(r['sample_count'],r['state']) for r in rows]),bits)
 return rows

def main():
 execution=json.loads((P/'EXECUTION.json').read_text());assert execution['exit_code']==0
 source=json.loads((P/'SOURCE_IDENTITY.json').read_text())
 ledger_path=P/'SOURCE_CHANGE_LEDGER.json'
 ledger=json.loads(ledger_path.read_text())['changes'] if ledger_path.exists() else {}
 for p,h in source['source_sha256'].items():
  if sha(p)==h:continue
  change=ledger[p]
  assert change['execution_sha256']==h and sha(p)==change['delivery_sha256']
  assert sha(change['execution_snapshot'])==h
  assert sha(change['delivery_snapshot'])==change['delivery_sha256']
 assert sha(P/'RUN/MODEL.xml')==sha(B/'MODEL.xml')=='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
 actual=json.loads((P/'RUN/PREREGISTRATION.json').read_text());oldspec=json.loads((B/'PREREGISTRATION.json').read_text())
 config_diff={k:dict(old=v,new=actual.get(k)) for k,v in oldspec.items() if k!='source_hashes' and actual.get(k)!=v};assert not config_diff
 body=load(P/'RUN/BODY.npz');oldbody=load(B/'BODY.npz');assert len(body['time'])==4001
 assert np.max(abs(body['time']-oldbody['time']))<1e-12
 for key in ['qpos','qvel','activation','command']:assert np.array_equal(body[key][0],oldbody[key][0])
 # First .2s is preserved and independent of the new interface. Derived boundary forces
 # at exactly .2s are refreshed by the original API and excluded from this gate.
 for key in body:
  assert np.array_equal(body[key][:400],oldbody[key][:400]),key
 before=load(P/'RUN/HANDOVER_BEFORE.npz');after=load(P/'RUN/HANDOVER_AFTER.npz')
 protected=set(before)-{'ctrl','qfrc_applied','xfrc_applied'}
 assert all(np.array_equal(before[k],after[k]) for k in protected)
 assert not after['qfrc_applied'].any() and not after['xfrc_applied'].any()
 assert abs(float(after['time'])-1.3)<1e-10
 pre=load(P/'PREWARM_FINAL_STATE.npz');boundary=load(P/'RUN/NEURAL_BOUNDARY.npz');prewith=load(P/'RUN/PREWITHDRAW_NEURAL_STATE.npz');final=load(P/'RELEASE_FINAL_STATE.npz')
 for k in boundary:assert np.array_equal(boundary[k],prewith[k]) and np.array_equal(boundary[k],pre[k]),k
 assert np.array_equal(after['ctrl'],pre['pending'])
 raster=load(P/'NEURAL_RASTER.npz')['spikes'];assert raster.shape==(3600,12000)
 manifest=json.loads((P/'RUN/MAPPING.json').read_text());groups=[np.array(r['local_indices'],int) for r in manifest['rows']]
 motor=sorted(set(i for g in groups for i in g));assert len(motor)==204
 records=json.loads((P/'RUN/PREWARM_NEURAL_RECORDS.json').read_text())+json.loads((P/'RUN/NEURAL_RECORDS.json').read_text());assert len(records)==1800
 rates=np.zeros(90);pending=np.zeros(90);decay=math.exp(-.0005/.005);max_rate_error=0.;counts=[]
 for i,r in enumerate(records):
  pair=raster[2*i:2*i+2];assert np.array_equal(r['applied_command'],pending)
  c=np.array([int(pair[:,g].sum()) for g in groups]);counts.append(c)
  for sample in pair:
   instant=np.array([sample[g].mean()/.0005 if len(g) else 0. for g in groups]);rates=decay*rates+(1-decay)*instant
  err=float(abs(rates-np.asarray(r['decoder']['rates_hz'])).max());max_rate_error=max(max_rate_error,err);assert err<1e-11
  pending=np.clip(rates/100,0,1);assert np.max(abs(pending-r['decoded_next_command']))<1e-12
  assert np.array_equal(c,r['decoder']['pool_spike_counts'])
  assert int(pair.sum())==r['total_network_spikes'] and int(pair[:,motor].sum())==r['motor_spikes']
  assert r['applied_source_interval']==i-1
  if i==1099:assert np.max(abs(rates-pre['decoder_rates']))<1e-11 and np.array_equal(pending,pre['pending'])
 assert np.array_equal(raster.sum(axis=0),final['spike_count'])
 assert np.array_equal(raster[:2200].sum(axis=0),pre['spike_count'])
 assert np.max(abs(rates-final['decoder_rates']))<1e-11 and np.array_equal(pending,final['pending'])
 neural=load(P/'RUN/NEURAL.npz');assert np.array_equal(raster[2200:].reshape(700,2,12000).sum(axis=1),neural['spike_counts'])
 for state,n in [(pre,2200),(final,3600)]:
  ring=np.zeros_like(state['_ring'])
  for i in range(n):ring[i%len(ring)]=raster[i]
  assert np.array_equal(ring,state['_ring']) and int(state['ring_pos'])==n%len(ring)
 schema=json.loads((P/'CONTACT_SCHEMA.json').read_text());planes=set(schema['floor_geom_indices'])
 chunks=json.loads((P/'CHUNKS.json').read_text());acc={};max_force=max_tangent=max_shared=0.;total_contacts=jac_contacts=0
 friction=dict(positive_normal_rows=0,cone_excess_max=0.,positive_gap_loaded_rows=0,nonpositive_gap_loaded_rows=0)
 for item in chunks:
  assert sha(item['file'])==item['sha256'];z=load(item['file']);rows=z.pop('contact_rows');off=z.pop('contact_offsets')
  assert len(off)==len(z['step'])+1 and off[0]==0 and off[-1]==len(rows);total_contacts+=len(rows)
  for k,v in z.items():acc.setdefault(k,[]).append(v)
  for i in range(len(z['step'])):
   r=rows[off[i]:off[i+1]];f=np.zeros((6,3));nf=np.zeros(3);nfc=0;flags=np.zeros(6,bool);surface=np.zeros(6,bool);speeds=np.zeros(6)
   jf=bool(z['jacobian_sample'][i]);jac_contacts+=len(r) if jf else 0
   if len(r):
    assert np.isfinite(r[:,:18]).all() and np.isfinite(r[:,26:]).all()
    assert np.isfinite(r[:,18:26]).all() if jf else np.isnan(r[:,18:26]).all()
   for rr in r:
    g1,g2=int(rr[0]),int(rr[1]);leg=int(rr[4]);assert int(rr[2])==schema['geom_bodyid'][g1] and int(rr[3])==schema['geom_bodyid'][g2]
    normal=rr[12:15];assert abs(normal@normal-1)<1e-12
    if jf:
     rel=rr[18:21];tangent=rel-(rel@normal)*normal
     max_tangent=max(max_tangent,float(abs(tangent-rr[21:24]).max()),abs(float(np.linalg.norm(tangent))-rr[24]),abs(float(rel@normal)-rr[25]))
    if g1 in planes or g2 in planes:
     signed=rr[15:18]*(1 if g1 in planes else -1)
     if leg>=0:
      f[leg]+=signed;flags[leg]=True;surface[leg]|=rr[5]<=0
      if jf:speeds[leg]=max(speeds[leg],rr[24])
      if rr[26]>0:
       friction['positive_normal_rows']+=1;friction['positive_gap_loaded_rows' if rr[5]>0 else 'nonpositive_gap_loaded_rows']+=1
       tangforce=rr[15:18]-normal*(rr[15:18]@normal)
       friction['cone_excess_max']=max(friction['cone_excess_max'],float(np.linalg.norm(tangforce)-rr[27]*rr[26]))
     else:nf+=signed;nfc+=1
   max_force=max(max_force,float(abs(f-z['foot_force'][i]).max()),float(abs(nf-z['nonfoot_force'][i]).max()))
   assert nfc==z['nonfoot_count'][i] and np.array_equal(flags,z['foot_candidate'][i]) and np.array_equal(surface,z['foot_surface_candidate'][i])
   assert np.array_equal(f[:,2]>0,z['foot_loaded'][i])
   if jf:assert np.array_equal(speeds,z['foot_max_contact_tangent_speed'][i])
   n=int(z['step'][i]);kind=int(z['kind'][i]);eligible=n%40==0 and n!=16000 and (n!=104000 or kind==1)
   if eligible:
    j=n//40
    for field,key in [('foot_force','foot_force'),('foot_origin','foot_origins'),('com','com')]:max_shared=max(max_shared,float(abs(z[field][i]-body[key][j]).max()))
  print('ARRAY_VERIFIED',Path(item['file']).name,flush=True)
 acc={k:np.concatenate(v) for k,v in acc.items()};regular=acc['kind']==0
 assert np.array_equal(acc['step'][regular],np.arange(1,160001));assert (acc['kind']==1).sum()==1 and (acc['kind']==2).sum()==1
 assert np.array_equal(acc['step'][acc['kind']==1],[104000]);assert max_force<1e-10 and max_tangent<1e-9 and max_shared<1e-10
 release=regular&(acc['step']>104000);assert release.sum()==56000
 assert np.all(acc['phase'][release]==3) and not acc['assist_root_max'][release].any() and not acc['assist_leg_norm'][release].any() and not acc['external_max'][release].any() and not acc['mocap_count'].any() and not acc['warning_count'].any()
 assert execution['counters']['observer_jacobian_calls']==jac_contacts*2
 assert execution['counters']['observer_contact_force']==total_contacts
 assert execution['counters']['observer_forward']==len(acc['step'])==160002
 weight=json.loads((P/'RUN/GAINS.json').read_text())['weight']
 # Use identical selected endpoint roles for the new and old native trajectories.
 summary={};all_episodes={}
 for phase,label in [(2,'steady_assisted'),(3,'release')]:
  mask=regular&(acc['phase']==phase);ix=np.flatnonzero(mask);first=ix[acc['nonfoot_count'][ix]>0]
  s=dict(endpoint_samples=len(ix),first_nonfoot_relative_release_s=float(acc['time'][first[0]]-1.3) if len(first) else None,nonfoot_steps=len(first),nonfoot_occupancy=float(len(first)/len(ix)),com_height_min=float(acc['com'][ix,2].min()),com_height_last=float(acc['com'][ix[-1],2]),foot_total_fz_over_weight_mean=float(acc['foot_force'][ix,:,2].sum(axis=1).mean()/weight),foot_total_fz_over_weight_min=float(acc['foot_force'][ix,:,2].sum(axis=1).min()/weight),per_foot_mean_fz_over_weight=acc['foot_force'][ix,:,2].mean(axis=0)/weight,root_endpoint_displacement=acc['root'][ix[-1],:3]-acc['root'][ix[0],:3],legs={})
  all_episodes[label]={}
  for j,l in enumerate(LEGS):
   all_episodes[label][l]={};s['legs'][l]={}
   for field in ['foot_candidate','foot_surface_candidate','foot_loaded']:
    bits=acc[field][ix,j];e=episodes(bits,acc['step'][ix],acc['time'][ix]);all_episodes[label][l][field]=e
    s['legs'][l][field]=dict(true_fraction=float(bits.mean()),transitions=len(e)-1,observed_true_runs=sum(r['state'] for r in e))
   grid=ix[acc['jacobian_sample'][ix]];loaded_grid=grid[acc['foot_loaded'][grid,j]]
   s['legs'][l]['material_speed_half_ms']=dict(grid_endpoint_frames=len(grid),positive_fz_frames=len(loaded_grid),mean_peak_candidate_tangent_speed=float(acc['foot_max_contact_tangent_speed'][loaded_grid,j].mean()) if len(loaded_grid) else None,max_peak_candidate_tangent_speed=float(acc['foot_max_contact_tangent_speed'][loaded_grid,j].max()) if len(loaded_grid) else None,meaning='Per foot max among current candidate contact material-point tangent speeds, conditional on aggregate signed Fz>0; model length/s; not a slip distance')
  s['signed_per_foot_mean_world_force']=acc['foot_force'][ix].mean(axis=0)
  s['signed_nonfoot_mean_world_force']=acc['nonfoot_force'][ix].mean(axis=0)
  summary[label]=s
 dump('NATIVE_EPISODES.json',all_episodes)
 shared={}
 for label,bb in [('new_root888',body),('old121',oldbody)]:
  # Include release-after boundary; exactly 1400 right-endpoint bins exclude boundary.
  ix=np.arange(2600,4001);ep=ix[1:];non=ep[bb['nonfoot_count'][ep]>0];f=bb['foot_force'][ep,:,2].sum(axis=1)/weight
  shared[label]=dict(boundary_frames=len(ix),endpoint_bins=len(ep),first_nonfoot_relative_release_s=float(bb['time'][non[0]]-1.3) if len(non) else None,nonfoot_endpoint_count=len(non),nonfoot_occupancy=float(len(non)/len(ep)),com_start=bb['com'][2600],com_end=bb['com'][-1],com_height_min=float(bb['com'][ix,2].min()),root_displacement=bb['qpos'][-1,:3]-bb['qpos'][2600,:3],foot_fz_over_weight_mean=float(f.mean()),foot_fz_over_weight_min=float(f.min()),foot_fz_over_weight_last=float(f[-1]),per_foot_fz_over_weight_mean=bb['foot_force'][ep,:,2].mean(axis=0)/weight,per_foot_fz_over_weight_last=bb['foot_force'][-1,:,2]/weight,positive_command_entries=int(np.count_nonzero(bb['command'][ep])),saturated_command_entries=int(np.count_nonzero(bb['command'][ep]>=1)),candidate_transitions=np.count_nonzero(np.diff(bb['foot_contact'][ix].astype(int),axis=0),axis=0))
 # Old native summary is recomputed from raw saved chunk arrays using same mask.
 oldacc={k:[] for k in ['step','kind','phase','nonfoot_count','foot_force','com','root','foot_candidate','foot_surface_candidate','foot_loaded']}
 for chunk in json.loads((N/'CHUNKS.json').read_text()):
  assert sha(chunk['file'])==chunk['sha256']
  with np.load(chunk['file']) as z:
   for k in oldacc:oldacc[k].append(z[k])
 oldacc={k:np.concatenate(v) for k,v in oldacc.items()};mask=(oldacc['kind']==0)&(oldacc['step']>104000)
 assert mask.sum()==56000
 oldnative=dict(endpoint_samples=int(mask.sum()),nonfoot_steps=int(np.count_nonzero(oldacc['nonfoot_count'][mask])),nonfoot_occupancy=float(np.mean(oldacc['nonfoot_count'][mask]>0)),foot_fz_over_weight_mean=float(oldacc['foot_force'][mask,:,2].sum(axis=1).mean()/weight),per_foot_mean_fz_over_weight=oldacc['foot_force'][mask,:,2].mean(axis=0)/weight)
 first=oldacc['step'][mask][oldacc['nonfoot_count'][mask]>0];oldnative['first_nonfoot_relative_release_s']=float((first[0]-104000)*.0000125) if len(first) else None
 oldnative['transitions']={k:np.count_nonzero(np.diff(oldacc[k][mask].astype(int),axis=0),axis=0) for k in ['foot_candidate','foot_surface_candidate','foot_loaded']}
 np.savez_compressed(P/'TRAJECTORY_COMPARISON.npz',time=body['time'],new_com=body['com'],old_com=oldbody['com'],new_root=body['qpos'][:,:7],old_root=oldbody['qpos'][:,:7],new_foot_force=body['foot_force'],old_foot_force=oldbody['foot_force'])
 dump('SUMMARY.json',dict(new_native=summary,old_native_release=oldnative,shared_half_ms=shared,friction_all_phases=friction,force_weight_model_units=weight,neural=dict(network_spikes_prewarm=int(raster[:2200].sum()),network_spikes_release=int(raster[2200:].sum()),motor_spikes_release=int(raster[2200:,motor].sum()),motor_cells=len(motor)),notes='Events are sampled contact flags, not steps. Friction includes all phases. Material velocities evaluated only on .5ms grid and explicit boundaries; no continuous-time slip distance claimed. Absolute biological force calibration NOT_ESTABLISHED.'))
 dump('INDEPENDENT_VERIFICATION.json',dict(passed=True,model_load=0,body_step=0,neural_step=0,original_config_exact=True,initial_state_exact=True,first_point2s_original_body_exact=True,handover_physical_continuity_exact=True,prewarm_neural_8arrays_rates_pending_exact=True,half_step_raster_decoder_verified=True,max_rates_error=max_rate_error,max_signed_force_reconstruction_error=max_force,max_tangent_algebra_error=max_tangent,max_shared_native_body_error=max_shared,total_contact_rows=total_contacts,jacobian_contact_rows=jac_contacts,release_endpoint_steps=56000,source_hashes_verified=len(source['source_sha256']),limitations='Array audit validates recorded Jacobian algebra, not independent engine Jacobian recomputation. No extra physical or neural integration.'))
 print('INDEPENDENT_PHYSICAL_HANDOVER_PASS',flush=True)
if __name__=='__main__':main()
