"""Predeclared D8 normal versusmatchednoodor/cut/swap counts, replayseparate."""
from contracts import *
def main():
 verify();import numpy as np;sys.path.insert(0,str(C));from spatial import arrival_metrics
 layouts=read(ROOT/'PREREG_PROPOSAL.json')['layouts'];counts=dict(normal=0,cut=0,swap=0,noodor=0);rows=[];proof={};bodyall=True;cutall=True
 base=ROOT/'trials/D_noodor';br=read(base/'RESULT.json');ba=read(base/'VERIFY.json');require(ba['integrity_all_pass'] and all(ba['checks'].values()) and ba['result_sha256']==sha(base/'RESULT.json'),'newmatchednoodoraudit')
 with np.load(base/'geometry_1ms.npz',allow_pickle=False) as z:basepath=z['sensor_midpoint_xyz'][5799:,:2].copy();clock=z['time'][5799:].copy()
 bodyall=bool(br['functional']['stable_body']);proof[str(base/'RESULT.json')]=sha(base/'RESULT.json');proof[str(base/'VERIFY.json')]=sha(base/'VERIFY.json')
 for cfg in layouts:
  entry=dict(layout=cfg,controls={});normalgeometry=None
  for control in ('normal','cut','swap'):
   name=control+'_'+cfg['id'];folder=ROOT/'trials'/name;r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json'),'Dactualaudit '+name)
   stable=bool(r['functional']['stable_body']);bodyall=bodyall and stable;arrived=stable and r['spatial_metrics']['arrived_endpoint_proxy'];counts[control]+=int(arrived)
   entry['controls'][control]=dict(arrival_counts=arrived,spatial_metrics=r['spatial_metrics'],stable_body=stable,original_engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],yaw_deg=float(np.degrees(r['yaw_change_rad'])),cut_precondition=r['cut_precondition'],audit=str(folder/'VERIFY.json'))
   if control=='cut':cutall=cutall and r['cut_precondition']['effective_response_before_cut']
   if control=='normal':
    with np.load(folder/'geometry_1ms.npz',allow_pickle=False) as z:normalgeometry=z['source_center_xy'][5799].copy()
   proof[str(folder/'RESULT.json')]=sha(folder/'RESULT.json');proof[str(folder/'VERIFY.json')]=sha(folder/'VERIFY.json')
  distance=np.linalg.norm(basepath-normalgeometry,axis=1);inside=distance<=.06;met=arrival_metrics(clock,inside,clock-5.8);arrived=br['functional']['stable_body'] and met['arrived_endpoint_proxy'];counts['noodor']+=int(arrived);entry['controls']['noodor']=dict(arrival_counts=bool(arrived),shared_physical_trial='D_noodor',independent_repetition=False,source_center_for_geometry=normalgeometry.tolist(),dstart=float(distance[0]),dend=float(distance[-1]),dmin=float(distance.min()),**met);rows.append(entry)
 replay=[]
 for cfg in read(ROOT/'PREREG_PROPOSAL.json')['replay_trials']:
  folder=ROOT/'trials'/cfg['id'];skip=ROOT/('REPLAY_NOT_RUN_'+cfg['id']+'.json')
  if skip.exists():replay.append(read(skip));continue
  r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');cmp=read(folder/'REPLAY_COMPARISON.json');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json'),'Dreplayaudit');replay.append(dict(id=cfg['id'],status='AUDITED_COMMAND_REPLAY_NOT_CLOSED_LOOP',comparison=cmp,counts_as_new_closed_loop_success=False));proof[str(folder/'VERIFY.json')]=sha(folder/'VERIFY.json');proof[str(folder/'REPLAY_COMPARISON.json')]=sha(folder/'REPLAY_COMPARISON.json')
 flags=dict(normal6of8=counts['normal']>=6,noodor_difference4=counts['normal']-counts['noodor']>=4,cut_difference4=counts['normal']-counts['cut']>=4,swap_difference4=counts['normal']-counts['swap']>=4,all_body_stable=bodyall,all8_cut_preconditions_met=cutall,replay2_audited=all(r['status']=='AUDITED_COMMAND_REPLAY_NOT_CLOSED_LOOP' for r in replay))
 result=dict(status='D_CONFIRMATION_PASS' if all(flags.values()) else 'D_CONFIRMATION_FAILED_OR_INCOMPLETE',all_confirmation_targets_pass=all(flags.values()),flags=flags,arrival_counts=counts,rows=rows,replay=replay,source_hashes=proof,unique_physical_trials=25+sum(r['status']=='AUDITED_COMMAND_REPLAY_NOT_CLOSED_LOOP' for r in replay),noodor_authoritative_unique_trial=1,noodor_layout_evaluations=8,replicates_per_layout_control=1,no_food_ingestion_or_parking_claim=True,if_used_for_future_training_oldD_becomes_development=True,old_strict_motor_qualification_unchanged=True,old_strict_motor_engineering_qualified=False)
 with (ROOT/'D_CONFIRMATION_SUMMARY.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
 verify();print(json.dumps(dict(counts=counts,flags=flags,all_confirmation_targets_pass=result['all_confirmation_targets_pass'])))
if __name__=='__main__':main()
