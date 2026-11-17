"""Cdevelopment descriptive distances/arrival vsnewmatchednoodor; no newgates ortraining."""
from contracts import *
def main():
 verify();import numpy as np
 rows=[];proof={};proposal=read(ROOT/'PREREG_PROPOSAL.json');base=ROOT/'trials/C_noodor';require((base/'VERIFY.json').exists(),'all6Cneeded beforesummary')
 with np.load(base/'geometry_1ms.npz',allow_pickle=False) as z:basepath=z['sensor_midpoint_xyz'][5799:,:2].copy()
 for cfg in proposal['layouts']:
  folder=ROOT/'trials'/cfg['id'];r=read(folder/'RESULT.json');a=read(folder/'VERIFY.json');require(a['integrity_all_pass'] and all(a['checks'].values()) and a['result_sha256']==sha(folder/'RESULT.json'),'Cactualauditevidence')
  require(r['checkpoint_sha256']==MOTOR_SHA and r['behavior_checkpoint_sha256']==SENSOR_SHA,'samepoliciesC')
  with np.load(folder/'geometry_1ms.npz',allow_pickle=False) as z:center=z['source_center_xy'][5799].copy()
  bd=np.linalg.norm(basepath-center,axis=1);baseline_reduction=float(bd[0]-bd[-1]);metrics=dict(r['spatial_metrics']);metrics.update(matched_noodor_d_start_center=float(bd[0]),matched_noodor_d_end_center=float(bd[-1]),matched_noodor_d_min_center=float(bd.min()),matched_noodor_distance_reduction=baseline_reduction,extra_distance_reduction_vs_noodor=metrics['distance_reduction']-baseline_reduction)
  rows.append(dict(condition=cfg['id'],development_metrics=metrics,stable_body=r['functional']['stable_body'],original_engineering_pass=r['engineering_pass'],slip=r['autonomous']['slip_ratio_max'],actual_yaw_deg=float(np.degrees(r['yaw_change_rad'])),all6=r['autonomous_cycles']['all_legs_completed'],nativeNF=r['native']['post_nonfoot_peak_native'],integrity_all_pass=True,result=str(folder/'RESULT.json'),audit=str(folder/'VERIFY.json'),arrival_not_required_forCdevelopment=True,frontnotcausalproof=cfg['id']=='C_front'))
  proof[str(folder/'RESULT.json')]=sha(folder/'RESULT.json');proof[str(folder/'VERIFY.json')]=sha(folder/'VERIFY.json')
 save=dict(status='C6_DEVELOPMENT_COMPLETE_NOT_D_CONFIRMATION',rows=rows,all_integrity_pass=True,all_stable_body=all(r['stable_body'] for r in rows),arrivals=sum(r['development_metrics']['arrived_endpoint_proxy'] for r in rows if r['condition']!='C_noodor'),unique_physical_trials=6,independent_replicates_per_condition=1,no_training=True,no_arrival_success_gate_forC=True,food_ingestion=False,parking=False,source_hashes=proof,video_status='PENDING_PARENT_RENDER',next='Parent review Cdevelopment actual pathways and causalcomparisons; Dheldout requires separately frozen admissible approach strategy')
 with (ROOT/'C_DEVELOPMENT_SUMMARY.json').open('x') as f:json.dump(save,f,indent=2,ensure_ascii=False);f.write('\n')
 verify();print(json.dumps(dict(complete=True,arrivals=save['arrivals'],all_stable_body=save['all_stable_body'],unique_physical_trials=6)))
if __name__=='__main__':main()
