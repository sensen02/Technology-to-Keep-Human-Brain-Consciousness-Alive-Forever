"""ActualXaction primarysupervision, conditional distance explicitly NOTsupervision."""
import argparse,time
from sources import ROOT,read,save,sha,verify

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-evaluation',action='store_true');a=p.parse_args()
 if not a.execute_parent_evaluation:p.error('Parentfinal900evaluationrequired')
 start=time.monotonic()
 from learning import cpu1,metrics,tf_forward,forward,resource_gate
 cpu1()
 import torch,numpy as np
 import common
 from scan import make_scan,make_actual_scan
 from data import prepare,production_sequence,dev_sources
 tr=read(ROOT/'TRAIN/RESULT.json')
 if tr['updates']!=900 or tr['new_updates']!=300 or sha(tr['checkpoint'])!=tr['sha256']:raise ValueError('Actual900only')
 for step in (600,750,900):
  v=read(ROOT/'TRAIN'/f'VERIFY_{step:06d}.json')
  if not v['integrity_all_pass'] or sha(v['checkpoint'])!=v['checkpoint_sha256']:raise ValueError('All3freshsavedproofsrequired')
 m,payload=common.load_checkpoint(tr['checkpoint'],tr['sha256'],final=True);data=prepare(m);actualscan=make_actual_scan(m);condscan=make_scan(m);rows=metrics(actualscan,condscan,data)
 with torch.no_grad():
  tf=tf_forward(actualscan,data);cond=forward(condscan,data);path=ROOT/'EVALUATION_ARRAYS.npz';np.savez_compressed(path,actual_target=data['y'].numpy().copy(),valid=data['valid'].numpy().copy(),v1=data['v1'].numpy().copy(),target_selector=data['target_selector'].numpy().copy(),halfspan=data['halfspan'].numpy().copy(),actual_post_targets=data['actual_post_targets'].numpy().copy(),actualX_action=tf[3].numpy().copy(),actualX_residual=tf[1].numpy().copy(),conditional_mean=cond[0].numpy().copy(),conditional_residual=cond[1].numpy().copy(),conditional_action=cond[5].numpy().copy(),conditional_post_targets=cond[4].numpy().copy());dev=[]
  for raw in dev_sources(read(ROOT/'INPUT_BINDINGS.json')):
   mean,h=production_sequence(m,raw);dev.append(dict(id=raw['entry']['id'],actual_rows=len(raw['x']),labels_available=False,teacher_MSE=None,action_RMS=float(mean.tanh().square().mean().sqrt()),new96_absmax=float(h[:,256:].abs().max())))
 resource_gate(start);verify(payload['config']['source_hashes']);verify(payload['config']['data_hashes'])
 save(ROOT/'EVALUATION.json',dict(kind='STATE_ALIGNED_ACTION_FINAL900_EVALUATION_V1',step=900,checkpoint=tr['checkpoint'],checkpoint_sha256=tr['sha256'],actualX_rows=rows['actualX_rows'],conditional_rows=rows['conditional_rows'],primary_supervision_metric='actualX_actionfit',conditional_action_label_distance='COUNTERFACTUAL_DISTANCE_NOT_SUPERVISION',DEV=dev,arrays=dict(path=str(path),sha256=sha(path)),source_hashes=payload['config']['source_hashes'],data_hashes=payload['config']['data_hashes'],physical_steps=0,behavior_pass=None,new_neck_deployed=False,conditional_notphysicalcounterfactual=True))
if __name__=='__main__':main()
