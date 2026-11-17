"""ActualXfit andconditionaltargetfeedbackfit keptseparate; no physicalacceptance."""
import argparse
from sources import ROOT,read,save,sha,verify


def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-parent-evaluation',action='store_true');a=p.parse_args()
 if not a.execute_parent_evaluation:p.error('Parentfinalevaluation required')
 from learning import cpu1,metrics,forward
 cpu1()
 import torch,numpy as np
 import common,model
 from scan import make_scan
 from data import prepare,production_sequence,dev_sources
 tr=read(ROOT/'TRAIN/RESULT.json')
 if tr['updates']!=600 or tr['checkpoint']!=str(ROOT/'TRAIN/checkpoint_000600.pt') or sha(tr['checkpoint'])!=tr['sha256']:raise ValueError('Actualfinal600only')
 for step in (0,300,600):
  v=read(ROOT/'TRAIN'/f'VERIFY_{step:06d}.json')
  if not v['integrity_all_pass'] or sha(v['checkpoint'])!=v['checkpoint_sha256']:raise ValueError('All3freshsavedverifyrequired')
 m,payload=common.load_checkpoint(tr['checkpoint'],tr['sha256'],final=True);data=prepare(m);sc=make_scan(m);actualsc=model.make_scan(m);rows=metrics(sc,data)
 with torch.no_grad():
  out=forward(sc,data);actualres,actualh=actualsc(data['features']);actual_action=(data['oldmean']+actualres).tanh();path=ROOT/'EVALUATION_ARRAYS.npz'
  np.savez_compressed(path,actual_target=data['y'].numpy().copy(),valid=data['valid'].numpy().copy(),v1=data['v1'].numpy().copy(),target_selector=data['target_selector'].numpy().copy(),halfspan=data['halfspan'].numpy().copy(),actual_post_targets=data['actual_post_targets'].numpy().copy(),conditional_mean=out[0].numpy().copy(),conditional_residual=out[1].numpy().copy(),conditional_action=out[5].numpy().copy(),conditional_post_targets=out[4].numpy().copy(),actualX_action=actual_action.numpy().copy(),old_mean=data['oldmean'].numpy().copy(),actualX_residual=actualres.numpy().copy())
  actual_rows=[]
  for i,raw in enumerate(data['raw']):
   keep=data['valid'][:,i].bool();err=(actual_action[:,i]-data['y'][:,i]).square();actual_rows.append(dict(id=raw['entry']['id'],valid_rows=int(keep.sum()),masked_action_MSE=float(err[keep].mean()),masked_action_RMSE=float(err[keep].mean().sqrt()),unmasked_action_MSE=float(err.mean())))
  dev=[]
  for raw in dev_sources(read(ROOT/'INPUT_BINDINGS.json')):
   mean,h=production_sequence(m,raw);dev.append(dict(id=raw['entry']['id'],actual_rows=len(raw['x']),labels_available=False,teacher_MSE=None,action_RMS=float(mean.tanh().square().mean().sqrt()),new96_absmax=float(h[:,256:].abs().max())))
 verify(payload['config']['source_hashes']);verify(payload['config']['data_hashes'])
 save(ROOT/'EVALUATION.json',dict(kind='CONDITIONAL_TARGETFEEDBACK_FINAL600_EVALUATION_V1',step=600,checkpoint=tr['checkpoint'],checkpoint_sha256=tr['sha256'],conditional_rows=rows,actualX_rows=actual_rows,DEV=dev,arrays=dict(path=str(path),sha256=sha(path)),source_hashes=payload['config']['source_hashes'],data_hashes=payload['config']['data_hashes'],physical_steps=0,behavior_pass=None,new_neck_deployed=False,conditional_notphysicalcounterfactual=True))


if __name__=='__main__':main()
