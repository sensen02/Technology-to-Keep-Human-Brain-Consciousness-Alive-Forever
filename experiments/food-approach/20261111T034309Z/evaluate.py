"""Final1000 sensory heldout evaluation; physical qualification explicitly forbidden."""
import json
from pathlib import Path
import numpy as np,torch
from common import HERE,digest,save_json,guard
from learning import load_behavior,production


def values(model,x):
 with torch.no_grad():
  out,state=production(model,torch.tensor(x.transpose(1,0,2).copy(),dtype=torch.float64))
 return out.T.numpy(),state.transpose(0,1).numpy()

def main():
 torch.set_num_threads(1);guard();out=HERE/'EVALUATION';out.mkdir(exist_ok=False)
 ck=HERE/'TRAIN/checkpoint_001000.pt';sha=digest(ck);m,p=load_behavior(ck,sha)
 if p['curve'][-1]['step']!=1000:raise ValueError('only final1000 heldout evaluation')
 meta=json.loads((HERE/'DATASET.json').read_text())
 with np.load(HERE/'heldout.npz',allow_pickle=False) as z:x=z['receptors'].copy();y=z['teacher_turn'].copy();time=z['time_s'].copy()
 pred,states=values(m,x);swapped,_=values(m,x[:,:,::-1].copy())
 conf=meta['splits']['heldout']['episodes_config'];angles=np.asarray([r['angle'] for r in conf]);food=np.asarray([not r['no_food'] for r in conf])
 mask=(np.abs(angles[:,None])>=.04)&(x.mean(-1)>=.05)&food[:,None]
 def sign_fraction(output):return float(np.mean(np.sign(output[mask])==np.sign(np.broadcast_to(angles[:,None],output.shape)[mask])))
 mse=float(np.mean((pred-y)**2));sign=sign_fraction(pred);zero_rms=float(np.sqrt(np.mean(pred[~food]**2)))
 criteria=dict(heldout_MSE=mse<=.02,correct_sign_fraction=sign>=.90,zerofood_RMS=zero_rms<=.02)
 rows=[]
 for i,cfg in enumerate(conf):
  qualified=mask[i];rows.append(dict(**cfg,index=i,MSE=float(np.mean((pred[i]-y[i])**2)),turn_RMS=float(np.sqrt(np.mean(pred[i]**2))),qualified_sign_samples=int(qualified.sum()),correct_sign_fraction=None if not qualified.any() else float(np.mean(np.sign(pred[i,qualified])==np.sign(cfg['angle']))),swapped_MSE=float(np.mean((swapped[i]-y[i])**2)),mean_turn_last200ms=float(pred[i,-40:].mean()),label_mean_last200ms=float(y[i,-40:].mean())))
 mirror=[]
 for i,a in enumerate(conf):
  if a['angle']<=0 or a['no_food']:continue
  j=next(k for k,b in enumerate(conf) if b['angle']==-a['angle'] and b['amplitude']==a['amplitude'] and b['onset_s']==a['onset_s'] and b['seed']==a['seed'])
  both=mask[i]&mask[j]
  mirror.append(dict(positive_index=i,negative_index=j,angle=a['angle'],amplitude=a['amplitude'],onset_s=a['onset_s'],seed=a['seed'],opposite_sign_fraction=float(np.mean(pred[i,both]*pred[j,both]<0)) if both.any() else None,output_sum_RMS=float(np.sqrt(np.mean((pred[i]+pred[j])**2))),note='Each noisyepisode usesdeclaredseed; mirroredinputs are not bitexact bilateral swaps because noisechannels retainorderedrngdraws.'))
 with np.load(HERE/'train.npz',allow_pickle=False) as z:trainx=z['receptors'].copy()
 center_indices=[i for i,cfg in enumerate(meta['splits']['train']['episodes_config']) if cfg['angle']==0 and not cfg['no_food']]
 center,_=values(m,trainx[center_indices]);center_control=dict(kind='TRAINING_CENTER_CONTROL_NOT_HELDOUT',RMS=float(np.sqrt(np.mean(center**2))),bias=float(center.mean()))
 report=dict(scope_tag='SENSOR_PRETRAIN_ONLY',status='SENSORY_ENGINEERING_TARGETS_PASS' if all(criteria.values()) else 'SENSORY_ENGINEERING_TARGETS_FAIL',all_predeclared_targets_pass=all(criteria.values()),criteria=criteria,
  metrics=dict(heldout_MSE=mse,qualified_sign_fraction=sign,qualified_sign_samples=int(mask.sum()),zerofood_RMS=zero_rms,zerofood_bias=float(pred[~food].mean()),zero_output_baseline_MSE=float(np.mean(y**2)),zero_output_baseline_qualified_sign_fraction=0.,swapped_input_MSE=float(np.mean((swapped-y)**2)),swapped_input_correct_original_sign_fraction=sign_fraction(swapped),swap_output_sum_RMS=float(np.sqrt(np.mean((pred+swapped)**2)))),
  center_control=center_control,heldout_episodes=rows,mirrored_pairs=mirror,checkpoint=str(ck),checkpoint_sha256=sha,
  source_hashes={str(HERE/'evaluate.py'):digest(HERE/'evaluate.py'),**p['config']['source_hashes']},data_hashes=p['config']['data_hashes'],physical_steps=0,Core_constructions=0,deployment_qualified=False,physical_approach_established=False,
  limitations=['Statichead engineeringreactivecommandlabels only; heldout angle/dose/noise generalization is not navigation or biologicalvalidation.','No locomotor model loaded or deployed; motorqualification unaffected.','All failcases and frozen thresholds retained; inputswap is a sensorydirection diagnostic, not actualphysicalcausal ablation.'])
 np.savez_compressed(out/'PREDICTIONS.npz',receptors=x,teacher_turn=y,learned_turn=pred,swapped_turn=swapped,hidden=states,qualified_sign_mask=mask,time_s=time)
 report['predictions_sha256']=digest(out/'PREDICTIONS.npz');save_json(out/'RESULT.json',report);guard()
 print(json.dumps(dict(status=report['status'],criteria=criteria,metrics=report['metrics'])),flush=True)
if __name__=='__main__':main()
