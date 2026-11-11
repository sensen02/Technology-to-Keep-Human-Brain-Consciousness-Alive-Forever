"""Only final3000 is evaluated on fresh pre-frozen sensory test, not physical navigation."""
import json
import numpy as np,torch
from common import HERE,PREVIOUS,load_behavior,production,digest,save_json,guard

def values(m,x):
 with torch.no_grad():p,h=production(m,torch.tensor(x.transpose(1,0,2).copy(),dtype=torch.float64))
 return p.T.numpy(),h.transpose(0,1).numpy()

def main():
 torch.set_num_threads(1);guard();out=HERE/'EVALUATION';out.mkdir(exist_ok=False)
 ck=HERE/'TRAIN/checkpoint_003000.pt';sha=digest(ck);m,p=load_behavior(ck,sha)
 meta=json.loads((HERE/'FRESH_TEST.json').read_text());conf=meta['episodes_config']
 if digest(HERE/'fresh_test.npz')!=meta['sha256']:raise ValueError('fresh test changed')
 with np.load(HERE/'fresh_test.npz',allow_pickle=False) as z:x=z['receptors'].copy();y=z['teacher_turn'].copy();time=z['time_s'].copy()
 pred,states=values(m,x);swap,_=values(m,x[:,:,::-1].copy());angle=np.asarray([c['angle'] for c in conf]);amp=np.asarray([c['amplitude'] for c in conf]);food=(angle!=0)&(amp>0);nofood=amp==0;center=(angle==0)&(amp>0)
 mask=(np.abs(angle[:,None])>=.04)&(x.mean(-1)>=.05)&food[:,None]
 def signrate(q):return float(np.mean(np.sign(q[mask])==np.sign(np.broadcast_to(angle[:,None],q.shape)[mask])))
 allmse=float(np.mean((pred-y)**2));foodmse=float(np.mean((pred[food]-y[food])**2));sign=signrate(pred);zero=float(np.sqrt(np.mean(pred[nofood]**2)))
 criteria=dict(all_episode_MSE=allmse<=.02,nonzero_bearing_food_only_MSE=foodmse<=.02,qualified_turn_sign=sign>=.90,no_food_RMS=zero<=.02)
 rows=[]
 for i,c in enumerate(conf):
  valid=mask[i]
  rows.append(dict(**c,index=i,MSE=float(np.mean((pred[i]-y[i])**2)),turn_RMS=float(np.sqrt(np.mean(pred[i]**2))),qualified_sign_samples=int(valid.sum()),correct_sign_fraction=None if not valid.any() else float(np.mean(np.sign(pred[i,valid])==np.sign(c['angle']))),mean_turn_last200ms=float(pred[i,-40:].mean()),target_last200ms=float(y[i,-40:].mean())))
 pairs=[]
 for i,c in enumerate(conf):
  if c['angle']<=0:continue
  j=next(j for j,b in enumerate(conf) if b['angle']==-c['angle'] and b['amplitude']==c['amplitude'] and b['onset_s']==c['onset_s'] and b['seed']==c['seed'])
  both=mask[i]&mask[j];pairs.append(dict(positive_index=i,negative_index=j,opposite_sign_fraction=float(np.mean(pred[i,both]*pred[j,both]<0)) if both.any() else None,output_sum_RMS=float(np.sqrt(np.mean((pred[i]+pred[j])**2)))))
 metrics=dict(fresh_all_MSE=allmse,fresh_food_only_MSE=foodmse,qualified_sign_fraction=sign,qualified_sign_samples=int(mask.sum()),no_food_RMS=zero,no_food_bias=float(pred[nofood].mean()),center_RMS=float(np.sqrt(np.mean(pred[center]**2))),center_bias=float(pred[center].mean()),zero_output_baseline_MSE=float(np.mean(y**2)),zero_output_baseline_food_MSE=float(np.mean(y[food]**2)),swapped_input_MSE=float(np.mean((swap-y)**2)),swapped_input_food_MSE=float(np.mean((swap[food]-y[food])**2)),swapped_correct_original_sign=signrate(swap),swap_output_sum_RMS=float(np.sqrt(np.mean((pred+swap)**2))))
 report=dict(status='SENSOR_RESPONSE_ENGINEERING_TARGETS_PASS' if all(criteria.values()) else 'SENSOR_RESPONSE_ENGINEERING_TARGETS_FAIL',scope_tag='SENSOR_PRETRAIN_ONLY',all_predeclared_targets_pass=all(criteria.values()),criteria=criteria,metrics=metrics,episodes=rows,mirrored_pairs=pairs,checkpoint=str(ck),checkpoint_sha256=sha,source_hashes={str(HERE/'evaluate.py'):digest(HERE/'evaluate.py'),**p['config']['source_hashes']},data_hashes=p['config']['data_hashes'],old_test_status='DEVELOPMENT_VALIDATION_HISTORY_NOT_CURRENT_FRESH_TEST',physical_steps=0,Core_constructions=0,deployment_qualified=False,physical_approach_established=False,
  limitations=['Only static-head synthetic local odor responses and declared reactive training labels.','No motor model loaded; passing sensor metrics cannot establish physical food approach or biological validity.','Old1000 failed heldout result remains FAIL; current fresh test was frozen before correction updates.','All-case and food-only MSE gates jointly enforced before outcome; center results cannot dilute a failing food subset.'])
 np.savez_compressed(out/'PREDICTIONS.npz',receptors=x,teacher_turn=y,learned_turn=pred,swapped_turn=swap,hidden=states,qualified_sign_mask=mask,time_s=time)
 report['predictions_sha256']=digest(out/'PREDICTIONS.npz');save_json(out/'RESULT.json',report);guard();print(json.dumps(dict(status=report['status'],criteria=criteria,metrics=metrics)),flush=True)
if __name__=='__main__':main()
