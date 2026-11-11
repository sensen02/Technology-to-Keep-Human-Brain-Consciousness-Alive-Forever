"""Independent array-only metric and identity recomputation; no body/model construction."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 r=json.loads((HERE/'EVALUATION/RESULT.json').read_text());meta=json.loads((HERE/'FRESH_TEST.json').read_text());checks={}
 checks['all_source_data_hashes_unchanged']=all(sha(p)==h for group in ('source_hashes','data_hashes') for p,h in r[group].items())
 checks['prediction_hash']=sha(HERE/'EVALUATION/PREDICTIONS.npz')==r['predictions_sha256']
 with np.load(HERE/'EVALUATION/PREDICTIONS.npz',allow_pickle=False) as z:x=z['receptors'].copy();y=z['teacher_turn'].copy();pred=z['learned_turn'].copy();swap=z['swapped_turn'].copy();hidden=z['hidden'].copy();mask=z['qualified_sign_mask'].copy()
 conf=meta['episodes_config'];angle=np.asarray([c['angle'] for c in conf]);amp=np.asarray([c['amplitude'] for c in conf]);food=(angle!=0)&(amp>0);zero=amp==0;center=(angle==0)&(amp>0)
 checks['full44x240x2_inputs_and_hidden32']=x.shape==(44,240,2) and hidden.shape==(44,240,32) and pred.shape==(44,240)
 checks['all_values_finite_bounded']=all(np.isfinite(q).all() for q in (x,y,pred,swap,hidden)) and np.max(np.abs(pred))<=1 and np.max(np.abs(swap))<=1
 checks['initial_receptors_zero']=np.array_equal(x[:,0],np.zeros((44,2)))
 checks['teacher_label_provenance_exact']=np.array_equal(y,np.clip(angle[:,None]/.15,-1,1)*np.clip(x.mean(-1)/.15,0,1))
 checks['qualified_mask_exact']=np.array_equal(mask,food[:,None]&(np.abs(angle[:,None])>=.04)&(x.mean(-1)>=.05))
 recomputed=dict(fresh_all_MSE=float(np.mean((pred-y)**2)),fresh_food_only_MSE=float(np.mean((pred[food]-y[food])**2)),qualified_sign_fraction=float(np.mean(np.sign(pred[mask])==np.sign(np.broadcast_to(angle[:,None],pred.shape)[mask]))),no_food_RMS=float(np.sqrt(np.mean(pred[zero]**2))),no_food_bias=float(pred[zero].mean()),center_RMS=float(np.sqrt(np.mean(pred[center]**2))),center_bias=float(pred[center].mean()),zero_output_baseline_MSE=float(np.mean(y**2)),swapped_input_MSE=float(np.mean((swap-y)**2)))
 checks['all_metrics_recomputed_exact']=all(v==r['metrics'][k] for k,v in recomputed.items())
 criteria=dict(all_episode_MSE=recomputed['fresh_all_MSE']<=.02,nonzero_bearing_food_only_MSE=recomputed['fresh_food_only_MSE']<=.02,qualified_turn_sign=recomputed['qualified_sign_fraction']>=.90,no_food_RMS=recomputed['no_food_RMS']<=.02)
 checks['strict_registered_criteria_recomputed']=criteria==r['criteria'] and all(criteria.values())==r['all_predeclared_targets_pass']
 checks['sensor_only_flags']=r['scope_tag']=='SENSOR_PRETRAIN_ONLY' and r['physical_steps']==0 and r['Core_constructions']==0 and r['deployment_qualified'] is False and r['physical_approach_established'] is False
 old=HERE.parent/'20261111T034309Z';manifest=json.loads((old/'ARTIFACT_SHA256.json').read_text())
 checks['old_sealed_failed_experiment_unchanged']=all(sha(p)==v['sha256'] for p,v in manifest['files'].items())
 report=dict(all_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},recomputed_metrics=recomputed,no_food_max_absolute_turn=float(np.max(np.abs(pred[zero]))),no_food_each_episode_RMS=[float(np.sqrt(np.mean(v**2))) for v in pred[zero]],physical_steps=0,optimizer_updates=0,source_sha256=sha(__file__),checkpoint_sha256=r['checkpoint_sha256'])
 (HERE/'EVALUATION/INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(all_passed=report['all_passed'],checks=len(checks),no_food_max_absolute_turn=report['no_food_max_absolute_turn'])))
 if not report['all_passed']:raise ValueError('independent sensory evaluation failed')
if __name__=='__main__':main()
