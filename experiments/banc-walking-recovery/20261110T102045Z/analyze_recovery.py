"""Read-only verification and coverage of stronger mixed trajectories."""
import json
import numpy as np
from scipy.spatial import cKDTree
from recovery_bc import validate_recovery
from identity import HERE,atomic_json,sha256


def main():
    meta=validate_recovery(HERE/'RECOVERY_DATA')
    base=HERE.parent/'20261110T094051Z/SMOOTH_DATASET'
    train=[]
    for i in [0,1]:
        with np.load(base/f'episode_{i:03d}.npz',allow_pickle=False) as z:train.append(z['X238'].copy())
    train=np.concatenate(train)[:,:195];mean=train.mean(0);scale=np.maximum(train.std(0),.05)
    tree=cKDTree((train-mean)/scale)
    results=[]
    for ep in meta['episodes']:
        with np.load(ep['path'],allow_pickle=False) as z:
            x=z['X238'].copy();stage=z['stage'].copy();student=z['student_action'].copy();expert=z['Y42'].copy()
        distance,_=tree.query((x[stage==1,:195]-mean)/scale,k=1)
        report=json.loads(__import__('pathlib').Path(ep['report_path']).read_text())
        metrics=report['stage_metrics'].get('mixed',{})
        results.append(dict(episode=ep['episode'],seed=ep['seed'],split=ep['split'],
           decisions=ep['samples'],mixed_decisions=ep['mixed_samples'],early_stop=ep['early_stop'],
           actual_state_nearest_train_RMS_median=float(np.median(distance)/np.sqrt(195)),
           actual_state_nearest_train_RMS_p95=float(np.quantile(distance,.95)/np.sqrt(195)),
           expert_student_action_RMSE=float(np.sqrt(np.mean((student[stage==1]-expert[stage==1])**2))),
           actual_mixed_metrics=metrics))
    out=dict(all_metadata_hashes_verified=True,beta=meta['config']['beta'],results=results,
             interpretation='actual mixed training examples, NOT autonomous neural walking. Coverage measured physical195 only using base TRAIN stats.',
             source_meta_sha256=sha256(HERE/'RECOVERY_DATA/DATASET.json'))
    atomic_json(HERE/'RECOVERY_ANALYSIS.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='results'}),flush=True)
    for row in results:
        print(json.dumps(dict(episode=row['episode'],mixed=row['mixed_decisions'],early=row['early_stop'],
           coverage_RMS=row['actual_state_nearest_train_RMS_median'],coverage_p95=row['actual_state_nearest_train_RMS_p95'],
           action_error=row['expert_student_action_RMSE'],COM=row['actual_mixed_metrics'].get('com_min'))),flush=True)

if __name__=='__main__':main()
