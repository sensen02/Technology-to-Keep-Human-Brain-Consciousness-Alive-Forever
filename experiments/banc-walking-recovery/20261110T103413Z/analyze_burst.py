"""Evaluate actual burst/recovery coverage without re-running physics."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from burst_recovery import validate_burst
from identity import HERE,atomic_json,sha256


def main():
    meta=validate_burst(HERE/'BURST_DATA')
    base=HERE.parent/'20261110T094051Z/SMOOTH_DATASET';x=[]
    for i in [0,1]:
        with np.load(base/f'episode_{i:03d}.npz') as z:x.append(z['X238'].copy()[:,:195])
    x=np.concatenate(x);mu=x.mean(0);scale=np.maximum(x.std(0),.05);tree=cKDTree((x-mu)/scale)
    results=[]
    for ep in meta['episodes']:
        report=json.loads(Path(ep['report_path']).read_text())
        with np.load(ep['path']) as z:
            obs=z['X238'].copy();stage=z['stage'].copy();actions=z['executed_action'].copy();expert=z['Y42'].copy()
        row=dict(episode=ep['episode'],split=ep['split'],samples=ep['samples'],early=ep['early_stop'],
             corrections=report['correction_statistics'],stage_metrics=report['stage_metrics'],coverage={})
        for s,name in [(1,'student_burst'),(2,'teacher_recovery')]:
            mask=stage==s
            if not mask.any():continue
            d,_=tree.query((obs[mask,:195]-mu)/scale,k=1)
            row['coverage'][name]=dict(median_RMS=float(np.median(d)/np.sqrt(195)),p95_RMS=float(np.quantile(d,.95)/np.sqrt(195)),
               executed_vs_expert_action_RMSE=float(np.sqrt(np.mean((actions[mask]-expert[mask])**2))))
        results.append(row)
        print(json.dumps(dict(episode=row['episode'],samples=row['samples'],early=row['early'],
              coverage=row['coverage'],corrections=row['corrections'])),flush=True)
    atomic_json(HERE/'BURST_ANALYSIS.json',dict(all_inputs_verified=True,results=results,
      dataset_sha256=sha256(HERE/'BURST_DATA/DATASET.json'),
      interpretation='short student/teacher assisted training, not autonomous walking; positive correction means observed error magnitude reduced during recovery, not proof of complete stabilizability'))

if __name__=='__main__':main()
