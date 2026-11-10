"""Saved-data sensor clipping and local coverage diagnostics, no simulation."""
import json
import numpy as np
from scipy.spatial import cKDTree
from identity import HERE,atomic_json


def main():
    root=HERE.parent;base=root/'20261110T094051Z/SMOOTH_DATASET'
    train=[]
    for i in [0,1]:
        with np.load(base/f'episode_{i:03d}.npz',allow_pickle=False) as z:train.append(z['X238'].copy())
    train=np.concatenate(train);mean=train[:,:195].mean(0);scale=np.maximum(train[:,:195].std(0),.05)
    tree=cKDTree((train[:,:195]-mean)/scale)
    groups=dict(q=(0,42),target_error=(42,84),qvel=(84,126),target=(126,168),
                load=(168,174),gravity=(174,177),root_v=(177,180),root_omega=(180,183),
                pad_height=(183,189),pad_velocity=(189,195))
    def clipping(x):return {name:float(np.mean(np.abs(x[:,a:b])>=9.999)) for name,(a,b) in groups.items()}
    result=dict(training_rows=len(train),training_clipping=clipping(train),evaluations={})
    for name,directory in [('smoothFF',root/'20261110T094051Z/EVAL_FF_SMOOTH'),
         ('normFF',root/'20261110T094554Z/EVAL_FF_NORM'),('normGRU',root/'20261110T094554Z/EVAL_GRU_NORM'),
         ('recoveryFF',HERE/'EVAL_FF_RECOVERY')]:
        with np.load(directory/'decisions_5ms.npz',allow_pickle=False) as z:
            x=z['X238'].copy();sources=z['source'];auto=sources==3;x=x[auto]
        distance,_=tree.query((x[:,:195]-mean)/scale,k=1)
        per=[]
        for lo,hi in [(0,.1),(.1,.5),(.5,1.),(1.,2.)]:
            a,b=int(lo/.005),int(hi/.005)
            per.append(dict(seconds=[lo,hi],nearest_training_standardized_L2_median=float(np.median(distance[a:b])),
              normalized_per_feature_RMS=float(np.median(distance[a:b])/np.sqrt(195)),clipping=clipping(x[a:b])))
        result['evaluations'][name]=dict(intervals=per,interpretation='distance in physical195 rawfeatures, frozen TRAIN scale floor.05; excludes directpreviousaction and speedcommand')
    recovery=json.loads((HERE/'RECOVERY_DATA/DATASET.json').read_text())
    for ep in recovery['episodes']:
        with np.load(ep['path'],allow_pickle=False) as z:x=z['X238'];stage=z['stage'];xx=x[stage==1]
        dist,_=tree.query((xx[:,:195]-mean)/scale,k=1)
        ep.clear();ep.update(mixed_nearest_training_L2_median=float(np.median(dist)),mixed_clipping=clipping(xx))
    result['assisted_recovery_coverage']=recovery['episodes']
    result['conclusion']='Quantifies on-trajectory narrowness and observed drift, not impossibility/sole-cause proof. Stronger mixed/burst aggregation needs independent physical validation.'
    atomic_json(HERE/'COVERAGE_DIAGNOSIS.json',result)
    print(json.dumps(dict(training_clip=result['training_clipping'],neural_distance_by_window={name:[r['normalized_per_feature_RMS'] for r in value['intervals']] for name,value in result['evaluations'].items()},assisted_recovery_L2median=[e['mixed_nearest_training_L2_median'] for e in result['assisted_recovery_coverage']])),flush=True)

if __name__=='__main__':main()
