"""Clarify local-window eligibility versus strict first-contact-free prefixes."""
from pathlib import Path
import numpy as np,json
O=Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T231405Z')
def arrays(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
dist=arrays(O/'DATA/DISTANCES.npz');threshold=np.quantile(dist['warmdev_distance'],.95);report={}
for name,distance_key in [('closed_train','closed_distance'),('closed_dev','closeddev_distance'),('aggregate_1',None),('aggregate_2',None)]:
    ds=arrays(O/'DATA'/(name+'.npz'));strict=np.zeros(len(ds['eligible']),bool)
    for source in np.unique(ds['source']):
        a=arrays(Path(source)/'NATIVE.npz')['rows'];touch=np.flatnonzero(a[:,7]>0);first=int(touch[0]) if len(touch) else len(a);ids=np.flatnonzero(ds['source']==source);strict[ids]=ds['eligible'][ids]&((ds['body_index'][ids]//2*80)<first)
    item=dict(local_eligible=int(ds['eligible'].sum()),strict_prefix=int(strict.sum()),eligible_after_prior_touch=int(np.sum(ds['eligible']&~strict)))
    if distance_key:
        values=dist[distance_key][strict];item.update(strict_distance_median=float(np.median(values)),strict_distance_p90=float(np.quantile(values,.9)),strict_beyond_warm_dev_p95=float(np.mean(values>threshold)))
    report[name]=item
report['interpretation']='Frozen mask is local1ms contact/height eligibility, not no-contact-ever. Training followed its declared executable mask. Earlier commentary before-failure phrasing was too strong; strict-prefix statistics are now separately reported, without retraining or changing original data.'
(O/'PREFIX_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report),flush=True)
