"""Optical-only check against individually ray-traced frames with chamber glass present."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from PIL import Image
import numpy as np

src,d=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('published_tracker',src/'track_markers.py')
tracker=importlib.util.module_from_spec(spec);spec.loader.exec_module(tracker)
info=json.loads((src/'input.json').read_text())
colors=np.array(list(info['colors'].values()))
names=list(info['colors'])
idx=[i for i,n in enumerate(names) if n.endswith('_tarsus_tip')]
ps=np.array(list(info['projection'].values()))
origin=np.array(json.loads((d/'rig_origin.json').read_text()))
fast=np.load(d/'insert/observed_feet_mm.npy')
records=[]
for t in [40,213,300]:
    xy=np.stack([tracker.detect(Image.open(d/'fullrgb'/f'{t}_{c}.png'),colors)[idx] for c in info['projection']])
    world,views=tracker.recover(ps,xy)
    mm=(world-origin)*1000
    for i,(p,n) in enumerate(zip(mm,views)):
        records.append(dict(frame=t,leg=names[idx[i]].split('_')[0],views=int(n),xyz_mm=p.tolist() if n>=2 else None,
                            difference_mm=float(np.linalg.norm(p-fast[t-1,i])) if n>=2 else None))
diff=[r['difference_mm'] for r in records if r['difference_mm'] is not None]
(d/'RAYCHECK.json').write_text(json.dumps(dict(n=len(records),valid=len(diff),median_3d_difference_um=np.median(diff)*1000 if diff else None,
    p90_3d_difference_um=np.percentile(diff,90)*1000 if diff else None,rows=records),indent=2))
