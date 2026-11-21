"""Exercise the missing-foot and featureless-ground paths without access to scene answers."""
import csv
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from infer import estimate

source,terrain=map(Path,sys.argv[1:])
info=json.loads((source/'input.json').read_text())
with np.load(source/'reconstructed.npz') as original:
    track={key:original[key][:1].copy() for key in ('frames','8_views','4_upper','views_used')}
    track['names']=original['names']
idx=list(track['names']).index('LF_tarsus_tip')
with tempfile.TemporaryDirectory() as temp:
    d=Path(temp)
    for c in info['projection']:
        if c.startswith('H'):
            Image.new('L',tuple(info['frame_size']),130).save(d/(c+'.png'))
    estimate(info,track,d,d/'blank.csv')
    states=[r['state'] for r in csv.DictReader((d/'blank.csv').open())]
    assert states==['unresolved']*6
    track['8_views'][0,idx]=np.nan
    track['views_used'][0,idx]=0
    estimate(info,track,terrain,d/'missing.csv')
    rows=list(csv.DictReader((d/'missing.csv').open()))
    assert rows[0]['state']=='unresolved' and not rows[0]['marker_gap_mm']
    assert all(r['state']!='unresolved' for r in rows[1:])
print('PASS: featureless ground 6/6 unresolved; missing LF unresolved; remaining five retained')
