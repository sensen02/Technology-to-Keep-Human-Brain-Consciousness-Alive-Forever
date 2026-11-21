"""Report the post hoc time extension and exactly paired old actor separately."""
from pathlib import Path
import json

import numpy as np

from physical_audit import HERE,older

selected=older.inspect(HERE/'HOLD_L')
extended=older.inspect(HERE/'HOLD_L_7200')
control=older.inspect(HERE/'HOLD_L_BASELINE')
for field in ('seed','side','geometry','neural_seed','model_sha','profile','food'):
    assert selected[field]==extended[field]==control[field]
assert selected['policy']==extended['policy'] and extended['policy']!=control['policy']
pref={}
for filename,entries in {'BODY.npz':('trace','poses','meal','commands','observation'),
                         'NECK.npz':('emitted','arrived','ascending','drive','feeding')}.items():
    with np.load(HERE/'HOLD_L'/filename,allow_pickle=False) as a,np.load(
                 HERE/'HOLD_L_7200'/filename,allow_pickle=False) as b:
        for label in entries:
            if len(b[label])<len(a[label]):
                pref[label]=False
                continue
            pref[label]=bool(np.array_equal(a[label],b[label][:len(a[label])]))
report=dict(original_5p2=selected,repaired_reward_extended=extended,prior_actor_extended=control,
            exact_original_prefix_replayed=all(pref.values()),prefix_fields=pref,
            included_in_primary_rate=False)
(HERE/'EXTENSION_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('extended',extended['intake'],extended['meal_pass'],extended['duration_s'],
      'baseline',control['intake'],control['meal_pass'],control['duration_s'],'prefix',pref)
