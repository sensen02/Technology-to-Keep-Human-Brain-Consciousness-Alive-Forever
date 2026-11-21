"""Check that streaming cues match the archived observer's strictly causal features."""
import json
import os
from pathlib import Path

import numpy as np

from treatment import CausalCue, load_models, predict
from adaptive_observer import cues

root=Path(os.environ["P1_ARCHIVE_ROOT"])
path=root/"experiments/open-brain/20261118T031132Z-flat-neck-assay/FORMAL/seed11701/NEURAL.npz"
with np.load(path,allow_pickle=False) as f:
    record={key:f[key] for key in ("an_arrived","dn_emitted")}
model=load_models()
features=[cues(record,lane) for lane in range(4)]
target=np.stack([np.stack([predict(model[name],x[name]) for x in features],axis=1)
                 for name in ("motor","other","fused")],axis=1)
online=CausalCue()
values=[]
for dn,an in zip(record["dn_emitted"],record["an_arrived"],strict=True):
    values.append(online.estimate())
    online.update(dn,an[:,1])
past=np.asarray(values)
delta=float(np.max(abs(target-past)))
assert target.shape==past.shape and delta<1e-10
print(json.dumps({"checked_ticks":len(past),"lanes":4,"maximum_prediction_difference_hz":delta}))
