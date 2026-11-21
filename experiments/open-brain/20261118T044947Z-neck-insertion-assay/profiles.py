"""Map the offline physical scenarios to named arrivals without changing brain/body."""
from pathlib import Path
import hashlib
import json

import numpy as np

HERE = Path(__file__).resolve().parent
MODES = ("unloaded_control", "retained_geometry", "unloaded_geometry", "unloaded_cable_proxy")
SEEDS = (10901, 11311, 11701)


def read_profiles(neck):
    path = HERE/"TRANSPORT_PROFILES.npz"
    with path.open("rb") as f:
        digest = hashlib.file_digest(f,"sha256").hexdigest()
    receipt=json.loads((HERE/"CABLE.json").read_text())
    if digest != receipt["data_sha256"]:
        raise ValueError("transport profile SHA mismatch")
    with np.load(path,allow_pickle=False) as data:
        subset=data["delay_ticks"]
        root_ids=data["root_ids"].tolist()
        baseline=np.array([neck.neck_delay[neck.port_index[r]] for r in root_ids])
        assert tuple(subset.shape)==(20000,len(MODES),len(root_ids))
        assert np.array_equal(baseline,data["baseline_ticks"])
        assert np.array_equal(data["time"],np.arange(20000)*.0001)
    columns = [neck.port_index[root] for root in root_ids]
    assert len(set(columns))==len(columns)
    assert np.all(subset>=1)
    full = np.broadcast_to(neck.neck_delay, (len(subset),len(MODES),len(neck.port_roots))).copy()
    full[:,:,columns]=subset
    return full,columns,root_ids
