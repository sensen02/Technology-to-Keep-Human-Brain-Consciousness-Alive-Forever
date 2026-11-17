"""Bind the continuous bench's zero-yaw stock course to the archived body behavior."""
from pathlib import Path

import numpy as np

from physical import BASE, HERE, bench, sha, write

current=HERE/"STOCK_PHYSICS"
original=BASE/"PROTO_native"
new=bench.read(current/"RESULT.json")
old=bench.read(original/"RESULT.json")
assert new["native_joint_pass"] is True and old["joint_sampled_pass"] is True
for path,result in ((current,new),(original,old)):
    for f,k in (("TRACE.npz","trace_sha256"),("DECISIONS.npz","decision_sha256")):
        assert sha(path/f)==result[k]
with np.load(current/"TRACE.npz",allow_pickle=False) as a, np.load(original/"TRACE.npz",allow_pickle=False) as b:
    shared=sorted(set(a.files)&set(b.files))
    for key in shared:
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    delta=a["com"][10100,:2]-a["com"][500,:2]
with np.load(current/"DECISIONS.npz",allow_pickle=False) as a, np.load(original/"DECISIONS.npz",allow_pickle=False) as b:
    for key in ("ms","x","action"):
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    np.testing.assert_array_equal(a["command"][:,0]>0.,b["intent"].astype(bool))
    assert not a["command"][:,1].any()
audit=bench.read(current/"NATIVE_AUDIT.json")
with np.load(current/"NATIVE.npz",allow_pickle=False) as a:
    assert sha(current/"NATIVE.npz")==audit["native_sha256"]
    contacts=a["forces"][500*80:10100*80]
    foot_values=contacts[:,6:12].sum(0)/contacts[:,:6].sum(0)
    legacy_slip=float(foot_values.max()/(np.linalg.norm(delta)/9.6))
    assert not a["pose"][:,7].any()
old_native=bench.read(original/"NATIVE_AUDIT.json")
expected=old_native["windows"]["walking"]["slip_max"]
np.testing.assert_allclose(legacy_slip,expected,atol=3e-12,rtol=0.)
write(HERE/"STOCK_COMPATIBILITY.json",dict(
    original=str(original),engineering_bench=str(current),passed=True,
    trace_shared_fields_exact=shared,decisions_x_action_ms_exact=True,
    strict_intent_equivalence=True,scope="Frozen zero-yaw stock command only",
    native_integrations=audit["native_integrations"],reference_world_x_native_forward=float(delta[0]/9.6),
    reference_world_xy_normalized_native_slip=legacy_slip,
    frame_sensitive_integrated_forward=audit["walking"]["forward"],
    frame_sensitive_integrated_slip=audit["walking"]["slip"],
    note="Integrated moving arc uses local instantaneous heading; straight exit preserves historical XY normalization and fixes the turnout heading.",
    pins={str(path):sha(path) for path in [current/"RESULT.json",current/"TRACE.npz",
        current/"DECISIONS.npz",current/"NATIVE_AUDIT.json",original/"NATIVE_AUDIT.json",original/"RESULT.json"]}))
