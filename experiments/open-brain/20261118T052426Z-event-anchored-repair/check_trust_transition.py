"""Withhold the second readout's calibration permission during a live phase trace."""
from pathlib import Path
import json
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"20261118T035424Z-adaptive-neck"))
from archive_io import digest, load_case
from anchored import AnchoredReadout

# This wrapper intentionally uses the exact original codec through the live runner.
from run_live import CanonicalEvents, DT

_,_,record=load_case(11701)
bitstream=record["an_arrived"][:,0:1]
actual=AnchoredReadout(CanonicalEvents((1,2)),1,DT)
flags=[]
for t,row in enumerate(bitstream):
    observed=actual.encode(row)
    assert np.array_equal(observed,record["brain_an_events"][t,0:1])
    measured=actual.rates[:,0]*.66
    trusted=t not in range(11000,11500)
    result,applied=actual.reconcile_left_amplitude(measured,np.array([trusted]))
    assert not applied[0] or np.array_equal(result,actual.rates[:,0])
    if not applied[0]:
        assert np.array_equal(result,measured)
    flags.append(bool(applied[0]))
assert flags[10999] and not any(flags[11000:12499]) and flags[12499]
report=dict(previously_calibrated=True,verification_removed_tick=11000,
     correction_stopped_same_tick=True,verification_returned_tick=11500,
     correction_restarted_after_fresh_100ms_gate_tick=12499,
     brain_events_bitexact_to_archived_control=True,checked_ticks=len(bitstream))
with (HERE/"TRUST_TRANSITION.json").open("x") as f:
    json.dump(report,f,indent=2);f.write("\n")
print(report)
