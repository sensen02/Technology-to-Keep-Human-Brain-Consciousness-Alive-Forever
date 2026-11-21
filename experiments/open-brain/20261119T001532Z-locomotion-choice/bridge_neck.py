"""Export measured Eon takeoff events for a native FlyBody physical test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cervical import FlightPaths, LABELS

PERIOD = 40
LAND_TICKS = 1200
TOTAL = 16500


def run(source, out, guide):
    recording = np.load(source / 'CHOICE.npz', allow_pickle=False)
    launch = recording['emitted'][:, :2]
    source_result = json.loads((source / 'RESULT.json').read_text())
    assert source_result['takeoff_gate_s'] is not None
    trace = np.zeros((TOTAL,4),dtype=bool)
    n = min(len(launch), TOTAL)
    trace[:n,:2] = launch[:n]
    landing_time = None
    if guide is not None:
        visual = json.loads((guide / 'RESULT.json').read_text())
        landing_time = visual['visual_trigger_s']
        assert landing_time is not None
        i = round(landing_time * 10000)
        trace[i:i+LAND_TICKS:PERIOD, 2:] = True
    cases = ('intact', 'cut', 'sham')
    raw = np.stack((trace, trace, np.zeros_like(trace)), axis=1)
    paths = [FlightPaths() for _ in cases]
    lanes = [p.integrate(raw[:,i],cut=i==1) for i,p in enumerate(paths)]
    events, levels = (np.stack(out, axis=1) for out in zip(*lanes,strict=True))
    out.mkdir()
    data = out / 'COMMANDS.npz'
    np.savez_compressed(data, labels=np.array(LABELS), cases=np.array(cases),
                        emitted=raw, arrived=events, levels=levels)
    assert np.array_equal(raw[:,0,:2], raw[:,1,:2])
    assert np.array_equal(raw[:n,0,:2], launch[:n])
    assert np.array_equal(levels[:n,0,:2], recording['post_neck'][:n,:2])
    record=dict(sha256=hashlib.sha256(data.read_bytes()).hexdigest(),dt_ms=.1,
                takeoff_source=str(source.resolve()), visually_triggered_landing_s=landing_time,
                takeoff_trace_sha256=hashlib.sha256((source/'CHOICE.npz').read_bytes()).hexdigest(),
                fixed_neck_parameters_sha256=paths[0].parameters_sha256,
                source_counts=raw.sum(axis=0).tolist(), arrivals=events.sum(axis=0).tolist())
    (out / 'IDENTITY.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--guide',type=Path)
    a=p.parse_args()
    run(a.source,a.output,a.guide)
