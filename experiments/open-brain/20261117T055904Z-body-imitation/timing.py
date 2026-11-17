"""Measure real DN arrival, stop suppression and any transient re-excitation."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ports=json.loads((HERE.parent/'20261117T051249Z-bidirectional/CATALOG.json').read_text())['ports']
result={}
for path in sorted((HERE/'HELD').glob('*/RESULT.json')):
    meta=json.loads(path.read_text())
    roots=meta['arrival_roots']
    case=meta['case']
    with np.load(path.parent/'TRACE.npz',allow_pickle=False) as saved:
        emitted,arrived,trace=saved['emitted'],saved['arrived'],saved['trace']
    forward=[i for i,root in enumerate(roots) if ports[root]['group']=='forward']
    brake=[i for i,root in enumerate(roots) if ports[root]['group']=='halt_web_a']
    drive=np.abs(trace[:,6:8]).max(1)>0
    def first(series):
        indices=np.flatnonzero(series)
        return int(indices[0]) if len(indices) else None
    m={'first_forward_emission_tick': first(emitted[:,forward].sum(1)),
       'first_forward_arrival_tick': first(arrived[:,forward].sum(1)),
       'first_brake_arrival_tick': first(arrived[:,brake].sum(1)),
       'first_body_drive_tick': first(drive), 'forward_emitted': int(emitted[:,forward].sum()),
       'forward_arrived': int(arrived[:,forward].sum()), 'brake_arrived': int(arrived[:,brake].sum())}
    if case['brake']:
        both=np.flatnonzero((np.arange(len(trace))>=m['first_brake_arrival_tick']) & (np.arange(len(trace))<case['stop']))
        stopped=both[~drive[both]]
        assert len(stopped)
        first_suppressed=int(stopped[0])
        rebound=np.flatnonzero(drive & (np.arange(len(trace))>=case['stop']))
        rd=(np.linalg.norm(trace[rebound[-1],1:3]-trace[rebound[0],1:3]) if len(rebound) else 0.)
        m.update(first_fully_suppressed_tick=first_suppressed,
                 forward_arrivals_while_braked=int(arrived[stopped][:,forward].sum()),
                 decay_rebound_ticks=len(rebound), decay_rebound_displacement_mm=float(rd))
        assert m['forward_arrivals_while_braked']>0
    result[case['name']]=m
cut, restored=result['held_cut'],result['held_restore']
assert cut['forward_emitted']>0 and cut['forward_arrived']==0 and cut['first_body_drive_tick'] is None
assert restored['first_forward_arrival_tick']>=5080+20 and restored['first_body_drive_tick']>=restored['first_forward_arrival_tick']
with (HERE/'HELD_TIMING.json').open('x') as f:
    json.dump(result,f,indent=2); f.write('\n')
print(result)
