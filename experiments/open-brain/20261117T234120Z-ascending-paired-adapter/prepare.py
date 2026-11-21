"""Collect actual fixed-neck input features from archived body force records.

This is sensory data preparation, NOT a trained original-brain imitation model.
No target array is synthesized when the original encoder has no target.
"""
import argparse
import time
import numpy as np
import torch

from common import *
from fixed_neck import FixedCircuit, parameters

DT = .0001
TAUS = np.array([.005, .020, .050])
CASES = (
    ('train_left', 'train', BODY/'CALIBRATED/arc_left_1_student'),
    ('train_right', 'train', BODY/'CALIBRATED/arc_right_1_student'),
    ('held_left', 'heldout', BODY/'HELD/arc_left_2_student'),
    ('held_right', 'heldout', BODY/'HELD/arc_right_2_student'),
)


def collect(case, split, source, cut_side, output):
    started = time.monotonic()
    catalog = read(BIDIR/'CATALOG.json')
    ports = {r: p for r, p in catalog['ports'].items() if p['group'] == 'front_touch'}
    assert len(ports) == 2
    partial = dict(catalog, ports=ports)
    circuit = FixedCircuit(partial)
    assert list(p['side'] for p in ports.values()) == ['left', 'right']
    with np.load(source/'TRACE.npz', allow_pickle=False) as a:
        # Source body leg order is lf,lm,lh,rf,rm,rh; force is native normal load.
        loads = a['normal_load'][:, [0, 3]].copy()
    metadata = read(source/'RESULT.json')
    assert metadata['steps'] == len(loads) == 20000
    assert metadata['trace_sha256'] == sha(source/'TRACE.npz')
    assert metadata['weight_sha256'] == sha(BODY/'FIT/epoch040.pt')
    assert np.isfinite(loads).all()
    output.mkdir()
    pins = pin(imported_sources() | {source/'TRACE.npz', source/'RESULT.json', FIXED/'PARAMETERS.json',
          BIDIR/'CATALOG.json', HERE/'SOURCE_AUDIT.json', HERE/'PROTOCOL.zh-CN.md', BODY/'FIT/epoch040.pt'})
    for p in ports.values():
        swc = BIDIR/'geometry'/p['object']['filename']
        assert sha(swc) == p['object']['sha256']
        pins[str(swc)] = sha(swc)
    receptors = [circuit.validate_inputs([r for r, _ in p['incoming_receptors']], 'body') for p in ports.values()]
    cut = np.array([p['side'] == cut_side for p in ports.values()], dtype=bool)
    drive = np.zeros((1, len(circuit.roots)), dtype=np.float32)
    rows = {k: [] for k in ('emitted', 'arrivals', 'features_hz', 'target_voltage_mv', 'receptor_stimuli')}
    filtered = np.zeros((2, 3))
    alpha = -np.expm1(-DT/TAUS)
    stimulus_load = np.zeros(2)
    # Keep each channel's actual first-hop targets for diagnostics; these are
    # not injected into a second duplicate copy of the same AN in the brain.
    targets = [[circuit.index[r] for r, _ in p['outgoing']] for p in ports.values()]
    for tick in range(len(loads)):
        if time.monotonic()-started > 180:
            raise TimeoutError('per-episode preparation budget exceeded')
        drive.fill(0.)
        # Force at t=(tick+1)dt cannot be consumed at tick*dt. At tick0 use no
        # history, then consume only the previously completed physics sample.
        if tick and tick % 5 == 0:
            stimulus_load = loads[tick-1].copy()
        use = (stimulus_load > .04) & (tick % 23 == 0)
        for side in range(2):
            if use[side]:
                drive[0, receptors[side]] = 68.75
        emitted, arrivals, _, voltage = circuit.step(drive, cut)
        filtered += alpha*(arrivals[0, :, None]/DT-filtered)
        # A cut does not clear distal biological target membrane history, but
        # decoder features here explicitly gate only the interrupted channel.
        filtered[cut] = 0.
        for key, value in dict(emitted=emitted[0], arrivals=arrivals[0], features_hz=filtered,
                target_voltage_mv=np.array([voltage[0, t].mean() for t in targets]),
                receptor_stimuli=use).items():
            rows[key].append(value.copy())
    arrays = {k: np.asarray(v) for k, v in rows.items()}
    arrays['normal_load_native'] = loads
    arrays['time_s'] = np.arange(len(loads))*DT
    np.savez_compressed(output/'INPUTS.npz', **arrays)
    verify(pins)
    write(output/'RESULT.json', dict(case=case, split=split, source=str(source), source_sha256=pins,
           inputs_sha256=sha(output/'INPUTS.npz'), neural_ticks=len(loads), physical_steps=0,
           body_force_samples_reused=len(loads), port_roots=list(ports), cut_side=cut_side,
           port_delay_ticks=circuit.neck_delay.tolist(),
           emitted_counts=arrays['emitted'].sum(0).tolist(), arrival_counts=arrays['arrivals'].sum(0).tolist(),
           target_status='MISSING_ORIGINAL_TEACHER_NO_TARGET_GENERATED',
           dt_s=DT, feature_taus_s=TAUS.tolist(),
           transduction='existing engineering threshold .04 native normal-load; 68.75mV pulse every23 ticks; '
                        'not a measured bristle transduction model',
           geometry_scope='same fixed source-derived BANC ports and synapses; only front-touch subgraph',
           runtime_source_hashes=pin(imported_sources()), wall_seconds=time.monotonic()-started))
    print(case, cut_side, arrays['arrivals'].sum(0).tolist(), round(time.monotonic()-started, 2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=[x[0] for x in CASES])
    parser.add_argument('output', type=Path)
    parser.add_argument('--cut-side', choices=('left', 'right', 'both', 'none'), default='none')
    args = parser.parse_args()
    if args.cut_side == 'both':
        raise ValueError('use independent left/right channel cuts for this first preparation protocol')
    torch.set_num_threads(1)
    collect(*next(x for x in CASES if x[0] == args.case), args.cut_side, args.output)
