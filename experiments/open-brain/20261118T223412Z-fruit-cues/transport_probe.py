"""Replay the Eon odor outputs over the qualified BANC neck and frozen body."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from probe_eon_odor import BASE, BIDIR, CASES
from preflight import BODY_DIR, build_scene, place

FIXED = BASE/'20261117T164207Z-fixed-neck'
TRAIN = BASE/'20261117T202200Z-body-stop-repair'
GROUND = BASE/'20261117T071007Z-ground-dn-response'
sys.path.insert(0, str(FIXED))
from fixed_neck import FixedCircuit, full_catalog
from adapters import CalibratedReceiver
sys.path[:0] = [str(TRAIN), str(GROUND), str(BASE/'20261117T192322Z-terrain-body-learning')]
from ground_response import Ground
from policy_repair import read as load_actor
from shared import advance, feature, observation, phase_state, setup
from proprioception import extra_live


KEEP = ('no_odor', 'behind_wall', 'open', 'PFL3_positive')


@torch.inference_mode()
def run(incoming, out):
    torch.set_num_threads(1)
    metadata = json.loads((incoming/'RESULT.json').read_text())
    with np.load(incoming/'TRACE.npz', allow_pickle=False) as data:
        emitted = data['dn_spikes'][:, [CASES.index(name) for name in KEEP]]
    dn_roots = list(metadata['metrics']['open']['dn_by_port'])
    catalog = full_catalog()['ports']
    groups = {catalog[root]['group'] for root in dn_roots} | {'front_touch', 'front_load'}
    ports = {root: registry for root, registry in catalog.items() if registry['group'] in groups}
    circuit = FixedCircuit({'ports': ports}, batch=len(KEEP))
    columns = [circuit.port_index[root] for root in dn_roots]
    maps_path = BIDIR/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json'
    anchors = json.loads(maps_path.read_text())['probes']
    calibration_file = GROUND/'REFINE/CALIBRATION.json'
    calibration = json.loads(calibration_file.read_text())
    readers = [CalibratedReceiver(circuit) for _ in KEEP]
    for receiver in readers:
        for key, (_, _, sign) in receiver.probes.items():
            roots = anchors['/'.join(key)]
            receiver.probes[key] = roots, [circuit.index[root] for root in roots], sign
    body_weight = BASE/'20261117T211320Z-body-command-calibration/FIT/epoch040.pt'
    actor = load_actor(body_weight)
    bodies = [build_scene(None) for _ in KEEP]
    with np.load(BODY_DIR/'CANONICAL_INIT.npz', allow_pickle=False) as source:
        snapshot = {k: source[k] for k in ('qpos', 'qvel', 'ctrl_joint')}
    for body in bodies:
        place(body, snapshot, (0., 0.), 0.)
        body.motor.reset(seed=137)
        setup(body, 'terminal')
    cpgs = [copy.deepcopy(body.motor.cpg_network) for body in bodies]
    freqs = [cpg.intrinsic_freqs.copy() for cpg in cpgs]
    ticks = len(emitted) + 3500
    drive = np.zeros((len(KEEP), len(circuit.roots)), dtype=np.float32)
    events = np.zeros((len(KEEP), len(ports)), dtype=bool)
    cut = np.zeros(len(ports), dtype=bool)
    arrived = np.zeros((ticks, *emitted.shape[1:]), dtype=bool)
    command = np.zeros((len(KEEP), 2))
    controls = [None] * len(KEEP)
    physical = np.zeros((ticks, len(KEEP), 6))
    for tick in range(ticks):
        events[:, columns] = emitted[tick] if tick < len(emitted) else False
        _, got, _, voltage = circuit.step(drive, cut, eon_output=events)
        arrived[tick] = got[:, columns]
        if tick % 5 == 0:
            for i, reader in enumerate(readers):
                reader.update(voltage[i:i+1], .0005)
                command[i] = Ground.decode(SimpleNamespace(receiver=reader, calibration=calibration))
        for i, (body, cpg, freq) in enumerate(zip(bodies, cpgs, freqs, strict=True)):
            advance(cpg, freq, command[i])
            if tick % 10 == 0:
                obs = observation(body)
                phase = phase_state(cpg)
                inputs = np.r_[feature(body.d.qpos.copy(), body.d.qvel.copy(),
                                       obs.stumbling_contact_forces, command[i], phase, body.weight),
                               extra_live(body)].astype(np.float32)
                control, pads, _ = actor.act(inputs, phase)
                controls[i] = control, pads
            body.apply(*controls[i])
            q = body.d.xmat[body.thorax].reshape(3, 3)
            physical[tick, i, :3] = body.d.xpos[body.thorax]
            physical[tick, i, 3] = np.arctan2(q[1, 0], q[0, 0])
            physical[tick, i, 4:] = command[i]
            assert not body.d.warning.number.any()
    padded = np.pad(emitted, ((0, ticks-len(emitted)), (0, 0), (0, 0)))
    for j, col in enumerate(columns):
        d = circuit.neck_delay[col]
        assert not arrived[:d, :, j].any()
        assert np.array_equal(arrived[d:, :, j], padded[:-d, :, j])
    assert not physical[:, 0, 4:].any()
    rows = {}
    for i, name in enumerate(KEEP):
        delta = physical[-1, i, :3] - physical[0, i, :3]
        rows[name] = dict(arrived=int(arrived[:, i].sum()), displacement_xyz_mm=delta.tolist(),
                          yaw_degrees=float(np.rad2deg(physical[-1, i, 3] - physical[0, i, 3])),
                          max_forward_command=float(np.max(physical[:, i, 4:].mean(axis=1))))
    out.mkdir()
    np.savez_compressed(out/'TRACE.npz', arrived=arrived, pose_and_command=physical)
    pins = [Path(__file__), incoming/'TRACE.npz', incoming/'RESULT.json',
            maps_path, calibration_file, body_weight, FIXED/'PARAMETERS.json', FIXED/'READOUT.json']
    (out/'RESULT.json').write_text(json.dumps(dict(status='OPEN_LOOP_EON_ODOR_MOTION_REPLAY',
        native_body_steps=ticks * len(KEEP), actual_live_olfactory_feedback=False,
        body_ascending_feedback=False, eon_steps=0, signal_arrivals_tick_exact=True, results=rows,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in pins}),
        indent=2, allow_nan=False)+'\n')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eon_probe', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.eon_probe, args.output)
