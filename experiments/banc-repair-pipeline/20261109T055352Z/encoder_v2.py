"""Sensory encoder v2: same cells, same ports, only the transduction is made information-
preserving.

The v1 encoder (experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py) maps 90
physical values (42 joint angles, 42 joint velocities, 6 foot loads) onto 24 independent
signals: knee angle, mean coxa angle, |knee velocity| and load per leg. That map has local
rank 24 of 90, so it makes exactly identical input vectors for states that differ only in
knee-velocity sign, in coxa axis, or in any trochanter angle. No readout can recover those
distinctions afterwards.

v2 keeps the frozen identity qualification and every port of v1 and only changes which
physical value each qualified port carries. Registered assumptions, all uncalibrated:

* a chordotonal-organ port in a leg may carry any of that leg's joint angles (v1 already
  assumed this for the tibia only);
* a hair-plate port may carry any single coxa axis instead of the mean of the three;
* hook direction keeps the sign of the tibia angular velocity (the v1 name denotes a
  directional sensor but the v1 formula took an absolute value).

Unobserved and left that way: coxa pitch/roll velocities and all trochanter velocities,
because there are not enough qualified ports and inventing ports would fabricate cells.
"""
from collections import OrderedDict
from pathlib import Path
import importlib.util
import json
import sys

import numpy as np

V1_DIR = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder')
BODY_WEIGHT_NATIVE = 24.446354755878005
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')          # the order the v1 encoder declares
N_NEURONS = 20000
CURRENT_BOUNDS = (0.0, 0.06)


def _load_v1():
    spec = importlib.util.spec_from_file_location('encoder_v1', V1_DIR / 'encoder.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules['encoder_v1'] = m
    spec.loader.exec_module(m)
    return m


def _split(ports, k):
    ports = list(ports)
    if len(ports) < k:
        raise ValueError('pool too small for %d groups' % k)
    out, start = [], 0
    for i in range(k):
        n = len(ports) // k + (1 if i < len(ports) % k else 0)
        out.append(ports[start:start + n])
        start += n
    return out


class SensoryEncoderV2:
    """encode(q42, v42, load6) -> (20000,) currents, plus a signal manifest."""

    def __init__(self, v1=None):
        self.v1 = v1 or _load_v1().SensoryEncoder()
        names = tuple(self.v1.joint_names)
        self.joint_names = names
        nidx = {n: i for i, n in enumerate(names)}
        # signal list: (leg, signal_name, joint_or_none, kind, ports)
        signals = []
        for leg in LEGS:
            claw = self.v1.channel_local_indices[leg]['claw_position']
            hair = self.v1.channel_local_indices[leg]['hairplate_angle']
            hook = self.v1.channel_local_indices[leg]['hook_direction']
            camp = self.v1.channel_local_indices[leg]['campaniform_strain']
            angled = _split(claw, 4)
            coxad = _split(hair, 3)
            vel = _split(hook, 2)
            for jn, ports in zip(('Tibia_pitch', 'Trochanter_yaw', 'Trochanter_pitch', 'Trochanter_roll'), angled):
                signals.append((leg, 'chordotonal_angle_' + jn.lower(), nidx['joint_' + leg + jn], 'angle', ports))
            for jn, ports in zip(('Coxa_yaw', 'Coxa_pitch', 'Coxa_roll'), coxad):
                signals.append((leg, 'hairplate_angle_' + jn.lower(), nidx['joint_' + leg + jn], 'angle', ports))
            for jn, ports in zip(('Tibia_pitch', 'Coxa_yaw'), vel):
                signals.append((leg, 'hook_rate_' + jn.lower(), nidx['joint_' + leg + jn], 'rate', ports))
            signals.append((leg, 'campaniform_load', None, 'load', camp))
        self.signals = signals
        self.n_signals = len(signals)
        self.index_maps = []
        for leg, name, ji, kind, ports in signals:
            self.index_maps.append((leg, name, ji, kind, np.asarray(ports, dtype=np.int64)))
        self.used_ports = np.concatenate([m[4] for m in self.index_maps])
        if len(np.unique(self.used_ports)) != len(self.used_ports):
            raise ValueError('a port is assigned to two signals')

    # ---- transduction -----------------------------------------------------------
    def _angle(self, qn, ji):
        return 0.01 + 0.04 * qn[ji]

    def _rate(self, v, ji):
        return 0.03 + 0.03 * np.tanh(v[ji] / 10.0)

    def _load(self, load, leg):
        li = LEGS.index(leg)
        return 0.05 * np.tanh(max(float(load[li]), 0.0) / (BODY_WEIGHT_NATIVE / 6.0))

    def independence_matrix(self):
        """90 physical inputs -> independent signals. Rank = number of distinguishable signals."""
        A = np.zeros((90, self.n_signals))
        for k, (leg, name, ji, kind, _p) in enumerate(self.signals):
            if kind == 'angle':
                A[ji, k] = 1.
            elif kind == 'rate':
                A[42 + ji, k] = 1.
            else:
                A[84 + LEGS.index(leg), k] = 1.
        return A

    def encode(self, q42_rad, qvel42_rad_s, foot_load6_native):
        q = np.asarray(q42_rad, float)
        v = np.asarray(qvel42_rad_s, float)
        load = np.asarray(foot_load6_native, float)
        if q.shape != (42,) or v.shape != (42,) or load.shape != (6,):
            raise ValueError('expect (42,), (42,), (6,)')
        if not (np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(load).all()):
            raise ValueError('non-finite input')
        lo, hi = self.v1.joint_ranges_rad.T
        qn = (np.clip(q, lo, hi) - lo) / (hi - lo)
        cur = np.zeros(N_NEURONS)
        for leg, name, ji, kind, ports in self.index_maps:
            val = self._angle(qn, ji) if kind == 'angle' else (
                self._rate(v, ji) if kind == 'rate' else self._load(load, leg))
            cur[ports] = val
        np.clip(cur, *CURRENT_BOUNDS, out=cur)
        return cur

    def contract(self):
        return dict(
            schema='banc-sensory-v2/1',
            status='UNCALIBRATED_ENGINEERING_TRANSDUCTIONS_ON_IDENTITY_QUALIFIED_CELLS',
            same_cells=True, neuron_count=N_NEURONS,
            qualified_ports=int(len(self.used_ports)),
            independent_signals=self.n_signals,
            local_rank_v1=24, local_rank_v2=int(np.linalg.matrix_rank(self.independence_matrix())),
            input_dimension=90,
            per_leg_signals=[s[1] for s in self.signals[:10]],
            assumptions=[
                'a chordotonal-organ port may carry any joint angle of its own leg, not only the tibia',
                'a hair-plate port may carry one coxa axis instead of the mean of three',
                'hook direction keeps the sign of the tibia angular velocity',
                'gains and offsets are the v1 engineering values, not measured tuning'
            ],
            still_unobserved=['all trochanter angular velocities', 'coxa pitch angular velocity',
                              'coxa roll angular velocity', 'no vision/olfaction/food/energy channel'],
            ports_per_signal=dict(OrderedDict((f'{leg}:{name}', int(len(p)))
                                              for (leg, name, _j, _k, p) in self.index_maps)),
            does_not_claim=['measured fly sensory tuning', 'that the extra dimensions survive to motor neurons',
                            'any new cell identity or synapse'])


if __name__ == '__main__':
    enc = SensoryEncoderV2()
    out = Path(__file__).resolve().parent
    c = enc.contract()
    (out / 'SENSORY_V2_CONTRACT.json').write_text(json.dumps(c, indent=2) + '\n')
    print(json.dumps({k: c[k] for k in ('qualified_ports', 'independent_signals', 'local_rank_v1', 'local_rank_v2')}))
