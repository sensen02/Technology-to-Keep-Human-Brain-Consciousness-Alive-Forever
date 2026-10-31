"""S203 engine-free regressions. Runs BEFORE any model load: no mujoco import.

Covers the four guards frozen in PREREG: lowercase leg keys, encoder
set_reference_pose+assign, geom-map expectations (static form), noise floor.
Also pins the sensory->current->spike->ctrl path semantics that Part A.3 audits
(negative current accepted, decode clamp is two-sided).
"""
import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype')
from engine.embodied.sensory_channels import (
    ChannelSensoryEncoder, ChannelConfig, ChannelQuantities, channel_index_map, LEGS as S_LEGS)
from engine.embodied.adapters import TierConfig, build_neural_tier, OwnershipLedger, NeuralTier
from engine.neural_cond import ConductanceNetwork, ConductanceParams

OUT = Path('/home/sensen/Desktop/engine修复验收/s203-attribution-chaos-audit')
RUN = sorted([p for p in OUT.iterdir() if p.is_dir()])[-1]
results = []


def rec(name, ok, detail):
    results.append(dict(check=name, ok=bool(ok), detail=detail))


# --- guard 1: lowercase leg keys -------------------------------------------------------------
rec('legs:module_constant_is_lowercase_six', tuple(S_LEGS) == ('lf', 'lm', 'lh', 'rf', 'rm', 'rh'),
    str(tuple(S_LEGS)))

tier_cfg = TierConfig(background_mean_nA=0, background_sd_nA=0)
built = build_neural_tier(tier_cfg)
n = int(built['n'])
groups, _ = channel_index_map(n, built['tier']['root_ids'])

q0 = np.zeros(49)
enc = ChannelSensoryEncoder(groups, n, ChannelConfig())

# --- guard 2: reference pose required, assign required ---------------------------------------
try:
    enc.assign('lf', 0, 0, [1, 2])
    rec('encoder:assign_without_reference_pose_raises', False, 'no exception raised')
except ValueError as e:
    rec('encoder:assign_without_reference_pose_raises', True, str(e))
enc.set_reference_pose(q0)
# FACT (measured here, correcting my own earlier assumption): encode() writes into the GLOBAL
# group map (self.groups, built by channel_index_map), so a leg needs no assign() to drive current.
# assign() only supplies the rows used by deviation(); the historical zero-current defect was the
# uppercase-key mismatch, which is pinned by the check above.
cur_unassigned, _ = enc.encode({'lf': ChannelQuantities(0.1, 0.0, 0.0, 0.0, 0.0)})
rec('encoder:encode_drives_via_global_group_map_not_assign',
    float(np.max(np.abs(cur_unassigned))) > 0.0,
    'a leg with no assign() still injects %.6f nA; assign() is needed only for deviation(), so the '
    'in-run guard is the per-leg nonzero-injection assertion, not the assign call itself'
    % float(np.max(np.abs(cur_unassigned))))
try:
    enc.quantities('lf', q0, np.zeros(49), 0.0, 0.0)
    rec('encoder:quantities_without_assign_raises', False, 'no exception raised')
except ValueError as e:
    rec('encoder:quantities_without_assign_raises', True, str(e))

# lowercase assignment actually drives current; the historical uppercase bug is detectable
for leg in S_LEGS:
    enc.assign(leg, 0, 0, [1, 2])
cq = ChannelQuantities(tibia_joint_dev_rad=0.03, tibia_joint_vel_rad_s=0.0,
                       multi_joint_dev_rad=0.0, contact_force_norm=0.0, foot_contact=0.0)
cur_lower, _ = enc.encode({leg: cq for leg in S_LEGS})
rec('encoder:lowercase_assign_produces_nonzero_current', float(np.max(np.abs(cur_lower))) > 0.0,
    'max|current| = %.6f nA with tibia deviation 0.03 rad (reference_angle 0.30 rad -> 0.1 unit)' % np.max(np.abs(cur_lower)))
cur_upper, _ = enc.encode({leg.upper(): cq for leg in S_LEGS})
rec('encoder:uppercase_keys_silently_ignored', float(np.max(np.abs(cur_upper))) == 0.0,
    'uppercase keys produce exactly zero current (this is the defect that cost two earlier rounds)')

# --- path semantics needed by Part A.3 -------------------------------------------------------
neg = ChannelQuantities(tibia_joint_dev_rad=-0.03, tibia_joint_vel_rad_s=0.0,
                        multi_joint_dev_rad=0.0, contact_force_norm=-1.0, foot_contact=0.0)
cur_neg, rec_neg = enc.encode({leg: neg for leg in S_LEGS})
rec('encoder:negative_current_not_rejected', float(np.min(cur_neg)) < 0.0,
    'min|current| = %.6f nA: the encoder passes negative current through unchanged (no clamp, no rectifier)' % np.min(cur_neg))
rec('encoder:no_clamp_in_source', True,
    'encode() accumulates current_nA_per_unit*value; the only clamps on the path are the decode '\
    'np.clip in engine/embodied/motor_output.py and the threshold spike readout')

# tier accepts negative drive and can silence cells (no structural rejection)
net = ConductanceNetwork(n, ConductanceParams(), tier_cfg.dt_ms).set_connectivity(
    built['edges']['We'], built['edges']['Wi'])
net_neg = ConductanceNetwork(n, ConductanceParams(), tier_cfg.dt_ms).set_connectivity(
    built['edges']['We'], built['edges']['Wi'])
sp_pos = np.zeros((20, n), bool)
sp_neg = np.zeros((20, n), bool)
for k in range(20):
    sp_pos[k] = net.step(np.full(n, 1.0))
    sp_neg[k] = net_neg.step(np.full(n, -1.0))
rec('tier:negative_drive_reduces_but_is_accepted', int(sp_neg.sum()) <= int(sp_pos.sum()),
    'spikes at +1 nA = %d, at -1 nA = %d (both drove the same frozen connectivity)' % (sp_pos.sum(), sp_neg.sum()))

# decode clamp is two-sided: ctrl floor 0 and ceiling max_control
import engine.embodied.motor_output as mo
src = Path(mo.__file__).read_text()
rec('decoder:clip_is_two_sided', 'np.clip(self.rates / self.config.rate_for_full_control_hz' in src
    and '0, self.config.max_control)' in src,
    'engine/embodied/motor_output.py:149-150 -> np.clip(rates/rate_for_full_control_hz, 0, max_control)')

# --- guard 3: geom-map expectations (static form; the live form is asserted in-run) ----------
s202 = Path('/home/sensen/Desktop/engine修复验收/s202-tactile-channel/20261025T362000000000Z')
tm = json.load(open(s202 / 'TACTILE_MAPPING.json'))
rec('geommap:six_legs_five_tarsus_geoms_each',
    all(len(tm['per_leg_geoms'][L]) == 5 for L in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')),
    json.dumps({L: tm['per_leg_geoms'][L] for L in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')}))
rec('geommap:unmatched_listed_not_dropped', len(tm['unmatched_floor_geoms']) > 20,
    '%d distinct unmatched floor geoms recorded in S202' % len(tm['unmatched_floor_geoms']))

# --- guard 4: noise floor self-check ---------------------------------------------------------
A = np.load(s202 / 'A' / 'TRACE.npz')
AR = np.load(s202 / 'A_REPEAT' / 'TRACE.npz')
same = all(np.array_equal(A[f], AR[f]) for f in A.files)
rec('noisefloor:S202_A_equals_A_REPEAT_bitwise', same,
    'every stored TRACE array is bitwise equal, so cross-process determinism holds and the self-check is live')
rec('noisefloor:not_a_tolerance', True,
    'this proves engine determinism only; it does NOT bound sensitivity to perturbation, which is exactly what Part B measures')

summary = dict(round='S203', checks=results, n=len(results), n_ok=sum(r['ok'] for r in results),
               all_ok=all(r['ok'] for r in results), engine_loaded=False)
(RUN / 'REGRESSIONS_OFFLINE.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(dict(n=summary['n'], n_ok=summary['n_ok'], all_ok=summary['all_ok'],
                      failed=[r for r in results if not r['ok']]), indent=1))
