"""S204 Part A.2 hard check + open-loop calibration gate.

1) Replay the stored H0 per-sample spike matrices through the PLATFORM decoder and require the
   resulting ctrl to equal the stored H0 ctrl BITWISE. Also recompute with an independent numpy
   reimplementation of the formula. If either fails, this round's evidence is void.
2) Replay the SAME spikes at H1 and H2 (same input, bitwise) and measure the off-rail fraction of
   non-empty actuators in command windows 1-3. This is the frozen gate's precondition.
3) Report the closed-loop arms (H0/H1/H2) separately, noting that their spike tickets diverge.
"""
import json
import dataclasses
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype')
from engine.embodied import adapters as ad
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig, build_motor_manifest

RUN = Path('/home/sensen/Desktop/engine修复验收/s204-readout-calibration-audit/20261025T073450411660Z')
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')

cen = json.load(open(RUN / 'READOUT_CENSUS.json'))
empty_idx = [c['actuator_index'] for c in cen['rows'] if c['pool_size'] == 0]
nonempty_idx = [c['actuator_index'] for c in cen['rows'] if c['pool_size'] > 0]
pool_of = {c['actuator_index']: c['local_indices'] for c in cen['rows']}

tier_cfg = ad.TierConfig(background_mean_nA=0, background_sd_nA=0)
built = ad.build_neural_tier(tier_cfg)
n = int(built['n'])
import mujoco
m = mujoco.MjModel.from_xml_path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
manifest_rows = manifest['rows']

z = np.load(RUN / 'H0' / 'TRACE.npz')
ctrl_stored = z['ctrl']
ncmd = ctrl_stored.shape[0]
NT_STEPS = 10
bits = np.unpackbits(z['spikes_packed'], axis=1)[:, :NT_STEPS * n].reshape(ncmd, NT_STEPS, n).astype(bool)

SPIKE_DT = 0.0005
NT_MS = 0.5


def replay(rate_full):
    cfg = dataclasses.replace(MotorOutputConfig(), rate_for_full_control_hz=float(rate_full))
    dec = MotorOutput(manifest, cfg)
    out = np.zeros((ncmd, m.nu))
    for k in range(ncmd):
        pend, _ = dec.consume(bits[k], spike_dt_s=NT_MS / 1000.0,
                              start_s=k * 0.005, end_s=(k + 1) * 0.005)
        out[k] = np.asarray(pend, float)
    return out


def replay_numpy(rate_full):
    """Independent reimplementation of the documented formula (no platform decoder)."""
    cfg = dataclasses.replace(MotorOutputConfig(), rate_for_full_control_hz=float(rate_full))
    decay = float(np.exp(-SPIKE_DT / cfg.filter_tau_s))
    groups = [np.asarray(r['local_indices'], int) for r in manifest_rows]
    rates = np.zeros(len(groups))
    out = np.zeros((ncmd, m.nu))
    for k in range(ncmd):
        for sample in bits[k]:
            inst = np.array([(sample[idx].mean() / SPIKE_DT) if idx.size else 0.0 for idx in groups])
            rates = decay * rates + (1.0 - decay) * inst
        out[k] = np.clip(rates / cfg.rate_for_full_control_hz, 0.0, cfg.max_control)
    return out


# ---- 1) hard check: platform replay reproduces stored ctrl bitwise -------------------------
c100 = replay(100.0)
c100_np = replay_numpy(100.0)
bitwise_platform = bool(np.array_equal(c100, ctrl_stored))
bitwise_numpy = bool(np.array_equal(c100_np, ctrl_stored))
max_abs_platform = float(np.max(np.abs(c100 - ctrl_stored)))
max_abs_numpy = float(np.max(np.abs(c100_np - ctrl_stored)))

# ---- 2) open-loop gate: same spikes, different H -------------------------------------------
replays = {'H0': c100, 'H1': replay(200.0), 'H2': replay(1000.0)}
H_VALUES = {'H0': 100.0, 'H1': 200.0, 'H2': 1000.0}
WIN = range(1, 4)
def off_rail(ctrl, idxs):
    sub = ctrl[list(WIN)][:, idxs]
    railed = (sub <= 1e-12) | (sub >= 1.0 - 1e-12)
    return dict(n_values=int(sub.size), at_rail=int(railed.sum()),
                off_rail=int((~railed).sum()), off_rail_fraction=float((~railed).mean()),
                at_zero=int((sub <= 1e-12).sum()), at_one=int((sub >= 1.0 - 1e-12).sum()),
                quantiles={q: float(np.quantile(sub, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
open_loop = {k: dict(rate_full=H_VALUES[k], nonempty=off_rail(v, nonempty_idx),
                     all90=off_rail(v, list(range(m.nu)))) for k, v in replays.items()}

# ---- 3) closed-loop arms ------------------------------------------------------------------
closed = {}
for arm in ('H0', 'H1', 'H2'):
    zz = np.load(RUN / arm / 'TRACE.npz')
    s = json.load(open(RUN / arm / 'SUMMARY.json'))
    closed[arm] = dict(rate_full=s['rate_for_full_control_hz'],
                       nonempty=off_rail(zz['ctrl'], nonempty_idx),
                       spikes_per_window=s['spikes_per_window_total'],
                       ctrl_equals_open_loop_h0=bool(np.array_equal(zz['ctrl'], c100)))
tickets = {a: closed[a]['spikes_per_window'] for a in closed}
tickets_identical = all(t == tickets['H0'] for t in tickets.values())
first_diverging_window = next((k for k in range(ncmd)
                              if any(tickets[a][k] != tickets['H0'][k] for a in ('H1', 'H2'))), None)

out = dict(
    hard_check=dict(
        platform_replay_equals_stored_bitwise=bitwise_platform,
        numpy_reimplementation_equals_stored_bitwise=bitwise_numpy,
        max_abs_diff_platform=max_abs_platform, max_abs_diff_numpy=max_abs_numpy,
        n_command_windows=ncmd, n_actuators=int(m.nu),
        verdict=('PASS: the stored ctrl is an exact function of the stored per-sample spike matrix, so the transfer path is '
                 'fully accounted for' if (bitwise_platform and bitwise_numpy) else
                 'FAIL: the recorded ctrl cannot be reproduced from the recorded spikes; this round is void')),
    transfer_formula=('rates_k = decay*rates_{k-1} + (1-decay)*inst_k, inst_k = mean over pool of (spikes_k/spike_dt); '
                      'decay = exp(-spike_dt/tau) = %.10f; ctrl = clip(rates/rate_full, 0, max_control)'
                      % float(np.exp(-SPIKE_DT / MotorOutputConfig().filter_tau_s))),
    open_loop_gate=dict(
        note=('same stored H0 spike matrices replayed at three full-scale values: the input is BITWISE identical across '
              'the three, which is the precondition the frozen gate asks for'),
        per_arm=open_loop),
    closed_loop_arms=dict(
        note=('the three closed-loop arms do NOT have identical spike tickets: changing the decoder changes ctrl, which '
              'changes the body, which changes the sensory input. The first window is identical by construction (all-zero '
              'hands-off start), and the tickets diverge from window %s on. They are reported as measured, not hidden.'
              % first_diverging_window),
        per_arm=closed, spike_tickets=tickets, tickets_bitwise_identical=tickets_identical,
        first_diverging_window=first_diverging_window))
(RUN / 'DECODER_TRANSFER_BITWISE.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(dict(hard_check=out['hard_check'],
                      open_loop={k: dict(h=H_VALUES[k], off_rail=round(open_loop[k]['nonempty']['off_rail_fraction'], 4),
                                         at_one=open_loop[k]['nonempty']['at_one']) for k in open_loop},
                      closed={k: dict(h=closed[k]['rate_full'],
                                      off_rail=round(closed[k]['nonempty']['off_rail_fraction'], 4))
                              for k in closed},
                      tickets_identical=tickets_identical, first_div=first_diverging_window), indent=1, default=str))
