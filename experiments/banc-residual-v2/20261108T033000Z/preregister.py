"""Preregistration: criterion-matched residual learning on top of the measured-data base network.

Frozen pieces reused unchanged: the full 42-joint body XML, the bounded incremental actuator
(30 rad/s target rate, 1 ms integration, finite PD128), the 183-dim proprioception contract,
the 5 ms policy tick, the frozen acceptance thresholds (support, non-foot, height, per-leg
material slip 0.5) and D1-D8 style gates expressed as my own recomputation.

New in this version (declared before running):
  action space   six TIBIA joints, one per leg, symmetric role (not the old asymmetric
                 coxa/trochanter set), residual bound +-0.003 rad per 5 ms tick
  base network   MODEL_MEASURED.npz (supervised on filtered measured motion), frozen
  reward         MUST MATCH the acceptance metric: hinge on the PER-LEG slip ratio
                 r_i = force-weighted tangential foot speed / forward speed, penalised above
                 0.35 to leave a 0.15 margin under the 0.5 gate; plus a stance-phase foot-speed
                 penalty (load > 0.3 W); the old aggregate F_n-weighted squared slip term is
                 removed because it is not the acceptance metric
  curriculum     forward-speed target 0.05 -> 0.12 -> 0.20 -> 0.30; a level advances only if
                 the deterministic 4 s evaluation passes support, non-foot, height, slip and
                 progress gates. No threshold is ever relaxed to let a level pass.

Limits declared in advance: the residual may only use proprioception (no phase, no time, no
teacher, no contact labels beyond the sensed foot load); the base network's own-target history
is its only memory; acceptance remains physics on the current body.
"""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = dict(
    schema='criterion-matched-residual-v2/1',
    scope='S5 replacement-ANN locomotion only; no connectome claim, no injury experiment',
    base_network=str(HERE.parent / '20261108T011500Z/MODEL_MEASURED.npz'),
    base_status='supervised on filtered measured motion; open-loop it collapses (support .607, non-foot 31.45 W, slip max 1.139)',
    residual=dict(action_joints='tibia pitch of all six legs, one per leg',
                  action_indices=[6, 13, 20, 27, 34, 41],
                  bound_rad_per_tick=.003, hidden=[128, 128], activation='tanh',
                  obs='183 proprioception + 294 own-target history (unchanged contract)',
                  std_bounds=[1e-05, 10 ** -1.5]),
    reward=dict(progress='clip(filtered_forward / v_target, -1, 1.5)',
                slip_hinge='2.0 * sum_i max(0, r_i - 0.35)^2 with r_i = vt_i / max(|filtered_forward|, 0.05)',
                stance_foot_speed='0.5 * sum_i (stance_i * vt_i^2), stance_i = load_i > 0.3 W',
                attitude='0.05 * |omega|^2 + 0.01 * gravity_xy^2',
                action='0.01 * mean(action^2)',
                support='5 * (sum(loads)/W - 1)^2 + 50 * nonfoot/W',
                termination='COM height < 0.95 or non-foot force > 0.05 W'),
    curriculum=dict(v_targets=[.05, .12, .20, .30], updates_per_level=32, rollout=256, decisions_per_update=256,
                    epochs=6, minibatch=64, lr=3e-4, clip=.2, gamma=.995, gae=.95, value_coef=.5, grad_norm=.5,
                    target_kl=.03, episode_total_s=8., warmup_s=1.),
    level_gates=dict(foot_support_mean_min=.90, nonfoot_force_max_W=.20 * 24.446354755878005,
                     com_height_min=1.20, slip_ratio_max=.5, forward_speed_min=.05,
                     note='gates are the frozen acceptance quantities; my evaluator recomputes them, it is not the frozen D-gate suite, and that is stated in every result file'),
    artifacts=['PREREG.json', 'TRAIN_LOG.json', 'LEVEL_*/RESULT.json', 'MODEL_RESIDUAL_V2.npz', 'SUMMARY.json'],
    failure_policy='if a level fails its gates after the allocated updates, the run stops at that level and the failure is reported with the measured numbers; no gate, bound or reward weight is adjusted to force a pass',
    accounting='every run reports native physics steps, policy ticks and wall time; no hidden drive, no root/pose write, no external oscillator',
)
(HERE / 'PREREG_RESIDUAL_V2.json').write_text(json.dumps(PREREG, indent=2) + '\n')
print(json.dumps({'prereg': str(HERE / 'PREREG_RESIDUAL_V2.json'), 'levels': PREREG['curriculum']['v_targets']}))
