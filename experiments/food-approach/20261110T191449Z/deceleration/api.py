"""Teacher-only deceleration APIs; no training or physics on import."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
TEACHER = HERE.parent/'teacher'/'turning_teacher.py'
_spec = importlib.util.spec_from_file_location('_deceleration_previous_turning_teacher', TEACHER)
previous = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = previous
_spec.loader.exec_module(previous)
previous.pin_baseline()
from cycle_reward import CycleConfig, CycleTracker

CONDITIONS = (
    dict(name='halfspeed_straight', speed_scale=.5, yaw_rate=0.),
    dict(name='halfspeed_smallleft', speed_scale=.5, yaw_rate=.004),
)

def make_teacher(core, case_name):
    """Return unchanged previous teacher, calibrated from actual supplied core."""
    cfg = next(c for c in CONDITIONS if c['name'] == case_name)
    return previous.make_teacher(core, speed_scale=cfg['speed_scale'], yaw_rate=cfg['yaw_rate'], prime_s=4.8)

def command2(reference, elapsed_s):
    phase, scale, curvature = reference.command(elapsed_s)
    physical = np.array([.075*scale, .075*scale*curvature])
    return np.array([scale-1., physical[1]/.008]), physical, phase

def halfspeed_gates(summary, cycles):
    return dict(forward=summary['forward_signed_native_per_s'] >= .025,
                slip=summary['slip_ratio_max'] <= .5,
                support=summary['foot_support_mean'] >= .90,
                com=summary['com_height_min'] >= 1.20,
                strict_nonfoot_zero=summary['nonfoot_force_max_native'] == 0,
                valid_slip=bool(all(summary['slip_valid_per_leg']) and summary['slip_has_progress']),
                all6cycles=cycles['all_legs_completed'])

def replay_cycles(trace):
    """Read-only sensed-cycle replay across complete modified window."""
    observer = CycleTracker(CycleConfig(timeout=4.)).reset(start_time=5.8)
    events = []
    for i in np.flatnonzero(trace['control_source'] == 5):
        events.extend(observer.update(dict(time=float(trace['time'][i]), loads=trace['loads'][i],
            padz=trace['padz'][i], relativefoot=trace['relativefoot'][i]))['events'])
    events.extend(observer.finalize()['events'])
    counts = {leg: sum(e['type']=='complete' and e['leg']==leg for e in events)
              for leg in ('LF','LH','LM','RF','RH','RM')}
    return dict(complete_per_leg=counts, all_legs_completed=all(v>=1 for v in counts.values()), events=events)

def load_curriculum_samples(case_name, *, require_verified=True, post_only=False):
    """Load existing evidence for future parent use; caller owns any later training.

    command2 is separate from X238. Phase and actual time are provenance only.
    Reject failed/unverified halfspeed evidence by default. Never runs a simulation.
    """
    import json
    if case_name not in [c['name'] for c in CONDITIONS]:
        raise ValueError('Only the two preregistered cases are available')
    if require_verified:
        verification = json.loads((HERE/'VERIFICATION.json').read_text())
        result = json.loads((HERE/case_name/'RESULT.json').read_text())
        if not verification['all_integrity_checks_passed'] or not result['all_halfspeed_gates']:
            raise ValueError('Case is not independently verified passing halfspeed evidence')
        from hashlib import sha256
        p = HERE/case_name/'SAMPLES_5MS.npz'
        if sha256(p.read_bytes()).hexdigest() != verification['artifact_sha256'][str(p)]:
            raise ValueError('Saved curriculum samples changed since verification')
    with np.load(HERE/case_name/'SAMPLES_5MS.npz', allow_pickle=False) as z:
        mask = z['post_prime'] if post_only else np.ones(len(z['actual_time']), dtype=bool)
        return {k: z[k][mask].copy() for k in z.files}
