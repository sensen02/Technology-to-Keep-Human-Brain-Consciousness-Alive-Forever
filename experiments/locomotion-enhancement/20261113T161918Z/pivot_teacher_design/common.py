"""Lazy physical imports and exact selected legacy AST. No execution on import."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import sys
import types

ROOT = Path(__file__).resolve().parent
BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
PEAK_SOURCE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/food-approach/20261111T224234Z/runtime.py')
SOURCE_CANDIDATE = 6
FILES = ('teacher.py', 'common.py', 'preflight.py', 'runner.py', 'audit_saved.py', 'run.py', 'PREREG.json')


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jsonable(v):
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)): return [jsonable(x) for x in v]
    if isinstance(v, Path): return str(v)
    if type(v).__module__.startswith('numpy'): return v.tolist() if hasattr(v, 'tolist') else v.item()
    return v


def save(p, value):
    with Path(p).open('x') as stream:
        json.dump(jsonable(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def read(p): return json.loads(Path(p).read_text())


def frozen():
    paths = [ROOT/n for n in FILES] + [BASE/'student_wave.py', BASE/'physics.py', BASE/'static_teacher.py', BASE/'wave_reference.py', BASE/'identity.py', PEAK_SOURCE]
    return {str(p): sha(p) for p in paths}


def verify(hashes):
    for p, digest in hashes.items():
        if sha(p) != digest: raise ValueError('Source drift: ' + p)


def cpu1():
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1': raise ValueError(name+'=1 required')
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})


def selected_ast(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    selected = []
    found = set()
    for node in tree.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in names:
            selected.append(node); found.add(name)
    if found != set(names): raise ValueError('Legacy AST names missing')
    body = ast.Module(body=selected, type_ignores=[])
    exec(compile(body, str(path)+'::exact_selected_legacy_AST', 'exec'), namespace)
    return hashlib.sha256(ast.dump(body, include_attributes=False).encode()).hexdigest()


def physical_stack():
    import teacher
    teacher.pin_baseline()
    import numpy as np
    import physics, identity, static_teacher
    ns = dict(np=np, Path=Path, sha256=identity.sha256, MAX_RATE=physics.MAX_RATE, summarise=physics.summarise)
    names = ('SOURCE', 'teacher_config', 'no_external', 'bounded_action', 'Trace', 'settle')
    old_trace_ast = selected_ast(BASE/'student_wave.py', names, ns)
    peak_ns = dict(np=np)
    old_peak_ast = selected_ast(PEAK_SOURCE, ('PeakRecorder',), peak_ns)
    if 'torch' in sys.modules: raise ValueError('Teacher-only stack unexpectedly imported Torch')
    return types.SimpleNamespace(**ns), physics, identity, static_teacher, peak_ns['PeakRecorder'], dict(legacy_trace_selected_AST_sha256=old_trace_ast, legacy_peak_selected_AST_sha256=old_peak_ast)


def snapshot(c):
    return {k: getattr(c.d, k).copy() for k in ('qpos', 'qvel', 'ctrl', 'xfrc_applied', 'qfrc_applied')} | dict(target=c.actuator.target_rad.copy(), buffer=c.buf.copy())


def assert_same(before, after):
    import numpy as np
    if before.keys() != after.keys() or any(not np.array_equal(before[k], after[k]) for k in before):
        raise ValueError('Physical handover/probe wrote actual state')


def yaw(c):
    import numpy as np
    rot = c.d.xmat[c.th].reshape(3, 3)
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


class PivotCycles:
    """Measurement only: actual loaded10ms -> unloaded5ms+clearance -> reload10ms.
    No forward-advance requirement. Feed warm last10ms to arm initial support.
    """
    def __init__(self, weight):
        self.weight = weight
        self.states = [dict(armed=False, load=0, low=0, active=False, peak=0., reload=0, age=0) for _ in range(6)]
        self.counts = [0]*6
        self.events = []

    def update(self, t, loads, padz):
        for i, s in enumerate(self.states):
            loaded = loads[i]/self.weight >= .1 and padz[i] <= .004
            unloaded = loads[i]/self.weight <= .05
            if not s['active']:
                if not s['armed']:
                    s['load'] = s['load']+1 if loaded else 0
                    if s['load'] >= 10: s['armed'] = True
                    continue
                s['low'] = s['low']+1 if unloaded else 0
                s['peak'] = max(s['peak'], float(padz[i])) if unloaded else 0.
                if s['low'] >= 5: s['active'] = True; s['age'] = 5
                else: continue
            else:
                s['age'] += 1
                if unloaded: s['peak'] = max(s['peak'], float(padz[i]))
            s['reload'] = s['reload']+1 if loaded else 0
            if s['age'] > 4000 or (loaded and s['peak'] <= .015):
                self.states[i] = dict(armed=False, load=0, low=0, active=False, peak=0., reload=0, age=0)
            elif s['peak'] > .015 and s['reload'] >= 10:
                self.counts[i] += 1
                self.events.append(dict(time=float(t), leg_index=i, peak_clearance=float(s['peak']), unloaded_dwell_ms=5, reload_dwell_ms=s['reload']))
                self.states[i] = dict(armed=True, load=10, low=0, active=False, peak=0., reload=0, age=0)


def candidate_metrics(rows, geometry, boundary, body_length, weight, condition, cycles):
    import numpy as np
    com = np.asarray([r['com'] for r in rows]); loads = np.asarray([r['loads'] for r in rows]); nf = np.asarray([r['nonfoot'] for r in rows])
    yy = np.unwrap(np.r_[boundary['yaw'], [g['thorax_yaw'] for g in geometry]])
    delta = com[:, :2]-np.asarray(boundary['com'])[:2]
    duration = float(rows[-1]['time'])-boundary['time']
    yaw_change = float(yy[-1]-yy[0]); drift = float(np.linalg.norm(delta, axis=1).max()/body_length)
    signed = yaw_change if condition == 'left_pivot' else -yaw_change
    com_velocity = np.diff(np.vstack([boundary['com'], com]), axis=0)/.001
    rotations = np.asarray([g['thorax_rotation'] for g in geometry])
    body_velocity = np.einsum('tji,tj->ti', rotations, com_velocity)/body_length
    force_integral = loads.sum(0)*.001
    contact_speed = np.asarray([r['speed'] for r in rows])
    tangent_speed = (loads*contact_speed).sum(0)/np.maximum(loads.sum(0), 1e-12)/body_length
    gates = dict(support=float(loads.sum(1).mean()/weight) >= .9, nonfoot_1ms=float(nf.max()) == 0., com=float(com[:, 2].min()) >= 1.2, drift=drift <= .5)
    if condition != 'stop': gates.update(yaw=math_degrees(signed) >= 45., all6_actual_pivot_cycles=all(n >= 1 for n in cycles.counts))
    return dict(duration_s=duration, body_velocity_mean_bodylength_per_s=body_velocity.mean(0).tolist(), body_velocity_RMS_bodylength_per_s=np.sqrt((body_velocity**2).mean(0)).tolist(), body_velocity_absmax_bodylength_per_s=np.abs(body_velocity).max(0).tolist(), per_leg_loaded_tangent_speed_bodylength_per_s=[float(v) if den > 1e-12 else None for v, den in zip(tangent_speed, force_integral)], per_leg_load_integral_native_force_s=force_integral.tolist(), yaw_change_rad=yaw_change, yaw_change_deg=math_degrees(yaw_change), achieved_yaw_rate_rad_s=yaw_change/duration, max_COM_xy_drift_bodylength=drift, final_COM_xy_drift_bodylength=float(np.linalg.norm(delta[-1])/body_length), COM_xy_path_bodylength=float(np.linalg.norm(np.diff(np.vstack([boundary['com'], com]), axis=0)[:, :2], axis=1).sum()/body_length), support_mean_over_weight=float(loads.sum(1).mean()/weight), min_per_leg_load_over_weight=(loads.min(0)/weight).tolist(), nonfoot_1ms_peak=float(nf.max()), COM_min_native=float(com[:, 2].min()), pivot_cycle_counts=list(cycles.counts), pivot_cycle_events=list(cycles.events), gates=gates, task_pass_candidate=None if condition == 'stop' else bool(all(gates.values())), stop_pass='NOT_DEFINED_SPEED_THRESHOLD_PENDING' if condition == 'stop' else 'NOT_APPLICABLE')


def math_degrees(x):
    import math
    return math.degrees(x)
