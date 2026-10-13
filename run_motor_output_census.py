"""Motor output census: which muscles can be driven at all, and which actually are.

Read-only. Joins the published motor manifest (pool -> muscle) with the saved episode's
applied commands and per-cell spike counts, so the report separates
  (a) actuators with no neural pool at all (structurally uncommandable),
  (b) actuators whose pool produced no spikes,
  (c) actuators that did receive drive,
and checks antagonist pairs per joint family for one-sided drive.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, re, sys, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
RUN = HERE / 'outputs/neural_muscle_closure/runs/add80e12-4b1b-45fc-90be-2336ddf7757c'
TRACE = HERE / 'outputs/model_branches/clock_refinement/clock_refinement_traces.npz'
OUT = HERE / 'outputs/model_branches/motor_output_census'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def family(muscle):
    """Coarse joint family and direction token for antagonist grouping."""
    if not muscle:
        return None, None
    m = muscle.replace('_muscle', '')
    direction = 'other'
    for token in ('flexor', 'extensor', 'rotator', 'remotor', 'promotor', 'adductor', 'abductor'):
        if token in m:
            direction = token
            break
    if 'tibia' in m:
        joint = 'tibia'
    elif 'trochanter' in m:
        joint = 'trochanter'
    elif 'pleural' in m or 'sternal' in m or 'tergopleural' in m:
        joint = 'coxa'
    else:
        joint = 'other'
    return joint, direction


def main(report):
    manifest = json.loads((RUN / 'motor_manifest.json').read_text())['measured']
    rows = manifest['rows']
    with np.load(TRACE) as z:
        controls = z['25us_normal_controls']
        spikes = z['25us_normal_spike_counts']
    assert controls.shape == (40, len(rows)), (controls.shape, len(rows))
    census = []
    for r in rows:
        idx = np.asarray(r.get('local_indices', []), dtype=int)
        muscle = r.get('muscle_target') or r.get('source_annotation', {}).get('muscle_target')
        joint, direction = family(muscle)
        cell_spikes = spikes[:, idx].sum() if idx.size else 0
        command = controls[:, r['actuator_index']]
        census.append(dict(
            actuator=r['actuator'], actuator_index=int(r['actuator_index']),
            leg=r.get('leg') or r.get('source_annotation', {}).get('leg'),
            muscle_target=muscle, joint_family=joint, direction=direction,
            pool_status=r['status'], pool_cells=int(idx.size),
            pool_spikes_total=int(cell_spikes),
            intervals_with_spikes=int(np.count_nonzero(spikes[:, idx].sum(axis=1)) if idx.size else 0),
            command_max=float(command.max()), command_mean=float(command.mean()),
            intervals_command_positive=int(np.count_nonzero(command > 0)),
        ))
    uncommandable = [c for c in census if c['pool_cells'] == 0]
    silent = [c for c in census if c['pool_cells'] > 0 and c['pool_spikes_total'] == 0]
    driven = [c for c in census if c['pool_spikes_total'] > 0]
    # Antagonist check inside each (leg, joint family).
    groups = {}
    for c in census:
        if not c['muscle_target']:
            continue
        groups.setdefault((c['leg'], c['joint_family']), []).append(c)
    antagonists = {}
    for key, members in sorted(groups.items()):
        directions = sorted({m['direction'] for m in members})
        if len(directions) < 2:
            continue
        antagonists['/'.join(map(str, key))] = dict(
            directions=directions,
            driven_directions=sorted({m['direction'] for m in members if m['pool_spikes_total'] > 0}),
            per_direction={d: dict(actuators=[m['actuator'] for m in members if m['direction'] == d],
                                   spikes=int(sum(m['pool_spikes_total'] for m in members if m['direction'] == d)),
                                   command_mean=float(np.mean([m['command_mean'] for m in members if m['direction'] == d])))
                           for d in directions})
    total_pool_spikes = int(sum(c['pool_spikes_total'] for c in census))
    result = dict(
        status='MEASURED_MOTOR_OUTPUT_CENSUS',
        label='manual read-only output census, not a controller and not a recruitment model',
        question='Which modelled muscles can receive neural drive, which actually do, and is antagonist drive one-sided?',
        source=dict(manifest=str(RUN / 'motor_manifest.json'), manifest_sha256=sha(RUN / 'motor_manifest.json'),
                    trace=str(TRACE), trace_sha256=sha(TRACE), episode='25us_normal, 200 ms, 40 command intervals'),
        summary=dict(actuators=len(census), uncommandable_no_pool=len(uncommandable),
                     silent_pool_no_spikes=len(silent), driven=len(driven),
                     pool_cells_total=int(sum(c['pool_cells'] for c in census)),
                     pool_spikes_total=total_pool_spikes,
                     commands_all_zero=int(np.count_nonzero(controls.max(axis=1) == 0))),
        uncommandable=[{k: c[k] for k in ('actuator', 'leg', 'muscle_target', 'pool_status')} for c in uncommandable],
        silent_pools=[{k: c[k] for k in ('actuator', 'leg', 'muscle_target', 'pool_cells')} for c in silent],
        antagonist_groups=antagonists,
        census=census,
        interpretation=['An actuator with no pool has no neural path, so its command is structurally fixed regardless of sensing or network dynamics.',
                        'A pool that never spikes is present but not driven in this episode.',
                        'One-sided antagonist drive means the joint family received torque in one direction only during the episode.'],
        limitations=['One episode and one tier; pool membership is the existing annotation, not re-derived.',
                     'Spike counts are the published per-5ms counts, not a new measurement.',
                     'No recruitment hypothesis is validated here; shared and partial pools are reported as-is.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'motor_output_census.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(OUT / 'motor_output_census.npz',
                        actuator=np.asarray([c['actuator'] for c in census]),
                        leg=np.asarray([str(c['leg']) for c in census]),
                        muscle=np.asarray([str(c['muscle_target']) for c in census]),
                        pool_cells=np.asarray([c['pool_cells'] for c in census]),
                        pool_spikes_total=np.asarray([c['pool_spikes_total'] for c in census]),
                        command_mean=np.asarray([c['command_mean'] for c in census]),
                        command_max=np.asarray([c['command_max'] for c in census]))
    print(json.dumps(result['summary'], indent=2))
    print(json.dumps(antagonists, indent=2)[:2500])
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('motor_output_census', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
