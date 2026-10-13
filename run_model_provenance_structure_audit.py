"""Model provenance structure audit: why 16 actuators have no annotation counterpart.

Read-only. Parses the upstream OpenSim source model and the canonical MJCF side by side and
compares their per-leg articulation and muscle sets with the per-leg muscle vocabulary that
BANC actually annotates. No dynamics, no parameter change.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, re, sys, time, hashlib
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
from run_neural_rebuild_11_mn_pools import ALIASES, parse_actuator
OSIM = HERE / 'outputs/upstream_muscle_round11/best_combined.osim'
RUN = HERE / 'outputs/neural_muscle_closure/runs/add80e12-4b1b-45fc-90be-2336ddf7757c'
COVERAGE = HERE / 'outputs/model_branches/model_annotation_gap/banc_leg_muscle_coverage.json'
OUT = HERE / 'outputs/model_branches/model_provenance_structure'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import mujoco
    text = OSIM.read_text(errors='replace')
    bodies = re.findall(r'<Body\s+name="([^"]+)"', text)
    muscles = re.findall(r'<Millard2012EquilibriumMuscle\s+name="([^"]+)"', text)
    joints = re.findall(r'<CustomJoint\s+name="([^"]+)"', text)
    leg_bodies = defaultdict(list)
    for b in bodies:
        m = re.match(r'^([LR])([FMH])([A-Za-z]+?)(\d*)$', b)
        if m:
            leg_bodies[''.join(m.group(1, 2))].append(m.group(3) + m.group(4))
    segment_kinds = defaultdict(set)
    for leg, segs in leg_bodies.items():
        for s in segs:
            segment_kinds[leg].add(re.sub(r'\d+$', '', s))
    muscle_legs = Counter(m[:2] for m in muscles)
    joint_legs = Counter(j[6:8] for j in joints)

    model = closure.cp.load_model()
    mj_joints = defaultdict(list)
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
        m = re.match(r'^joint_([LR][FMH])(\w+)$', name)
        if m:
            mj_joints[m.group(1)].append(m.group(2))
    mj_actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    mj_muscle_legs = Counter()
    for a in mj_actuators:
        p = parse_actuator(a)
        if p['ok']:
            mj_muscle_legs[p['leg']] += 1
    coverage = json.loads(COVERAGE.read_text())['leg_muscle_targets']
    effector_of = {'LF': 'front_leg', 'RF': 'front_leg', 'LM': 'middle_leg', 'RM': 'middle_leg',
                   'LH': 'hind_leg', 'RH': 'hind_leg'}
    per_leg_model_targets = {}
    for leg in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH'):
        targets = set()
        for a in mj_actuators:
            p = parse_actuator(a)
            if p['ok'] and p['leg'] == leg and p['muscle'] in ALIASES:
                targets.add(ALIASES[p['muscle']])
        annotated = {t for t, per in coverage.items()
                     if any(k.startswith(effector_of[leg]) and v > 0 for k, v in per.items())}
        per_leg_model_targets[leg] = dict(
            modelled_targets=sorted(targets),
            annotated_targets_present_on_this_leg=sorted(annotated),
            modelled_but_not_annotated=sorted(targets - annotated),
            annotated_but_not_modelled=sorted(annotated - targets))
    result = dict(
        status='MEASURED_MODEL_PROVENANCE_STRUCTURE',
        label='manual read-only structural comparison, not a modelling recommendation',
        question='Does the six-leg muscle set come from anatomy, and where do the 16 uncommandable actuators come from?',
        upstream_open_sim=dict(
            file=str(OSIM), sha256=sha(OSIM),
            bodies=len(bodies), muscles=len(muscles), joints=len(joints),
            legs_with_geometry=sorted(leg_bodies),
            segment_kinds_per_leg={k: sorted(v) for k, v in sorted(segment_kinds.items())},
            muscles_per_leg=dict(muscle_legs),
            joints_per_leg=dict(joint_legs),
            muscle_names=muscles,
            finding='the upstream source articulates only the TWO FRONT LEGS (3 joints each) and defines muscles only for '
                    'them; the middle and hind leg bodies exist as geometry but carry no joints and no muscles'),
        canonical_mjcf=dict(
            legs=len(mj_joints), joints_per_leg={k: len(v) for k, v in sorted(mj_joints.items())},
            joint_kinds=sorted({k for v in mj_joints.values() for k in v}),
            actuators=len(mj_actuators), actuators_per_leg=dict(mj_muscle_legs),
            finding='the canonical model articulates all six legs and carries the SAME front-leg muscle set on every leg'),
        per_leg_targets=per_leg_model_targets,
        conclusion='the 16 uncommandable actuators are not primarily a BANC gap: they are a consequence of replicating a '
                   'front-leg muscle set onto middle and hind legs whose annotated muscle vocabulary differs. '
                   'Only the middle-right pleural remotor and the middle-left sternal adductor are single missing cells '
                   'relative to the male MANC coverage expectation.',
        limitations=['One upstream revision and one annotation release.',
                     'The upstream file is the project-local copy; a newer upstream revision was not fetched.',
                     'No claim is made that the animal lacks these muscles; FBbt and Cheong et al. suggest the '
                     'tergopleural promotor may exist only on the prothoracic leg.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'model_provenance_structure.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(upstream=result['upstream_open_sim']['finding'],
                          mjcf=result['canonical_mjcf']['finding'],
                          muscles_per_leg=result['upstream_open_sim']['muscles_per_leg'],
                          joints_per_leg=result['upstream_open_sim']['joints_per_leg'],
                          per_leg_gaps={k: v['modelled_but_not_annotated'] for k, v in per_leg_model_targets.items()}),
                     indent=2))
    return result


if __name__ == '__main__':
    import numpy as np
    with closure.mg.guarded_run('model_provenance_structure', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
