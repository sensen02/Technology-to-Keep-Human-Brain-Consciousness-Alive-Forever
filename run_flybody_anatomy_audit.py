"""FlyBody leg joint definitions: the source-based anatomy available locally.

Read-only. Extracts the published FlyBody MuJoCo model's per-leg joint classes (axis, range,
spring reference) from the installed flygym package, and records exactly what that does and
does not license for extending this project's body model. No model is modified.
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
OUT = HERE / 'outputs/model_branches/flybody_anatomy'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classes_with_joint(text):
    """Return class name -> joint attributes, for the leg default tree."""
    out = {}
    for m in re.finditer(r'<default class="([^"]+)">(.*?)</default>', text, re.S):
        name, body = m.group(1), m.group(2)
        jm = re.search(r'<joint([^>]*)/>', body)
        if not jm:
            continue
        attrs = dict(re.findall(r'(\w+)="([^"]*)"', jm.group(1)))
        out[name] = attrs
    return out


def main(report):
    import flygym
    pkg = Path(flygym.__file__).resolve().parent
    xml = pkg / 'assets/model/flybody/fruitfly.xml'
    assert xml.is_file(), xml
    text = xml.read_text()
    classes = classes_with_joint(text)
    leg_classes = {k: v for k, v in classes.items()
                   if any(t in k for t in ('coxa', 'femur', 'tibia', 'tarsus'))}
    actuators = re.findall(r'<(?:general|position|motor|muscle)\s+name="([^"]+)"', text)
    leg_actuators = sorted(a for a in actuators if re.search(r'coxa|femur|tibia|tars', a))
    per_leg = {}
    for token in ('T1', 'T2', 'T3'):
        per_leg[token] = [a for a in leg_actuators if a.endswith(f'_{token}_left')]
    # our canonical model's leg joint count for comparison
    import mujoco
    base = closure.cp.load_model()
    our_joints = sorted({(mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_JOINT, j) or '').split('_', 2)[-1]
                         for j in range(base.njnt)
                         if (mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_JOINT, j) or '').startswith('joint_')})
    result = dict(
        status='MEASURED_FLYBODY_JOINT_DEFINITIONS',
        label='manual read-only extraction of a published model definition, not a behaviour result',
        question='Does a published Drosophila model provide leg joint axes and ranges that this project lacks?',
        source=dict(package='flygym (installed)', file=str(xml), sha256=sha(xml),
                    note='FlyBody model shipped with flygym; the same package provides the canonical loader'),
        leg_joint_classes=leg_classes,
        tarsus_and_femur_summary=dict(
            tarsus_axis='(1 0 0) in the tarsus body frame, with per-leg ranges '
                        'T1 -0.7..1.2, T2 -1.0..1.8, T3 -0.8..1.2 and per-leg spring references',
            femur_joints='an extend and a twist joint exist per leg; extend axis (1 0 0) range -0.15..2 (T1) '
                         'and -0.7..1.5 (T3), twist axis (0 1 0) range -1..1',
            tibia_axis='(1 0 0) range -1.35..1.3 (same local axis as the tarsus, i.e. one flexion axis family)',
            actuators_per_leg=len(leg_actuators) // 6,
            actuator_names=per_leg),
        our_model=dict(canonical_joint_suffixes=our_joints, leg_joints_per_leg=int(base.njnt - 7) // 6),
        what_this_licenses=['the existence of a tibia-tarsus joint (and further tarsomere joints) in a published '
                            'Drosophila model',
                            'per-leg tarsus joint RANGES and spring references that can be cited instead of invented',
                            'confirmation that the leg flexion axis family is the same local axis for tibia and tarsus',
                            'confirmation that a femur articulation is modelled with an extend and a twist axis'],
        what_this_does_not_license=['copying an axis number directly: the body frames of the two models differ, so the '
                                    'axis must be matched through the segment geometry, not by value',
                                    'any muscle attachment geometry: this model drives joints, not individual muscles, '
                                    'so it provides no tarsus/femur muscle origin or insertion',
                                    'any transmitter, connectivity or pool assignment'],
        limitations=['Single installed version of the package; the model file is the shipped asset.',
                     'Frame correspondence between the two models is asserted by convention, not proven here.',
                     'No model was modified and no episode was run.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'flybody_anatomy.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(tarsus=result['tarsus_and_femur_summary']['tarsus_axis'],
                          femur=result['tarsus_and_femur_summary']['femur_joints'],
                          actuators_per_leg=result['tarsus_and_femur_summary']['actuators_per_leg'],
                          ours=result['our_model']), indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('flybody_anatomy', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
