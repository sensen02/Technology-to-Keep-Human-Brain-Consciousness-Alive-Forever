"""Model-versus-annotation structural gap audit.

Read-only. Compares what the canonical body model can actuate with what the BANC
annotation names as leg muscle targets, in both directions:
  forward gap  = model actuators whose muscle has no annotated motor pool for that leg
  reverse gap  = annotated leg muscle targets with no actuator in the model at all
No substitution, mirroring or guessing is performed or recommended.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import json, re, sys, time, hashlib
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_neural_muscle_closure as closure
from run_neural_rebuild_11_mn_pools import ALIASES, LEG_OF, parse_actuator
FEATHER = HERE / 'data/banc888/banc_888_meta.feather'
RUN = HERE / 'outputs/neural_muscle_closure/runs/add80e12-4b1b-45fc-90be-2336ddf7757c'
OUT = HERE / 'outputs/model_branches/model_annotation_gap'
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


COVERAGE_JSON = HERE / 'outputs/model_branches/model_annotation_gap/banc_leg_muscle_coverage.json'


def main(report):
    import mujoco
    xml_path = closure.cp.CANONICAL_XML
    text = Path(xml_path).read_text()
    block = text[text.index('<actuator>'):text.index('</actuator>')]
    actuators = re.findall(r'name="([^"]+)"', block)
    model = closure.cp.load_model()
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    leg_joints = sorted(j for j in joints if j and re.match(r'^joint_[LR][FMH]', j))
    dof_kinds = Counter(re.sub(r'^joint_[LR][FMH]', '', j) for j in leg_joints)

    parsed = {a: parse_actuator(a) for a in actuators}
    unparsed = [a for a, p_ in parsed.items() if not p_['ok']]
    no_alias = sorted({p_['muscle'] for p_ in parsed.values() if p_['ok'] and p_['muscle'] not in ALIASES})
    model_muscles = sorted({p_['muscle'] for p_ in parsed.values() if p_['ok']})
    model_targets = sorted({ALIASES[p_['muscle']] for p_ in parsed.values()
                            if p_['ok'] and p_['muscle'] in ALIASES})
    if not COVERAGE_JSON.exists():
        raise FileNotFoundError(f'missing {COVERAGE_JSON}; regenerate with venv/bin/python '
                                f'tools_export_banc_leg_muscle_coverage.py (pandas interpreter)')
    exported = json.loads(COVERAGE_JSON.read_text())
    assert exported['feather_sha256'] == sha(FEATHER), 'exported coverage is stale'
    coverage = exported['leg_muscle_targets']
    annotated_targets = sorted(coverage)
    manifest = json.loads((RUN / 'motor_manifest.json').read_text())['measured']
    unresolved = [r for r in manifest['rows'] if r['status'] == 'unresolved']
    result = dict(
        status='MEASURED_MODEL_VS_ANNOTATION_GAP',
        label='manual read-only structural gap audit, not a recommendation to substitute anything',
        model=dict(xml=str(xml_path), xml_sha256=sha(xml_path), actuators=len(actuators),
                   distinct_muscle_names=len(model_muscles), muscle_names=model_muscles,
                   unparsed_actuator_names=unparsed, muscles_without_alias=no_alias,
                   distinct_annotated_targets_of_the_model=model_targets,
                   leg_joint_kinds=dict(dof_kinds), leg_joints=len(leg_joints),
                   dof_per_leg=len(leg_joints) // 6),
        annotation=dict(feather=str(FEATHER), feather_sha256=sha(FEATHER),
                        efferent_leg_cells=int(exported['efferent_leg_cells']),
                        leg_muscle_targets={k: coverage[k] for k in annotated_targets},
                        coverage_export=str(COVERAGE_JSON), coverage_export_sha256=sha(COVERAGE_JSON)),
        reverse_gap=dict(
            meaning='annotated leg muscle targets with no actuator in the canonical model',
            targets=[t for t in annotated_targets if t not in model_targets],
        ),
        forward_gap=dict(
            meaning='model actuator muscles with no annotated motor pool on their own leg',
            unresolved_actuators=len(unresolved),
            by_reason=dict(Counter(
                'target absent from the dataset entirely' if r['source_annotation'].get('target_exists_on') == {}
                else 'target annotated only on other legs' if all(v == 0 for k, v in r['source_annotation'].get('target_exists_on', {}).items() if k == LEG_OF[r['source_annotation']['leg']]) and r['source_annotation'].get('target_exists_on')
                else 'target annotated on this leg but only the other side'
                for r in unresolved)),
            detail=[dict(actuator=r['actuator'], leg=r['source_annotation'].get('leg'),
                         target=r['source_annotation'].get('muscle_target'),
                         exists_on=r['source_annotation'].get('target_exists_on'))
                    for r in unresolved],
        ),
        interpretation=['The body model articulates coxa (3), trochanter (3) and tibia (1) per leg; it has no femur or tarsus joint.',
                        'Annotated tarsus/femur/long-tendon motor neurons therefore have no actuator to drive in this model.',
                        'Conversely some modelled muscles have no annotated motor pool for that leg, so they cannot be commanded.',
                        'Neither gap may be closed by mirroring cells across legs or sides; that would invent connectivity.'],
        limitations=['Single annotation release and single canonical model revision.',
                     'Coverage is counted from the efferent subset only.',
                     'No claim is made about what the animal has; this is a data/model coverage statement.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'model_annotation_gap.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(model=result['model'], reverse_gap=result['reverse_gap'],
                          forward_gap=dict(unresolved=result['forward_gap']['unresolved_actuators'],
                                           by_reason=result['forward_gap']['by_reason'])), indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('model_annotation_gap', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
