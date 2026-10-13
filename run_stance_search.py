"""Static stance search: can full body-weight support be reached with small penetration?

Read-only, static (no dynamics, no controller, no gain). Declared assumption:
  a standing posture must be symmetric — the same joint offsets on all six legs — with the
  root height as one extra parameter (8 parameters total).
Objective, fixed before seeing results:
  reach foot support = body weight while keeping every contact penetration no deeper than the
  deployed pose's own worst penetration (0.05 mm) and keeping body (non-foot) contacts at zero.
The search is deterministic (fixed seed) and bounded; failure to reach the target is a result,
not something to be tuned away.
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
XML = HERE / 'outputs/model_branches/fine_clock/fruitfly_zero_floor_25us.xml'
OUT = HERE / 'outputs/model_branches/stance_search'
PENETRATION_LIMIT_MM = 0.05
PARAMS = ['Coxa_yaw', 'Coxa_pitch', 'Coxa_roll', 'Trochanter_yaw', 'Trochanter_pitch',
          'Trochanter_roll', 'Tibia_pitch']
BOUNDS = {'Coxa_yaw': .25, 'Coxa_pitch': .25, 'Coxa_roll': .20, 'Trochanter_yaw': .25,
          'Trochanter_pitch': .30, 'Trochanter_roll': .20, 'Tibia_pitch': .30}
SEED = 20261012
N_RANDOM = 6000
N_REFINE = 3000
START = time.time()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(report):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))
    _, pose, _, joints, geom_leg = closure.prepare_body()
    weight = float(model.body_mass.sum()) * abs(float(model.opt.gravity[2]))
    floor = closure.cp.floor_geom_id(model)
    # Map (leg, joint-kind) -> qpos address.
    addr = {}
    for leg in closure.LEGS:
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ''
            m = re.match(rf'joint_{leg.upper()}([A-Za-z]+_(?:yaw|pitch|roll))$', name)
            if m and m.group(1) in PARAMS:
                addr[(leg, m.group(1))] = int(model.jnt_qposadr[j])
    assert len(addr) == 6 * 7, len(addr)
    d = mujoco.MjData(model)
    buf = np.zeros(6)

    def evaluate(vec, dz):
        q = pose.copy()
        q[2] = pose[2] + dz
        for leg in closure.LEGS:
            for i, kind in enumerate(PARAMS):
                q[addr[(leg, kind)]] = pose[addr[(leg, kind)]] + vec[i]
        d.qpos[:] = q
        d.qvel[:] = 0
        d.act[:] = 0
        d.ctrl[:] = 0
        mujoco.mj_forward(model, d)
        foot = 0.0
        other_contacts = 0
        deepest = 0.0
        per_leg = {leg: 0.0 for leg in closure.LEGS}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if floor not in (g1, g2):
                continue
            mujoco.mj_contactForce(model, d, i, buf)
            frame = np.asarray(c.frame).reshape(3, 3)
            vertical = float((frame.T @ buf[:3])[2])
            deepest = min(deepest, float(c.dist))
            leg = geom_leg.get(g1) if g1 in geom_leg else geom_leg.get(g2)
            if leg:
                foot += vertical
                per_leg[leg] += vertical
            else:
                other_contacts += 1
        return dict(support=foot / weight, other_contacts=other_contacts,
                    deepest_penetration_mm=deepest, per_leg={k: v / weight for k, v in per_leg.items()})

    def cost(ev):
        pen_excess = max(0.0, -ev['deepest_penetration_mm'] - PENETRATION_LIMIT_MM)
        return abs(ev['support'] - 1.0) + 5.0 * ev['other_contacts'] + 20.0 * pen_excess

    rng = np.random.default_rng(SEED)
    best = None
    history = []
    for k in range(N_RANDOM):
        vec = np.array([rng.uniform(-BOUNDS[p], BOUNDS[p]) for p in PARAMS])
        dz = float(rng.uniform(-0.15, 0.15))
        ev = evaluate(vec, dz)
        c = cost(ev)
        history.append(c)
        if best is None or c < best['cost']:
            best = dict(cost=c, vec=vec.copy(), dz=dz, **ev)
    base_vec, base_dz = best['vec'].copy(), best['dz']
    step = np.array([BOUNDS[p] for p in PARAMS]) * 0.25
    step_dz = 0.03
    for k in range(N_REFINE):
        cand = base_vec + rng.normal(0, 1, len(PARAMS)) * step * (1 - k / N_REFINE) ** 2
        cand_dz = base_dz + rng.normal(0, step_dz) * (1 - k / N_REFINE) ** 2
        ev = evaluate(cand, cand_dz)
        c = cost(ev)
        if c < best['cost']:
            best = dict(cost=c, vec=cand, dz=cand_dz, **ev)
            base_vec, base_dz, step = cand, cand_dz, step * 0.9
    feasible = (abs(best['support'] - 1.0) <= 0.05 and best['other_contacts'] == 0
                and -best['deepest_penetration_mm'] <= PENETRATION_LIMIT_MM)
    result = dict(
        status='STANCE_SEARCH_FEASIBLE' if feasible else 'STANCE_SEARCH_TARGET_NOT_REACHED',
        label='manual read-only static search, not a controller and not a standing result',
        declared_target=dict(support_ratio=1.0, tolerance=0.05, body_contacts=0,
                             max_penetration_mm=PENETRATION_LIMIT_MM,
                             symmetry='the same 7 joint offsets on all six legs, plus root height'),
        search=dict(parameters=PARAMS, bounds_rad=BOUNDS, seed=SEED, random_samples=N_RANDOM,
                    refinement_samples=N_REFINE, evaluations=N_RANDOM + N_REFINE),
        best=dict(dz_mm=float(best['dz']), offsets_rad={p: float(v) for p, v in zip(PARAMS, best['vec'])},
                  support_ratio=float(best['support']), body_contacts=int(best['other_contacts']),
                  deepest_penetration_mm=float(best['deepest_penetration_mm']),
                  per_leg_share=best['per_leg'], cost=float(best['cost'])),
        deployed_reference=evaluate(np.zeros(len(PARAMS)), 0.0),
        feasible=bool(feasible),
        interpretation=['A feasible entry would mean the body CAN be configured to carry its own weight on six '
                        'loaded feet without deep penetration; that posture would then be the honest target for a '
                        'closed-loop episode.',
                        'If no feasible entry exists, the limitation is the body model geometry itself, not the '
                        'neural command pattern.'],
        provenance=dict(xml=str(XML), xml_sha256=sha(XML), weight_model_units=weight),
        limitations=['Symmetric parameterisation only; per-leg asymmetry is not explored.',
                     'Static support is necessary, not sufficient, for standing.',
                     'No dynamics were run; the found posture, if any, still needs an episode with cut/replay controls.',
                     'The cost function is declared here and was not tuned after seeing results.'],
        memory_guard=report('completed'), wall_s=time.time() - START)
    if not np.isfinite(result['memory_guard'].get('previous_soft_limit_mb', 0.0)):
        result['memory_guard']['previous_soft_limit_mb'] = 'unlimited'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stance_search.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], best=result['best'],
                          deployed=result['deployed_reference']), indent=2))
    return result


if __name__ == '__main__':
    with closure.mg.guarded_run('stance_search', **closure.mg.BUDGETS['neural_muscle_closure']) as report:
        main(report)
