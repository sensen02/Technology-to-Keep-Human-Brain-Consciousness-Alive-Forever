"""End-to-end platform loop on the frozen body: energy, virtual food, feeding gate, selector.

Movement is produced ONLY by the frozen connectome module through the frozen bounded
readout (zero coefficients by default, i.e. the round-10 selected incumbent): this run
therefore documents what the platform does when the neural layer cannot walk. It is not a
walking result and no ANN or algorithmic gait is used here.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
ACT = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py')
PLANT = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-incremental/20261107T010000Z/plant_probe.py')
FOUND = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/foundation')
BASE = FOUND / 'BASE.npz'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=3.)
    ap.add_argument('--theta-scale', type=float, default=0.)
    ap.add_argument('--out', default=str(HERE / 'RUN_PLATFORM'))
    args = ap.parse_args()
    pf = load('pf', HERE / 'platform.py')
    act = load('act', ACT)
    probe = load('probe', PLANT)
    rt = load('rt', FOUND / 'runtime.py')
    import mujoco as mj
    model = act.load_full_model()
    d = probe.Plant().fresh()
    plant = probe.Plant()
    actuator = act.IncrementalActuator(model)
    actuator.reset(d)
    circ = rt.load_circuit()
    actor = rt.Actor(np.zeros(168) if args.theta_scale == 0 else np.full(168, args.theta_scale))
    base = np.load(BASE)
    leg_of_mn = np.asarray(base['mn_leg'])
    rostrum = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Rostrum')
    thorax = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Thorax')
    start_com = d.subtree_com[thorax].copy()
    heading = d.body(thorax).xipos - d.body(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Head')).xipos
    heading[2] = 0.
    heading /= np.linalg.norm(heading)
    lateral = np.array([-heading[1], heading[0], 0.])
    food_center = start_com + heading * .02
    plat = pf.PlatformStep(food=pf.FoodPatch(pf.FoodParams(center=tuple(food_center.tolist()), radius=.004)))
    steps = int(round(args.seconds / .005))
    rows = []
    physical = 0
    for k in range(steps):
        mn = circ.filtered_rates_hz[circ.mn_local]
        rate = actor.action(mn)
        actuator.update(rate, .001)
        for _ in range(5):
            d.ctrl[:] = actuator.torque(d)
            mj.mj_step(model, d); mj.mj_forward(model, d)
            physical += 1
            if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
                raise FloatingPointError('nonfinite body state at step %d' % physical)
            if d.warning.number.any():
                raise RuntimeError('MuJoCo warning at step %d' % physical)
        sens = np.zeros(20000)
        circ.advance(sens, np.zeros(20000), 5.)
        mouth = d.xpos[rostrum].copy()
        out = plat.step(.005, mouth)
        com = d.subtree_com[thorax].copy()
        rows.append(dict(t=k * .005, energy=out['energy'], food_mass=out['food_mass'],
                         decision=out['decision'], mouth_in_food=out['mouth_in_food'],
                         consumed=out['consumed'], mouth_distance=float(np.linalg.norm(mouth - np.asarray(food_center))),
                         foot_force_z=float(d.cfrc_ext[:, 2].sum()) if False else float(np.sum(d.actuator_force)),
                         com_x_lateral=float((com - start_com) @ lateral), com_x_forward=float((com - start_com) @ heading),
                         com_z=float(com[2]), mean_abs_rate=float(np.abs(rate).mean())))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    arr = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(out_dir / 'PLATFORM_TRACE.npz', **arr)
    decisions = {name: int(sum(1 for r in rows if r['decision'] == name)) for name in ('feed', 'forage', 'rest')}
    summary = dict(status='PLATFORM_LOOP_RUN_NOT_WALKING',
                   seconds=args.seconds, native_steps=physical, neural_steps=int(circ.steps),
                   theta_scale=args.theta_scale,
                   energy_start=float(rows[0]['energy']), energy_end=float(rows[-1]['energy']),
                   food_start=float(1.), food_end=float(rows[-1]['food_mass']),
                   decisions=decisions, min_mouth_distance=float(min(r['mouth_distance'] for r in rows)),
                   final_mouth_distance=float(rows[-1]['mouth_distance']),
                   forward_displacement=float(rows[-1]['com_x_forward']), lateral_displacement=float(rows[-1]['com_x_lateral']),
                   com_z_min=float(min(r['com_z'] for r in rows)), mean_abs_rate=float(np.mean([r['mean_abs_rate'] for r in rows])),
                   provenance=dict(movement='frozen_connectome_module_plus_frozen_bounded_readout',
                                   behaviour='engineering_selector_not_neural',
                                   feeding='declared_gate_on_virtual_food_no_food_collision'),
                   limits='No ANN or algorithmic gait is used here. The readout is the round-10 zero incumbent, so movement is expected to be negligible; this run tests platform wiring, not locomotion.',
                   source_hashes={str(p): sha(p) for p in (ACT, PLANT, FOUND / 'runtime.py', BASE, HERE / 'platform.py')})
    (out_dir / 'RESULT.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'native_steps', 'neural_steps', 'energy_start', 'energy_end',
                                              'food_end', 'decisions', 'min_mouth_distance', 'forward_displacement')}))


if __name__ == '__main__':
    main()
