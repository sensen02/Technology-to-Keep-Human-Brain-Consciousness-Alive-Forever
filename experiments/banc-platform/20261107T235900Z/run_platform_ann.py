"""Platform v2: energy/food/feeding/selector driven by a recorded, LABELLED replacement-ANN trajectory.

Movement label: replacement_ANN_not_connectome (the frozen u32 residual policy).
No native stepping happens here: qpos records are replayed through frozen kinematics
(one mj_forward per record) only to obtain the mouth-part position.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
ACT = ROOT / 'experiments/s5-ann-incremental/20261107T010000Z/incremental_actuator.py'
PF = ROOT / 'experiments/banc-platform/20261107T234820Z/platform_state.py'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', choices=('reachable', 'unreachable'), required=True)
    ap.add_argument('--radius', type=float, default=.05)
    ap.add_argument('--distance', type=float, default=None)
    ap.add_argument('--energy', type=float, default=.6)
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    dist = args.distance if args.distance is not None else (.30 if args.condition == 'reachable' else 3.0)
    sys.path.insert(0, str(ACT.parent))
    act = load('act', ACT)
    pf = load('pf', PF)
    import mujoco as mj
    model = act.load_full_model()
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)
    rostrum = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Rostrum')
    z = np.load(HERE / 'ANN_BODY_u32.npz')
    times = z['time']; qpos = z['qpos']
    mouths = np.zeros((times.size, 3)); coms = np.zeros((times.size, 3))
    for i in range(times.size):
        data.qpos[:] = qpos[i]; mj.mj_forward(model, data)
        mouths[i] = data.xpos[rostrum]; coms[i] = z['com'][i]
    heading = coms[-1] - coms[0]; heading[2] = 0.
    travelled = float(np.linalg.norm(heading))
    if travelled > 0:
        heading /= travelled
    else:
        heading = np.array([1., 0., 0.])
    start = mouths[0].copy()
    center = start + heading * dist
    plat = pf.PlatformStep(energy=pf.EnergyState(pf.EnergyParams(initial=args.energy)),
                           food=pf.FoodPatch(pf.FoodParams(center=tuple(center), radius=args.radius)))
    rows = []
    arrival = None
    for i in range(times.size):
        out = plat.step(float(times[i] - times[i - 1]) if i else .005, mouths[i])
        d = float(np.linalg.norm(mouths[i] - center))
        if arrival is None and d <= args.radius:
            arrival = dict(tick=i, time_s=float(times[i]), mouth_to_center=d)
        rows.append(dict(t=i, time_s=float(times[i]), decision=out['decision'], energy=out['energy'],
                         food_mass=out['food_mass'], consumed=out['consumed'], mouth_to_food=d,
                         travelled_forward=float((coms[i] - coms[0]) @ heading),
                         mouth_in_food=out['mouth_in_food']))
    arr = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    tag = args.tag or args.condition
    np.savez_compressed(HERE / f'PLATFORM_{tag}.npz', **arr)
    dec = {name: int(sum(1 for r in rows if r['decision'] == name)) for name in ('feed', 'forage', 'rest')}
    summary = dict(status='PLATFORM_WITH_LABELLED_ANN_MOVEMENT_RUN',
                   condition=args.condition, tag=tag, radius=args.radius, food_distance_native=dist,
                   ticks=int(times.size), duration_s=float(times[-1]),
                   movement_label='replacement_ANN_not_connectome',
                   movement_source='frozen u32 residual policy recorded trajectory',
                   movement_known_deficiency='material foot slip criterion FAILS; support gates pass',
                   selector_label='engineering_not_neural', feeding_label='declared_virtual_gate',
                   arrived=arrival is not None, arrival=arrival,
                   decisions=dec, energy_start=float(rows[0]['energy']), energy_end=float(rows[-1]['energy']),
                   food_start=1.0, food_end=float(rows[-1]['food_mass']),
                   max_forward_travel=float(rows[-1]['travelled_forward']),
                   min_mouth_to_food=float(min(r['mouth_to_food'] for r in rows)),
                   native_steps_added=0, neural_steps_added=0, training_updates=0,
                   provenance=dict(kinematics='frozen model, one mj_forward per recorded qpos; no stepping',
                                   channel='feeding has no feedback to the movement controller (it has no food sensing)'),
                   limits='ANN movement fails the frozen slip criterion, so this is not accepted walking and not connectome control. Feeding/food/selector are declared engineering components. No injury, no new physics claim.',
                   source_hashes={str(p): sha(p) for p in (ACT, PF, HERE / 'ANN_BODY_u32.npz', HERE / 'PREREG.json')})
    (HERE / f'RESULT_{tag}.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('condition', 'ticks', 'duration_s', 'arrived', 'arrival',
                                              'decisions', 'energy_start', 'energy_end', 'food_end',
                                              'max_forward_travel', 'min_mouth_to_food')}))


if __name__ == '__main__':
    main()
