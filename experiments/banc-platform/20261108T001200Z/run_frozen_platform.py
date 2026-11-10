"""Frozen-platform reproduction run with movement quality as explicit state.

Replays the same recorded labelled-ANN trajectory as round 14, but every tick now carries
MovementStatus so the record itself states that the movement layer is NOT accepted.
Zero native physics steps: one mj_forward per recorded qpos for mouth kinematics only.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys
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
    sys.path.insert(0, str(ACT.parent))
    act = load('act', ACT)
    pf = load('pf', PF)
    ms = load('ms', HERE / 'movement_status.py')
    import mujoco as mj
    model = act.load_full_model(); data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0); mj.mj_forward(model, data)
    rostrum = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'Rostrum')
    z = np.load(HERE / 'ANN_BODY_u32.npz')
    times, qpos, coms = z['time'], z['qpos'], z['com']
    mouths = np.zeros((times.size, 3))
    for i in range(times.size):
        data.qpos[:] = qpos[i]; mj.mj_forward(model, data); mouths[i] = data.xpos[rostrum]
    heading = coms[-1] - coms[0]; heading[2] = 0.
    travelled = float(np.linalg.norm(heading))
    heading = heading / travelled if travelled > 0 else np.array([1., 0., 0.])
    center = mouths[0] + heading * .30
    status = ms.ANN_U32
    plat = pf.PlatformStep(food=pf.FoodPatch(pf.FoodParams(center=tuple(center), radius=.05)))
    rows, arrival = [], None
    for i in range(times.size):
        out = plat.step(float(times[i] - times[i - 1]) if i else .005, mouths[i])
        d = float(np.linalg.norm(mouths[i] - center))
        if arrival is None and d <= .05:
            arrival = dict(tick=i, time_s=float(times[i]))
        rows.append(dict(t=i, time_s=float(times[i]), decision=out['decision'], energy=out['energy'],
                         food_mass=out['food_mass'], mouth_to_food=d, movement=status.record()))
    dec = {n: int(sum(1 for r in rows if r['decision'] == n)) for n in ('feed', 'forage', 'rest')}
    summary = dict(status='FROZEN_PLATFORM_REPRODUCED', movement=status.record(),
                   ticks=int(times.size), duration_s=float(times[-1]), arrived=arrival is not None, arrival=arrival,
                   decisions=dec, energy_start=float(rows[0]['energy']), energy_end=float(rows[-1]['energy']),
                   food_end=float(rows[-1]['food_mass']), min_mouth_to_food=float(min(r['mouth_to_food'] for r in rows)),
                   native_steps_added=0, neural_steps_added=0,
                   labels=dict(movement='replacement_ANN_not_connectome_not_accepted',
                               feeding='declared_virtual_gate_no_food_collision',
                               selector='engineering_threshold_rule_not_neural',
                               energy='declared_engineering_state_uncalibrated'),
                   source_hashes={str(p): sha(p) for p in (ACT, PF, HERE / 'ANN_BODY_u32.npz', HERE / 'movement_status.py')})
    (HERE / 'RESULT_REPRODUCTION.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'arrived', 'arrival', 'decisions', 'energy_start', 'energy_end', 'food_end')}))
    print('movement accepted:', summary['movement']['accepted'], '| deficiencies:', len(summary['movement']['deficiencies']))


if __name__ == '__main__':
    main()
