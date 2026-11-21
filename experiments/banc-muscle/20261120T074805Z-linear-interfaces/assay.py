"""Run a causal front-foot load -> BANC -> muscle bench on the canonical body."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import structural_rank

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / '20261120T061904Z-independent'
sys.path.insert(0, str(PARENT))
from body_observations import BASE, LEGS, MODEL, BodyObservations, load_pose, summary
from repair_graph import sha
from verify_local import ConductanceNetwork, DT_MS, load

SOURCES = BASE.parent / '20261120T034137Z-sensory-graph-threshold/GRAPH.json'


def bounded_edges(base, fractions):
    """Fractions follow CSR data order. Permit either a shared or one value per edge."""
    change = np.asarray(fractions, dtype=float)
    if change.shape not in ((), base.data.shape) or not np.isfinite(change).all() or np.any(abs(change) > .08):
        raise ValueError('expected a finite fractional change within ±0.08 for each existing edge')
    result = base.copy()
    result.data *= 1 + change
    return result


class Interfaces:
    def __init__(self, model):
        interface = json.loads((PARENT / 'INTERFACE.json').read_text())
        assert sha(MODEL) == interface['body_sha256']
        assert sha(PARENT / 'NETWORK.npz') == interface['graph_sha256']
        roots, weights = load()
        self.roots = roots
        self.network = [weights[k] + weights['repaired_' + k] for k in ('exc', 'inh')]
        for matrix in self.network:
            matrix.sort_indices()
        self.identity = dict(graph_sha=interface['graph_sha256'], body_sha=interface['body_sha256'])
        self.actuators = np.array([model.actuator(i).name for i in range(model.nu)])
        index = {str(root): i for i, root in enumerate(roots)}
        with np.load(HERE / 'VNC_CANDIDATES.npz', allow_pickle=False) as candidates:
            self.pre = np.array([index[r] for r in candidates['pre']])
            self.post = np.array([index[r] for r in candidates['post']])
        assert all(not np.any(np.asarray(w[self.post, self.pre])) for w in self.network)
        self.vnc_gain = np.zeros(len(self.pre))
        muscles = interface['muscles']
        self.exact = np.zeros((len(muscles), len(roots)), dtype=bool)
        self.gaps = np.zeros_like(self.exact)
        for row in muscles:
            arm = row['actuator_index']
            assert model.actuator(arm).name == row['actuator']
            self.exact[arm, [index[r] for r in row['annotated_candidate_roots']]] = True
            if not row['annotated_candidate_roots']:
                self.gaps[arm, [index[r] for r in row['inactive_functional_hypothesis_roots']]] = True
        self.motor = np.flatnonzero((self.exact | self.gaps).any(axis=0))
        self.exact = self.exact[:, self.motor]
        self.gaps = self.gaps[:, self.motor]
        labels = json.loads(SOURCES.read_text())['source_memberships']
        cells = [s for s in interface['sensory_candidates'] if s['group'] == 'leg_campaniform'
                 and labels[s['root_id']]['gate'] == 'verified_ACh']
        self.sensors = np.array([index[s['root_id']] for s in cells])
        assert len(cells) == 12 and all(s['leg'] == 'front_leg' for s in cells)
        self.feet = np.array([LEGS.index('LF' if s['side'] == 'left' else 'RF') for s in cells])
        self.load_gain = np.full(len(cells), .5)  # nA per body weight, provisional
        feco = [s for s in interface['sensory_candidates'] if s['candidate_observation'] is not None]
        self.feco = np.array([index[s['root_id']] for s in feco])
        self.feco_feet = np.array([LEGS.index(('L' if s['side'] == 'left' else 'R') +
                                   {'front_leg':'F','middle_leg':'M','hind_leg':'H'}[s['leg']]) for s in feco])
        self.angle = np.array([s['candidate_observation'] == 'tibia_angle_rad' for s in feco])
        assert all(s['candidate_observation'] in ('tibia_angle_rad','tibia_velocity_rad_s') for s in feco)
        self.feco_gain = np.zeros(len(feco))

        mask = self.exact | self.gaps
        self.motor_gain = mask / np.maximum(1, mask.sum(axis=1, keepdims=True))
        self.fractions = [np.zeros_like(w.data) for w in self.network]

    def parameters(self):
        arms, columns = np.nonzero(self.exact | self.gaps)
        return dict(**self.identity, sensory_root=self.roots[self.sensors], load_gain=self.load_gain,
                    muscle=self.actuators[arms], motor_root=self.roots[self.motor[columns]],
                    motor_gain=self.motor_gain[arms, columns], exact_candidate=self.exact[arms, columns],
                    exc_fraction=self.fractions[0], inh_fraction=self.fractions[1],
                    vnc_pre=self.roots[self.pre], vnc_post=self.roots[self.post], vnc_gain_uS=self.vnc_gain,
                    feco_root=self.roots[self.feco], feco_uses_angle=self.angle, feco_gain=self.feco_gain)

    def restore(self, path):
        template = self.parameters()
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(template):
                raise ValueError('checkpoint fields differ')
            trainable = ('motor_gain', 'load_gain', 'exc_fraction', 'inh_fraction', 'vnc_gain_uS', 'feco_gain')
            for key in template.keys() - set(trainable):
                np.testing.assert_array_equal(saved[key], template[key], err_msg=key)
            values = {}
            for key in trainable:
                v = np.asarray(saved[key], dtype=float)
                if v.shape != template[key].shape or not np.isfinite(v).all():
                    raise ValueError(key)
                values[key] = v
        if np.any(values['load_gain'] < 0) or np.any(values['motor_gain'] < 0):
            raise ValueError('negative interface gain')
        changes = [values[k] for k in ('exc_fraction', 'inh_fraction')]
        if any(np.any(abs(change) > .08) for change in changes):
            raise ValueError('existing edge change exceeds ±8%')
        self.fractions = changes
        self.load_gain = values['load_gain']
        self.vnc_gain = values['vnc_gain_uS']
        self.feco_gain = values['feco_gain']
        self.motor_gain[self.exact | self.gaps] = values['motor_gain']

    def connectivity(self, fractions):
        matrices = []
        for sign, w, d in zip((1, -1), self.network, fractions, strict=True):
            base = bounded_edges(w, d)
            active = self.vnc_gain * sign > 0
            if np.any(active):
                base += csr_matrix((sign * self.vnc_gain[active], (self.post[active], self.pre[active])),
                                   shape=base.shape)
            matrices.append(base)
        return matrices


def trial(model, obs, interfaces, pose, change, mode):
    net = ConductanceNetwork(len(interfaces.roots), dt_ms=DT_MS)
    fractions = interfaces.fractions if change is None else [change, change]
    net.set_connectivity(*interfaces.connectivity(fractions))
    mask = interfaces.exact | interfaces.gaps if mode == 'gaps' else interfaces.exact
    weights = interfaces.motor_gain * mask
    data = load_pose(model, pose)
    initial_angles = data.qpos[obs.tibia_qpos].copy()
    substeps = round(DT_MS / (1000 * model.opt.timestep))
    assert np.isclose(substeps * model.opt.timestep, DT_MS / 1000)
    motor_trace = np.zeros(len(interfaces.motor))
    drive = np.zeros(net.n)
    rows, neural = [], []
    for tick in range(round(.2 / model.opt.timestep) + 1):
        measured = obs.measure(data)
        rows.append((data.time, measured['ground_load_w'], measured['torso'], data.act.copy()))
        if tick == round(.2 / model.opt.timestep):
            break
        if tick % substeps == 0:
            loads = measured['ground_load_w'][interfaces.feet]
            drive[interfaces.sensors] = np.clip(loads * interfaces.load_gain, 0, .12)
            feco_input = np.where(interfaces.angle,
                (measured['tibia_angle_rad'] - initial_angles)[interfaces.feco_feet],
                measured['tibia_velocity_rad_s'][interfaces.feco_feet])
            drive[interfaces.feco] = np.clip(feco_input * interfaces.feco_gain, 0, .12)
            if mode == 'sensory_cut':
                drive.fill(0.)
            spikes = net.step(drive)
            motor_trace *= np.exp(-DT_MS / 10.)
            motor_trace += spikes[interfaces.motor]
            command = np.clip(weights @ motor_trace, 0, 1)
            if mode == 'motor_cut':
                command.fill(0.)
            neural.append((net.t_ms, int(spikes[interfaces.sensors].sum() + spikes[interfaces.feco].sum()),
                           int(spikes[interfaces.motor].sum()), command.copy()))
        mj.mj_step(model, data)
        if (tick + 1) % substeps == 0:
            data.ctrl[:] = command  # interval-end spike may only drive the following interval
        mj.mj_forward(model, data)
    assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied) and not any(data.warning.number)
    time, loads, torso, act = map(np.asarray, zip(*rows))
    end, sensory, motor, commands = map(np.asarray, zip(*neural))
    arrays = dict(time=time, ground_load_w=loads, torso=torso, activation=act,
                  neural_ms=end, sensory_spikes=sensory, motor_spikes=motor, commands=commands)
    assert all(np.isfinite(v).all() for v in arrays.values())
    report = summary(arrays)
    for key, active in [('sensory', sensory > 0), ('motor', motor > 0), ('command', commands.max(axis=1) > 0)]:
        report['first_' + key + '_ms'] = float(end[active][0]) if np.any(active) else None
    report['peak_command'] = float(commands.max())
    report['active_muscles'] = int((commands.max(axis=0) > 0).sum())
    return arrays, report


def main(out, checkpoint):
    out.mkdir(exist_ok=True, parents=True)
    model = mj.MjModel.from_xml_path(str(MODEL))
    obs = BodyObservations(model)
    interfaces = Interfaces(model)
    if checkpoint is not None:
        interfaces.restore(checkpoint)
    np.savez_compressed(out / 'PARAMETERS.npz', **interfaces.parameters())
    with np.load(BASE / 'body/archived_pose.npz') as saved:
        poses = dict(archived_six_feet=saved['qpos'], keyframe=model.key_qpos[0])
    cases = [('sensory_cut', None), ('motor_cut', None), ('annotated', None), ('gaps', None),
             ('gaps', -.08), ('gaps', .08)]
    mask = interfaces.exact
    counts = Counter(mask.sum(axis=0))
    result = dict(exact_pairs=int(mask.sum()), shared_roots=counts[2], gap_pairs=int(interfaces.gaps.sum()),
                  existing_edges=sum(w.nnz for w in interfaces.network),
                  checkpoint_sha=sha(out / 'PARAMETERS.npz'), assays={})
    for k, m in [('exact', mask), ('with_gaps', mask | interfaces.gaps)]:
        result[k] = dict(equal_pool_rank=int(np.linalg.matrix_rank(m.astype(float))),
                         structural_rank=int(structural_rank(csr_matrix(m))))
    for pose_name, pose in poses.items():
        for mode, change in cases:
            a, r = trial(model, obs, interfaces, pose, change, mode)
            key = f'{pose_name}_{mode}_' + ('checkpoint' if change is None else f'{round(change * 100):+d}')
            if mode in ('motor_cut', 'sensory_cut'):
                with np.load(PARENT / f'BODY_{pose_name}.npz') as old:
                    for field in ('time','ground_load_w','torso','activation'):
                        np.testing.assert_array_equal(a[field], old[field])
            result['assays'][key] = r
            np.savez_compressed(out / (key + '.npz'), **a)
            print(key, r['first_command_ms'], r['first_nonfoot_s'], r['mean_tail_foot_weight'], flush=True)
    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--parameters', type=Path)
    args = parser.parse_args()
    main(args.output, args.parameters)
