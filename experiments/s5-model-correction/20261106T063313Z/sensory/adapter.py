"""Pure sensory adapters. Importing this module cannot load/step a physics/neural model.
All new transfer functions are engineering proxies; no decoder/body-state bypass.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import sys
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
sys.path.insert(0, str(ROOT))
from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelQuantities, LEGS

ARMS = ('baseline', 'A_positive_force', 'B_club_motion', 'C_local_amplitude')


def finite(value, shape, label):
    a = np.asarray(value)
    if a.shape != shape or not np.isrealobj(a) or not np.isfinite(a).all():
        raise ValueError(f'{label}: expected finite real shape {shape}')
    return np.array(a, dtype=np.float64, copy=True)


@dataclass
class Layout:
    """42 scalar leg joints in source model joint order, not actuator order."""
    names: tuple
    joint_ids: np.ndarray
    qadr: np.ndarray
    vadr: np.ndarray
    reference: np.ndarray
    geom_names: tuple
    geom_body_ids: np.ndarray
    body_names: tuple
    geom_types: np.ndarray
    foot_leg: dict
    plane_type: int
    model_source: dict

    def __post_init__(self):
        self.names = tuple(self.names)
        if len(self.names) != 42 or len(set(self.names)) != 42:
            raise ValueError('42 unique named scalar joints required')
        for field in ('joint_ids', 'qadr', 'vadr', 'geom_body_ids', 'geom_types'):
            a = np.asarray(getattr(self, field))
            if a.ndim != 1 or a.dtype.kind not in 'iu':
                raise ValueError(field + ' must contain integer IDs')
            setattr(self, field, a.copy())
        for field in ('joint_ids', 'qadr', 'vadr'):
            a = getattr(self, field)
            if len(a) != 42 or len(set(a.tolist())) != 42 or np.any(a < 0):
                raise ValueError('invalid unique joint IDs/addresses')
        self.reference = finite(self.reference, (42,), 'reference')
        self.foot_leg = {int(k): str(v) for k, v in self.foot_leg.items()}
        n = len(self.geom_names)
        if len(self.geom_body_ids) != n or len(self.geom_types) != n:
            raise ValueError('unaligned geom metadata')
        if any(k < 0 or k >= n or v not in LEGS for k, v in self.foot_leg.items()):
            raise ValueError('invalid foot identities')
        if np.any(self.geom_body_ids < 0) or np.any(self.geom_body_ids >= len(self.body_names)):
            raise ValueError('invalid geom body IDs')
        self.rows = {}; self.tibia = {}; self.blocks = {}
        for leg in LEGS:
            rows = [i for i, name in enumerate(self.names) if name.lower().startswith('joint_' + leg)]
            if len(rows) != 7:
                raise ValueError('seven joints per leg required')
            self.rows[leg] = np.array(rows, dtype=int)
            tib = [i for i in rows if self.names[i] == 'joint_' + leg.upper() + 'Tibia_pitch']
            if len(tib) != 1:
                raise ValueError('unambiguous tibia joint required')
            self.tibia[leg] = tib[0]
            for segment in ('Coxa', 'Trochanter'):
                expected = {'joint_' + leg.upper() + segment + '_' + axis for axis in ('yaw', 'pitch', 'roll')}
                ids = [i for i in rows if self.names[i] in expected]
                if len(ids) != 3:
                    raise ValueError('joint block names unresolved: ' + leg + '/' + segment)
                self.blocks[(leg, segment.lower())] = np.array(ids, dtype=int)
        self.model_source = copy.deepcopy(self.model_source)

    def to_dict(self):
        return {k: (getattr(self, k).tolist() if isinstance(getattr(self, k), np.ndarray) else getattr(self, k))
                for k in self.__dataclass_fields__}

    @classmethod
    def from_model(cls, model, reference_qpos, joints, geom_leg, plane_type, model_source):
        """Reads an already existing model only; no compile, forward or step."""
        ids = [j for j in range(model.njnt)
               if any(model.joint(j).name.lower().startswith('joint_' + leg) for leg in LEGS)]
        names = tuple(model.joint(j).name for j in ids)
        qa = np.array([int(model.jnt_qposadr[j]) for j in ids]); va = np.array([int(model.jnt_dofadr[j]) for j in ids])
        out = cls(names, np.array(ids), qa, va, np.asarray(reference_qpos)[qa],
                  tuple(model.geom(g).name for g in range(model.ngeom)), np.asarray(model.geom_bodyid),
                  tuple(model.body(b).name for b in range(model.nbody)), np.asarray(model.geom_type),
                  geom_leg, int(plane_type), model_source)
        for leg, j in joints.items():
            if list(out.qadr[out.rows[leg]]) != list(j['q']) or out.qadr[out.tibia[leg]] != j['tib_q'] or out.vadr[out.tibia[leg]] != j['tib_v']:
                raise ValueError('new named layout disagrees with legacy joint order')
        return out


@dataclass
class RawObservation:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    geom_pairs: np.ndarray
    local_force6: np.ndarray
    frames: np.ndarray
    positions: np.ndarray
    distance: np.ndarray
    efc_address: np.ndarray

    def validate(self, layout):
        if not np.isfinite(self.time):
            raise ValueError('invalid sample time')
        self.qpos = finite(self.qpos, (49,), 'qpos')
        self.qvel = finite(self.qvel, (48,), 'qvel')
        self.joint_position = finite(self.joint_position, (42,), 'joint positions')
        self.joint_velocity = finite(self.joint_velocity, (42,), 'joint velocities')
        if not np.array_equal(self.qpos[layout.qadr], self.joint_position) or not np.array_equal(self.qvel[layout.vadr], self.joint_velocity):
            raise ValueError('raw joint vectors do not match full state')
        pairs = np.asarray(self.geom_pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
            raise ValueError('integer contact pairs required')
        nc = len(pairs)
        if np.any(pairs < 0) or np.any(pairs >= len(layout.geom_names)):
            raise ValueError('invalid contact geom ID')
        self.geom_pairs = pairs.copy()
        self.local_force6 = finite(self.local_force6, (nc, 6), 'contact force')
        # Contact normals are unilateral. Do not silently turn a malformed force into touch.
        if np.any(self.local_force6[:, 0] < 0):
            raise ValueError('negative contact normal force')
        self.frames = finite(self.frames, (nc, 3, 3), 'contact frames')
        self.positions = finite(self.positions, (nc, 3), 'contact positions')
        self.distance = finite(self.distance, (nc,), 'contact distances')
        a = np.asarray(self.efc_address)
        if a.shape != (nc,) or a.dtype.kind not in 'iu' or np.any(a < -1):
            raise ValueError('invalid efc addresses')
        self.efc_address = a.copy()
        return self


def capture_raw(model, data, layout, contact_force):
    """contact_force(model,data,index,out6) is injected; this never calls forward."""
    n = int(data.ncon); force = np.zeros((n, 6)); pairs = np.empty((n, 2), np.int64)
    frames = np.empty((n, 3, 3)); pos = np.empty((n, 3)); dist = np.empty(n); efc = np.empty(n, np.int64)
    for i in range(n):
        c = data.contact[i]; pairs[i] = (int(c.geom1), int(c.geom2))
        contact_force(model, data, i, force[i])
        frames[i] = np.asarray(c.frame).reshape(3, 3); pos[i] = c.pos; dist[i] = c.dist; efc[i] = c.efc_address
    return RawObservation(float(data.time), data.qpos.copy(), data.qvel.copy(),
                          data.qpos[layout.qadr].copy(), data.qvel[layout.vadr].copy(),
                          pairs, force, frames, pos, dist, efc).validate(layout)


def joint_quantities(position, velocity, reference, rows, tibia, force, touch):
    """One arithmetic path shared by raw observer and vector/.quantities contexts."""
    position = np.asarray(position); velocity = np.asarray(velocity); reference = np.asarray(reference)
    if position.ndim != 1 or position.shape != velocity.shape or position.shape != reference.shape:
        raise ValueError('joint vector shapes must agree')
    if not all(np.isrealobj(a) and np.isfinite(a).all() for a in (position, velocity, reference)):
        raise ValueError('finite joint vectors required')
    # Subtract after indexing, matching legacy observe_quantities arithmetic.
    dev = position[rows] - reference[rows]
    return ChannelQuantities(float(position[tibia] - reference[tibia]), float(velocity[tibia]),
                             float(np.mean(dev)), float(force), float(touch))


def quantities_from_raw(raw, layout, positive_force=False, eps=0.0):
    raw.validate(layout)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError('finite nonnegative force threshold required')
    force = dict.fromkeys(LEGS, 0.0); exists = dict.fromkeys(LEGS, 0.0); positive = dict.fromkeys(LEGS, 0.0)
    # Preserve source loop and scalar accumulation order for exact baseline matching.
    for pair, f in zip(raw.geom_pairs, raw.local_force6):
        for g, other in ((int(pair[0]), int(pair[1])), (int(pair[1]), int(pair[0]))):
            if g in layout.foot_leg and layout.geom_types[other] == layout.plane_type:
                leg = layout.foot_leg[g]
                force[leg] += float(np.linalg.norm(f[:3])); exists[leg] = 1.0
                if f[0] > eps:
                    positive[leg] = 1.0
    quantities = {leg: joint_quantities(raw.joint_position, raw.joint_velocity, layout.reference,
                                       layout.rows[leg], layout.tibia[leg], force[leg],
                                       positive[leg] if positive_force else exists[leg]) for leg in LEGS}
    return quantities, {'geometric_contact_exists': exists, 'positive_force_contact': positive,
                        'legacy_contact_norm_sum': force}


class SensoryAdapter:
    def __init__(self, legacy_encoder, layout, identity, arm='baseline'):
        if arm not in ARMS:
            raise ValueError('unknown isolated arm')
        self.legacy = legacy_encoder; self.layout = layout; self.identity = copy.deepcopy(identity); self.arm = arm
        self.c_groups = []
        seen = set()
        for group in identity['consensus_blocks']:
            leg, block = group['leg'], group['block']; idx = np.asarray(group['local_indices'], dtype=int)
            if (leg, block) not in layout.blocks or idx.ndim != 1 or not len(idx):
                raise ValueError('unresolved anatomical block')
            if any(int(i) in seen or int(i) not in self.legacy.groups[leg]['hairplate_angle'] for i in idx):
                raise ValueError('C target identity mismatch/overlap')
            seen.update(map(int, idx)); self.c_groups.append((leg, block, idx.copy()))
        if len(seen) != 29:
            raise ValueError('C requires precisely the 29 consensus hairplates')
        # Fixed annotation export and actual runtime encoder partition must match exactly.
        for leg in LEGS:
            for ch, ids in identity['channel_groups'][leg].items():
                if set(ids) != set(map(int, self.legacy.groups[leg][ch])):
                    raise ValueError('runtime sensory partition differs from exported identities')

    def quantities(self, raw):
        return quantities_from_raw(raw, self.layout, self.arm == 'A_positive_force')

    def encode(self, raw):
        qs, contact = self.quantities(raw)
        current, record = self.legacy.encode(qs)
        cfg = self.legacy.cfg
        record['adapter'] = dict(arm=self.arm, contact=contact, gain_status='ENGINEERING_NOT_CALIBRATED',
                                 club_component='NOT_MODELLED', local_amplitude='NOT_MODELLED',
                                 vibration_frequency_tuning='NOT_MODELLED', direction_restoration=False)
        if self.arm == 'B_club_motion':
            record['unmodelled_channels'] = [r for r in record['unmodelled_channels'] if r['channel'] != 'club_vibration']
            for leg in LEGS:
                value = min(abs(qs[leg].tibia_joint_vel_rad_s) / cfg.reference_velocity_rad_s, 1.0)
                ids = self.legacy.groups[leg]['club_vibration']; w = self.legacy.per_channel_weight['club_vibration']
                current[ids] = cfg.current_nA_per_unit * float(value) * w
                record['per_leg'][leg]['club_vibration'].update(value=float(value), modelled=True,
                    modelled_component='bidirectional_tibia_motion_proxy', vibration_frequency_tuning='NOT_MODELLED')
            record['adapter']['club_component'] = 'bidirectional_tibia_motion_proxy'
        if self.arm == 'C_local_amplitude':
            rows = []
            for leg, block, ids in self.c_groups:
                dev = raw.joint_position[self.layout.blocks[(leg, block)]] - self.layout.reference[self.layout.blocks[(leg, block)]]
                value = min(float(np.max(np.abs(dev))) / cfg.reference_angle_rad, 1.0)
                current[ids] = cfg.current_nA_per_unit * value * self.legacy.per_channel_weight['hairplate_angle']
                rows.append(dict(leg=leg, block=block, local_indices=ids.tolist(), value=value,
                                 modelled_component='local_angular_deviation_amplitude_ENGINEERING'))
                record['per_leg'][leg]['hairplate_angle']['subgroup_override'] = True
            record['adapter']['local_amplitude'] = rows
            record['adapter']['legacy_hairplate_record_scope'] = 'baseline value applies only to non-overridden cells'
        if not np.isfinite(current).all():
            raise ValueError('nonfinite adapter current')
        return current, record


def save_stream(path, raws, currents, layout):
    """Float64 variable-contact stream, no pickle. Enough to re-encode every arm."""
    if len(raws) != len(currents) or not raws:
        raise ValueError('nonempty aligned samples required')
    for raw in raws:
        raw.validate(layout)
    starts = np.r_[0, np.cumsum([len(r.geom_pairs) for r in raws])]
    fields = ('geom_pairs', 'local_force6', 'frames', 'positions', 'distance', 'efc_address')
    arrays = {k: np.concatenate([getattr(r, k) for r in raws], axis=0) for k in fields}
    arrays.update({k: np.stack([getattr(r, k) for r in raws]) for k in ('qpos', 'qvel', 'joint_position', 'joint_velocity')})
    arrays.update(time=np.array([r.time for r in raws]), contact_offsets=starts,
                  actual_current=np.asarray(currents, dtype=np.float64),
                  layout_json=np.array(json.dumps(layout.to_dict(), ensure_ascii=False)))
    p = Path(path)
    with p.open('xb') as f:
        np.savez_compressed(f, **arrays)


def load_stream(path):
    with np.load(path, allow_pickle=False) as z:
        layout = Layout(**json.loads(str(z['layout_json'])))
        bounds = z['contact_offsets']; raws = []
        for i, t in enumerate(z['time']):
            s = slice(int(bounds[i]), int(bounds[i + 1]))
            raws.append(RawObservation(float(t), z['qpos'][i], z['qvel'][i], z['joint_position'][i],
                z['joint_velocity'][i], z['geom_pairs'][s], z['local_force6'][s], z['frames'][s],
                z['positions'][s], z['distance'][s], z['efc_address'][s]).validate(layout))
        return layout, raws, z['actual_current'].copy()
