"""S2/S5 isolated physics wrapper. Native units; absolute SI NOT_ESTABLISHED.

Only snapshot Python dependencies are imported. PAD_XML and POSE_FILE retain their
original frozen identities. No trajectory, trainer or controller is run on import.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco as mj
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
SOURCES = HERE / 'sources'

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

rc = _load('_walking_snapshot_repair_core', SOURCES / 'repair_core.py')
act_mod = _load('_walking_snapshot_incremental_actuator', SOURCES / 'incremental_actuator.py')
sensory = _load('_walking_snapshot_proprioception', SOURCES / 'proprioception.py')
# Prevent even explicit uses of the snapshot defaults from reaching mutable code.
rc.ACT = SOURCES / 'incremental_actuator.py'
rc.SENSORY = SOURCES / 'proprioception.py'
PAD_XML, POSE_FILE = rc.PAD_XML, rc.POSE_FILE
PDT, MAX_RATE, WEIGHT = rc.PDT, rc.MAX_RATE, rc.WEIGHT
W = WEIGHT
LEG_ORDER = LEGS = rc.LEG_ORDER
NL, NJ, OBS, OBS238 = rc.NL, rc.NJ, rc.OBS, 238
NATIVE_DT = 12.5e-6
GATES = dict(foot_support_mean_min=.90, nonfoot_force_max_W=.20 * W,
             com_height_min=1.20, slip_ratio_max=.5, forward_speed_min=.05)

def __getattr__(name):
    return getattr(rc, name)


class Core(rc.Core):
    """Legacy 1ms actuator API; observations never advance derivative state.

    recorder is optional NativeRecorder. obs()/obs495(prompt) return 495 features;
    obs238(previous_action, target_speed) returns 238 deployment features.
    """
    def __init__(self, *, act_mod=None, sensory=None, mu=rc.MU, adhesion=True, recorder=None):
        self.recorder = recorder
        super().__init__(act_mod=act_mod or globals()['act_mod'],
                         sensory=sensory or globals()['sensory'], mu=mu, adhesion=adhesion)
        if not np.isclose(self.plant.opt.timestep, NATIVE_DT, rtol=0, atol=1e-15):
            raise ValueError('Expected unchanged 12.5us native timestep')

    def _reset_measurements(self):
        self._padz_prev = self.measure()['padz'].copy()
        self._pad_rate = np.zeros(NL)
        self.requested_rate = np.zeros(NJ)
        self._adhesion_points = np.empty((0, 8))
        self._adhesion_wrench = np.zeros_like(self.d.xfrc_applied)
        self._adhesion_reference_body_com = self.d.xipos.copy()
        self._adhesion_refresh_time_s = float(self.d.time)
        self.initial_heading = self.d.xmat[self.th].reshape(3, 3)[:2, 0].copy()
        norm = np.linalg.norm(self.initial_heading)
        if norm <= 1e-12:
            raise ValueError('Initial heading has no horizontal projection')
        self.initial_heading /= norm

    def reset(self):
        super().reset()
        self._reset_measurements()

    def reset_from(self, qpos, qvel):
        super().reset_from(qpos, qvel)
        self._reset_measurements()

    def _proprio_clearance(self):
        m = self.measure()
        proprio = self.sensory.features(self.shim, self.d, self.actuator.target_rad,
                                        m['loads'][rc.SENS_IDX])
        clearance = np.r_[np.clip(m['padz'] / rc.PAD_H_DIV, -10., 10.),
                          np.clip(self._pad_rate / rc.PAD_V_DIV, -10., 10.)]
        return proprio, clearance

    def obs(self, prompt=None):
        proprio, clearance = self._proprio_clearance()
        prompt = np.zeros(NL) if prompt is None else np.asarray(prompt, float)
        if prompt.shape != (NL,) or not np.isfinite(prompt).all():
            raise ValueError('prompt must be finite shape (6,)')
        return np.concatenate([proprio, clearance, self.buf[self.lags].reshape(-1), prompt])

    obs495 = obs

    def obs238(self, prev_action, target_speed):
        previous = np.asarray(prev_action, float)
        if previous.shape != (NJ,) or not np.isfinite(previous).all() or not np.isfinite(target_speed):
            raise ValueError('Expected finite previous action (42,) and scalar target speed')
        proprio, clearance = self._proprio_clearance()
        return np.r_[proprio, clearance, previous, float(target_speed)]

    def com_velocity(self):
        jac = np.zeros((3, self.plant.nv))
        mj.mj_jacSubtreeCom(self.plant, self.d, jac, self.th)
        return jac @ self.d.qvel

    def critic_extras(self):
        """16 privileged features: loads/W(6), nonfoot/W, COMz, slip(6), tilt(2).

        Remaining finite-horizon time is supplied by the trainer, never actor.
        """
        m = self.measure()
        rot = self.d.xmat[self.th].reshape(3, 3)
        tilt = [np.arctan2(rot[2, 1], rot[2, 2]), np.arcsin(np.clip(-rot[2, 0], -1., 1.))]
        return np.r_[m['loads'] / W, m['nonfoot'] / W, m['com'], m['speed'], tilt]

    def apply_adhesion(self, loads):
        super().apply_adhesion(loads)
        # Equivalent to core_v2: xfrc_applied moments are about body COM.
        force = self.d.xfrc_applied[:, :3].copy()
        self.d.xfrc_applied[:, 3:] += np.cross(self.d.xpos - self.d.xipos, force)
        self._adhesion_wrench = self.d.xfrc_applied.copy()
        self._adhesion_reference_body_com = self.d.xipos.copy()
        self._adhesion_refresh_time_s = float(self.d.time)
        points = []
        counts = np.maximum(self.contact_count(), 1)
        for ci in range(self.d.ncon):
            contact = self.d.contact[ci]
            ga, gb = int(contact.geom1), int(contact.geom2)
            if not (self.planes[ga] or self.planes[gb]):
                continue
            geom = gb if self.planes[ga] else ga
            leg, body = int(self.geom_leg[geom]), int(self.plant.geom_bodyid[geom])
            if leg >= 0 and self._adhesion_wrench[body, 2] < 0.:
                share = rc.ADH_FRAC * W / counts[leg]
                points.append(np.r_[body, leg, contact.pos.copy(), 0., 0., -share])
        self._adhesion_points = np.asarray(points).reshape((-1, 8))

    def tick_rate(self, rate):
        rate = np.asarray(rate, float)
        if rate.shape != (NJ,) or not np.isfinite(rate).all():
            raise ValueError('rate must be finite shape (42,)')
        rate = np.clip(rate, -MAX_RATE, MAX_RATE)
        self.requested_rate = rate.copy()
        self.actuator.update(rate, PDT)
        for k in range(80):
            if k % 40 == 0:
                self.apply_adhesion(self.measure()['loads'])
            self.d.ctrl[:] = self.actuator.torque(self.d)
            before = self.recorder.before(self) if self.recorder is not None else None
            mj.mj_step(self.plant, self.d)
            # mj_step's contact solve belongs to the beginning-of-step geometry.
            solved = self.recorder.solved(self, before) if self.recorder is not None else None
            mj.mj_forward(self.plant, self.d)
            if self.recorder is not None:
                self.recorder.after(self, before, solved)
        if self.d.warning.number.any():
            raise RuntimeError('MuJoCo warning')
        self.buf = np.vstack([self.buf[1:], self.actuator.target_rad[None]])
        self.t += 1
        m = self.measure()
        self._pad_rate = (m['padz'] - self._padz_prev) / PDT
        self._padz_prev = m['padz'].copy()
        return m

    def tick_policy(self, action, decision_ms=5):
        """Hold one action over integer milliseconds; integrate target each 1ms."""
        if int(decision_ms) != decision_ms or decision_ms < 1:
            raise ValueError('decision_ms must be a positive integer')
        for _ in range(int(decision_ms)):
            m = self.tick(action)
        return m


class NativeRecorder:
    """Bounded-memory native recorder; call close() or use as context manager.

    Each chunk contains fixed arrays with leading native-step dimension and
    ragged contact arrays indexed by contact_offsets (length steps+1). Forces
    come from the integration solve before mj_forward, state arrays say before/
    after explicitly. All values copied. Peak summary covers all recorded steps.
    """
    def __init__(self, directory, chunk_size=2048):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.glob('native_*.npz')):
            raise FileExistsError('Refusing to overwrite native recording')
        if int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError('chunk_size must be positive integer')
        self.chunk_size = int(chunk_size)
        self.rows = []
        self.chunks = []
        self.count = 0
        self.nonfoot_peak = 0.
        self.ground_peak = 0.
        self.com_min = np.inf
        self.leg_load_peak = np.zeros(NL)
        self.leg_speed_peak = np.zeros(NL)
        self.closed = False

    def before(self, c):
        if self.closed:
            raise RuntimeError('Recorder is closed')
        d = c.d
        return dict(time_before_s=float(d.time), qpos_before=d.qpos.copy(),
                    qvel_before=d.qvel.copy(), com_before=d.subtree_com[c.th].copy(),
                    com_velocity_before=c.com_velocity(),
                    external_body_wrench_native=d.xfrc_applied.copy(),
                    external_generalized_force_native=d.qfrc_applied.copy(),
                    adhesion_body_wrench_native=c._adhesion_wrench.copy(),
                    adhesion_applications=c._adhesion_points.copy(),
                    adhesion_reference_body_com=c._adhesion_reference_body_com.copy(),
                    adhesion_refresh_time_s=c._adhesion_refresh_time_s,
                    body_com_before=d.xipos.copy(),
                    action=c.requested_rate.copy() / MAX_RATE,
                    requested_rate_rad_s=c.requested_rate.copy(),
                    target_rad=c.actuator.target_rad.copy(),
                    commanded_torque_native=d.ctrl.copy(),
                    adhesion_enabled=np.asarray(c.adhesion_on))

    def solved(self, c, before):
        d, model = c.d, c.plant
        loads, weighted, counts = np.zeros(NL), np.zeros(NL), np.zeros(NL, int)
        nonfoot = 0.
        contacts = []
        ground = np.zeros(3)
        for ci in range(d.ncon):
            contact = d.contact[ci]
            ga, gb = int(contact.geom1), int(contact.geom2)
            local_force = np.zeros(6)
            mj.mj_contactForce(model, d, ci, local_force)
            frame = contact.frame.reshape(3, 3)
            force = frame.T @ local_force[:3]
            leg, geom, velocity = -1, -1, np.zeros(3)
            is_ground = bool(c.planes[ga] or c.planes[gb])
            if is_ground:
                geom = gb if c.planes[ga] else ga
                force = force * (1 if c.planes[ga] else -1)
                ground += force
                leg = int(c.geom_leg[geom])
                jac = np.zeros((3, model.nv))
                mj.mj_jac(model, d, jac, None, contact.pos, int(model.geom_bodyid[geom]))
                velocity = jac @ before['qvel_before']
                normal = max(0., float(force[2]))
                if leg < 0:
                    nonfoot += normal
                else:
                    loads[leg] += normal
                    weighted[leg] += normal * np.linalg.norm(velocity[:2])
                    counts[leg] += 1
            contacts.append((np.array([ga, gb, leg, int(is_ground)]), contact.pos.copy(),
                             frame.copy(), local_force.copy(), force.copy(), velocity.copy()))
        return dict(loads_native=loads, nonfoot_force_native=np.asarray(nonfoot),
                    contact_speed_native_per_s=weighted / np.maximum(loads, 1e-12),
                    ground_force_native=ground, contact_count=counts,
                    actual_torque_native=d.actuator_force.copy(), contacts=contacts)

    def after(self, c, before, solved):
        row = dict(before)
        row.update({key: value for key, value in solved.items() if key != 'contacts'})
        row.update(time_after_s=float(c.d.time), qpos_after=c.d.qpos.copy(),
                   qvel_after=c.d.qvel.copy(), com_after=c.d.subtree_com[c.th].copy(),
                   com_velocity_after=c.com_velocity())
        dt = row['time_after_s'] - row['time_before_s']
        mass = float(c.plant.body_subtreemass[c.th])
        acceleration = (row['com_velocity_after'] - row['com_velocity_before']) / dt
        external = row['external_body_wrench_native'][:, :3].sum(axis=0)
        # The free root's translational generalized force is world-frame force.
        # Joint torques and root moments do not add net whole-body linear force.
        external += row['external_generalized_force_native'][:3]
        residual = mass * acceleration - (row['ground_force_native'] + external + mass * c.plant.opt.gravity)
        row.update(com_acceleration_fd_native=acceleration,
                   external_total_force_native=external,
                   force_balance_residual_native=residual, contacts=solved['contacts'])
        self.rows.append(row)
        self.count += 1
        self.nonfoot_peak = max(self.nonfoot_peak, float(row['nonfoot_force_native']))
        self.ground_peak = max(self.ground_peak, float(row['ground_force_native'][2]))
        self.com_min = min(self.com_min, float(row['com_before'][2]), float(row['com_after'][2]))
        self.leg_load_peak = np.maximum(self.leg_load_peak, row['loads_native'])
        self.leg_speed_peak = np.maximum(self.leg_speed_peak, row['contact_speed_native_per_s'])
        if len(self.rows) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arrays = {key: np.asarray([r[key] for r in self.rows])
                  for key in self.rows[0] if key not in ('contacts', 'adhesion_applications')}
        arrays['adhesion_application_offsets'] = np.r_[0, np.cumsum(
            [len(r['adhesion_applications']) for r in self.rows])]
        arrays['adhesion_application_body_leg_point_force_native'] = np.concatenate(
            [r['adhesion_applications'] for r in self.rows], axis=0)
        contacts = [item for row in self.rows for item in row['contacts']]
        arrays['contact_offsets'] = np.r_[0, np.cumsum([len(r['contacts']) for r in self.rows])]
        shapes = [(4,), (3,), (3, 3), (6,), (3,), (3,)]
        for i, name in enumerate(('contact_geom_leg_ground', 'contact_position_world', 'contact_frame',
                                  'contact_wrench_frame_native', 'contact_force_world_native',
                                  'contact_velocity_world_native_per_s')):
            arrays[name] = np.asarray([item[i] for item in contacts]).reshape((-1,) + shapes[i])
        path = self.directory / f'native_{len(self.chunks):06d}.npz'
        np.savez_compressed(path, **arrays)
        self.chunks.append(dict(path=str(path), rows=len(self.rows),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        self.rows.clear()

    def summary(self):
        return dict(native_samples=self.count, native_dt_s=NATIVE_DT,
                    nonfoot_force_peak_native=self.nonfoot_peak if self.count else None,
                    ground_vertical_force_peak_native=self.ground_peak if self.count else None,
                    com_height_min_native=self.com_min if self.count else None,
                    per_leg_load_peak_native=self.leg_load_peak.tolist() if self.count else None,
                    per_leg_contact_speed_peak_native_per_s=self.leg_speed_peak.tolist() if self.count else None,
                    chunks=list(self.chunks), pending_rows=len(self.rows),
                    force_balance_note='COM velocity before/after integration; solve forces before refresh; '
                    'finite difference/stagger diagnostic, not original D6 certification; '
                    'includes applied body forces and free-root generalized translational forces',
                    force_units='native; absolute SI NOT_ESTABLISHED')

    def close(self):
        self.flush()
        self.closed = True
        return self.summary()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def summarise(rec, steps, *, initial_heading=None, native_summary=None):
    """Exact legacy engineering formulas plus explicit heading diagnostics.

    rec tuples: (time, COMxyz, loads6, nonfoot_native, speed6), normally 1ms ends.
    Legacy forward gate remains signed WORLD X; slip denominator remains XY norm.
    """
    if len(rec) < 2:
        raise ValueError('Summary requires at least two samples')
    times, com, loads, nonfoot, speed = [np.asarray([r[i] for r in rec]) for i in range(5)]
    if any(not np.isfinite(a).all() for a in (times, com, loads, nonfoot, speed)):
        raise ValueError('Nonfinite summary data')
    duration = float(times[-1] - times[0])
    if duration <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Summary timestamps must strictly increase')
    support = loads.sum(1) / W
    delta = (com[-1] - com[0])[:2]
    xy_speed = float(np.linalg.norm(delta) / duration)
    signed_world_x = float(delta[0] / duration)
    load_den = loads.sum(0)
    vt = (loads * speed).sum(0) / np.maximum(load_den, 1e-12)
    ratio = vt / max(xy_speed, 1e-9)
    out = dict(native_steps=int(steps), sample_count=len(rec), duration_s=duration,
               forward_speed_native_per_s=xy_speed, forward_signed_native_per_s=signed_world_x,
               lateral_speed_native_per_s=float(abs(delta[1]) / duration),
               foot_support_mean=float(support.mean()), foot_support_min=float(support.min()),
               nonfoot_force_max_W=float(nonfoot.max()), nonfoot_force_max_native=float(nonfoot.max()),
               com_height_min=float(com[:, 2].min()), free_fall_fraction=float(np.mean(support < .05)),
               per_leg_slip_ratio=dict(zip(LEGS, map(float, ratio))),
               slip_ratio_max=float(ratio.max()), slip_ratio_mean=float(ratio.mean()),
               com_start=float(com[0, 2]), com_end=float(com[-1, 2]),
               com_xy_travel=float(np.linalg.norm(delta)),
               slip_load_denominator_native=load_den.tolist(),
               slip_valid_per_leg=(load_den > 1e-12).tolist(),
               slip_has_progress=bool(xy_speed > 1e-9),
               legacy_gate_frame='world X; slip denominator XY displacement norm/time',
               legacy_nonfoot_W_name='raw native force, NOT normalized body weights',
               sample_dt_s=float(np.median(np.diff(times))))
    out['gates'] = dict(support=out['foot_support_mean'] >= .90,
                        nonfoot=out['nonfoot_force_max_native'] <= .20 * W,
                        com=out['com_height_min'] >= 1.20,
                        forward=signed_world_x >= .05, slip=out['slip_ratio_max'] <= .5)
    out['all_gates'] = bool(all(out['gates'].values()))
    if initial_heading is None:
        out['initial_heading_diagnostics'] = dict(status='UNKNOWN', reason='initial heading not supplied')
    else:
        heading = np.asarray(initial_heading, float)[:2]
        norm = np.linalg.norm(heading)
        if heading.shape != (2,) or not np.isfinite(heading).all() or norm <= 1e-12:
            raise ValueError('Invalid initial heading')
        heading = heading / norm
        out['initial_heading_diagnostics'] = dict(status='MEASURED',
            forward_signed_native_per_s=float(delta @ heading / duration),
            lateral_signed_native_per_s=float(delta @ np.array([-heading[1], heading[0]]) / duration),
            heading_world_xy=heading.tolist(), gate_applied=False)
    out['native_peak_summary'] = native_summary
    # Original conditions need declared autonomous/auxiliary windows, native low-pass
    # acceleration, loading-leg threshold and original source-body applicability.
    # A sample count alone does not establish any original acceptance condition.
    out['roadmap_D1_D8'] = {name: dict(status='UNKNOWN', reason=reason) for name, reason in (
        ('D1', 'autonomous native window not supplied'),
        ('D2', 'native low-pass acceleration and autonomous window not supplied'),
        ('D3', 'same-run auxiliary window absent'),
        ('D4', 'autonomous native window not supplied'),
        ('D4b', 'autonomous native window not supplied'),
        ('D5', 'same-run auxiliary window absent'),
        ('D6', 'original D6 excludes declared adhesion; force timing/filter contract unresolved'),
        ('D7', 'native window and loading-leg threshold absent'),
        ('D8', 'autonomous native window not supplied'))}
    out['original_acceptance'] = 'UNKNOWN; derived triad body and finite-PD actuator; not frozen-body certification'
    return out

summary = summarise
