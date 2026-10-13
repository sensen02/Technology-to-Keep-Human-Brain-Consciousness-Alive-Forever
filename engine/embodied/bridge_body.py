"""Independent engineered intent -> shipped FlyGym hybrid turning controller.

Not the canonical muscle backend and not a connectome motor circuit. Units: mm,
s, radians. Positive yaw is world +z (left when upright). No root constraints,
teleports, velocity writes or artificial stabilization occur during stepping.
As documented by engine/embodied/force_channel.py, contact-force magnitudes are
NON_PHYSICAL_LOAD signals here, not biologically calibrated load measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class BridgeBodyConfig:
    timestep_s: float = 0.0001
    seed: int = 0
    spawn_height_mm: float = 0.5
    frequency_hz: float = 12.0
    turn_asymmetry: float = 0.6
    render: bool = False


class BridgeBody:
    """Use ``step(forward=1, yaw=0, duration_s=.01)`` or named commands.

    Intent ranges: forward [0,1], yaw [-1,1]. Yaw means steering bias,
    NOT a requested angular velocity. No reverse or pivot-in-place guarantee.
    ``stand`` and ``stop`` hold the recorded neutral leg pose, with adhesion;
    this is an engineered position-actuated stop, not muscle/neural balance.
    ``observe()`` returns simulator ground truth, not reconstructed sensory data.
    """
    def __init__(self, config: BridgeBodyConfig | None = None):
        self.config = config or BridgeBodyConfig()
        c = self.config
        if not np.isfinite(c.timestep_s) or not 0 < c.timestep_s <= .001:
            raise ValueError('timestep_s must be finite and in (0, .001]')
        if not 0 <= c.turn_asymmetry < 1:
            raise ValueError('turn_asymmetry must be in [0,1)')
        os.environ.setdefault('MUJOCO_GL', 'egl')
        from flygym.compose import ActuatorType, FlatGroundWorld
        from flygym.simulation import Simulation
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain.common import make_locomotion_fly
        from flygym_demo.complex_terrain.cpg_controller import make_tripod_cpg_network
        from flygym_demo.complex_terrain.turning_controller import HybridTurningController
        from flygym_demo.complex_terrain.hybrid_controller import HybridControllerObservation
        import mujoco
        self._mujoco = mujoco
        self._observation_type = HybridControllerObservation
        self.fly = make_locomotion_fly('bridge')
        self.world = FlatGroundWorld(name='bridge_flat', half_size=1000.)
        self.world.add_fly(self.fly, spawn_position=[0., 0., c.spawn_height_mm],
                           spawn_rotation=Rotation3D('quat', [1., 0., 0., 0.]))
        if c.render:
            self.fly.add_tracking_camera(name='bridge_camera')
            self.world.mjcf_root.worldbody.add_light(pos=[0, 0, 20], dir=[0, 0, -1])
        self.sim = Simulation(self.world, timestep=c.timestep_s)
        self.model, self.data = self.sim.mj_model, self.sim.mj_data
        self.dof_order = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        net = make_tripod_cpg_network(timestep=c.timestep_s,
                                      intrinsic_frequency=c.frequency_hz, seed=c.seed)
        self.controller = HybridTurningController(timestep=c.timestep_s, cpg_network=net,
                                                  output_dof_order=self.dof_order)
        self.controller.reset(seed=c.seed, init_magnitudes=np.zeros(6))
        self._stand_angles = self.controller.preprogrammed_steps.default_pose_by_dof_order(self.dof_order)
        self._thorax_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bridge/c_thorax')
        if self._thorax_id < 0:
            raise RuntimeError('compiled thorax not found')
        self._renderer = None
        self.last_descending_signal = np.zeros(2)
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self) -> float:
        return self.config.timestep_s

    def snapshot(self) -> dict:
        return self.observe()

    def step(self, forward: float = 0., yaw: float = 0., *, duration_s: float | None = None,
             command: str | None = None, turn: float | None = None) -> dict:
        from flygym_demo.complex_terrain.common import LocomotionAction, apply_locomotion_action
        if turn is not None:
            if yaw != 0.:
                raise ValueError('use yaw or turn, not both')
            yaw = turn
        if command is not None:
            forward, yaw = {'stand': (0., 0.), 'stop': (0., 0.), 'forward': (1., 0.),
                            'left': (1., 1.), 'right': (1., -1.)}[command]
        if not np.isfinite([forward, yaw]).all() or not 0 <= forward <= 1 or not -1 <= yaw <= 1:
            raise ValueError('forward must be in [0,1], yaw in [-1,1], finite')
        duration = self.config.timestep_s if duration_s is None else duration_s
        n = round(duration / self.config.timestep_s)
        if n < 1 or not np.isclose(n * self.config.timestep_s, duration, atol=1e-10, rtol=0):
            raise ValueError('duration_s must be a positive integer number of timesteps')
        # Left side lower amplitude -> left turn; empirically checked by diagnostic.
        a = self.config.turn_asymmetry * yaw
        self.last_descending_signal = forward * np.array([1 - max(a, 0), 1 + min(a, 0)])
        for _ in range(n):
            if forward == 0:
                action = LocomotionAction(self._stand_angles, np.ones(6, dtype=bool))
                # CPG state is paused in stop; restart ramps amplitude from zero.
                self.controller.cpg_network.curr_magnitudes[:] = 0
                self.controller.retraction_correction[:] = 0
                self.controller.stumbling_correction[:] = 0
            else:
                sensory = self._observation_type.from_sim(self.sim, 'bridge')
                action = self.controller.step(self.last_descending_signal, sensory)
            apply_locomotion_action(self.sim, 'bridge', action)
            self.sim.step()
        return self.observe()

    def observe(self) -> dict:
        # mj_step integrates qpos after computing kinematics; refresh derived state
        # so position, orientation and contact sensors refer to this timestamp.
        self._mujoco.mj_forward(self.model, self.data)
        r = self.data.xmat[self._thorax_id].reshape(3, 3)
        contact = self.sim.get_ground_contact_info('bridge')
        return {'time_s': float(self.data.time),
                'position_model': self.data.xpos[self._thorax_id].copy(),
                'yaw_rad': float(np.arctan2(r[1, 0], r[0, 0])),
                'tilt_rad': float(np.arccos(np.clip(r[2, 2], -1, 1))),
                'heading_world': r[:, 0].copy(),
                'joint_angles_rad': self.sim.get_joint_angles('bridge').copy(),
                'joint_velocities_rad_s': self.sim.get_joint_velocities('bridge').copy(),
                'contact_found': np.asarray(contact[0]).copy(),
                'contact_forces_contact_frame': np.asarray(contact[1]).copy(),
                'contact_positions_mm': np.asarray(contact[3]).copy(),
                'descending_signal_left_right': self.last_descending_signal.copy()}

    def render(self) -> np.ndarray:
        if not self.config.render:
            raise RuntimeError('construct with BridgeBodyConfig(render=True)')
        if self._renderer is None:
            self._renderer = self._mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera='bridge_camera')
        return self._renderer.render().copy()

    def provenance(self) -> dict:
        from flygym_demo.complex_terrain import common, hybrid_controller, turning_controller, preprogrammed
        from importlib.resources import files
        paths = [Path(inspect.getfile(m)) for m in (common, hybrid_controller, turning_controller, preprogrammed)]
        paths += [Path(files('flygym_demo.complex_terrain') / 'assets/single_steps_untethered.pkl'), Path(__file__)]
        return {'implementation': 'independent engineered bridge; shipped HybridTurningController; position actuators plus adhesion',
                'not_claimed': ['connectome-generated motor control', 'muscle-canonical validation', 'biologically calibrated contact forces'],
                'config': asdict(self.config),
                'versions': {p: importlib.metadata.version(p) for p in ('flygym', 'mujoco', 'numpy')},
                'sources': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
                'gravity_mm_s2': self.model.opt.gravity.tolist(),
                'actuator_force_ranges': self.model.actuator_forcerange.tolist(),
                'free_root': True, 'warmup_s': 0., 'stop_strategy': 'recorded neutral pose, six adhesion actuators enabled',
                'units': {'position': 'mm', 'time': 's', 'angle': 'rad', 'force': 'model force units; uncalibrated'}}

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
