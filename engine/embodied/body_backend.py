"""Body backend: a physical adult-fly body in a scene, wrapped behind one API.

This module is the ONLY place that talks to FlyGym/MuJoCo.  Everything else in
the project talks to ``BodyBackend``, so the physics engine can be pinned,
versioned and, if necessary, replaced without touching the neural code.

VERIFIED FACTS ABOUT THE BACKEND (measured, not assumed)
--------------------------------------------------------
* FlyGym 2.1.0 / MuJoCo 3.9.0, Apache-2.0, EPFL Ramdya lab.
* locomotion fly: 69 body segments, 66 joint DOF, 42 actuated DOF.
* model units are MILLIMETRES: ``mujoco_globals.yaml`` states
  ``gravity: [0, 0, -9810]  # in mm/s^2`` and ``extent: 1  # mm``.
* ``Simulation`` methods used here return plain arrays, NOT dicts:
  joint angles (66,), joint velocities (66,), body positions (69,3),
  body rotations (69,4) as (w,x,y,z) quaternions,
  ground contact info as a 6-tuple of (6,)- and (6,3)-shaped arrays,
  actuator forces (42,) for ``ActuatorType.POSITION``.
* ``get_site_positions`` returns (0,3) for this fly: ``make_locomotion_fly``
  does not add anatomical joint sites.  Recorded, not hidden.

WHAT THE CONTROLLER IS
----------------------
``use_cpg_controller=True`` drives the fly with FlyGym's OWN tripod CPG
(``flygym_demo.complex_terrain.cpg_controller``).  That is an ENGINEERING
BASELINE: a hand-built oscillator network, NOT the connectome, and behaviour
produced by it must never be described as connectome-generated.  The flag is
reported in every record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Any

import numpy as np

MM = "mm"
SECONDS = "s"
RAD = "rad"
MM_PER_S = "mm/s"

#: GL backends to try, in order.  This machine has no CUDA GPU and no libOSMesa;
#: the winner is measured at run time and recorded rather than assumed.
GL_BACKENDS = ("egl", "osmesa", None)

GEOMETRY_NOTE = (
    "model units are millimetres (mujoco_globals.yaml: gravity [0,0,-9810] mm/s^2, "
    "extent 1 mm); values are recorded in model units and never silently rescaled")


@dataclass
class BodyConfig:
    """Everything needed to rebuild the identical body and scene."""
    fly_name: str = "nmf"
    timestep_s: float = 1e-4
    spawn_position_mm: tuple = (0.0, 0.0, 0.5)
    #: SPAWN ORIENTATION as a quaternion (w, x, y, z), IDENTITY by default so every previous
    #: result is bit-identical.  This exists so an episode can START the fly in a chosen pose --
    #: specifically INVERTED, which is the only honest way to ask whether a controller can right
    #: itself: the perturbation has to be an initial condition, not a mid-run poke, or the
    #: answer depends on the poke.  A 180-degree rotation about x is (0, 1, 0, 0).
    spawn_quat_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)
    #: Visual half-extent of the ground plane, in MM (FlyGym's own default is
    #: 1000).  The plane is INFINITE for collision -- MuJoCo planes always are --
    #: so a value smaller than the walk distance does NOT break the physics, but
    #: the RENDERED ground ends and the fly appears to walk on nothing.  Measured:
    #: with 20 mm the ground vanished after ~1.4 s of a 10 s walk.  This must
    #: therefore exceed the intended travel distance.
    world_half_size_mm: float = 1000.0
    cpg_intrinsic_frequency_hz: float = 12.0
    joint_stiffness: float = 0.05
    joint_damping: float = 0.06
    actuator_gain: float = 45.0
    #: Actuator FORCE LIMIT, in the model's own force unit (kg*mm/s^2 = 1 mN; the model's
    #: length unit is millimetres and its gravity is -9810).  FlyGym's default is (-65, 65)
    #: mN per actuator.
    #:
    #: MEASURED AND THIS IS WHY THE FIELD EXISTS: with the default limit the walking fly's
    #: total ground reaction force is 16,600 times its body weight, and a SINGLE tarsal
    #: contact reaches 44,670 uN -- which is 0.69 of this 65 mN limit.  The legs are pressed
    #: into the substrate until the actuators SATURATE, so the contact force is set by the
    #: force limit and not by the load.  That also explains a second measurement: lowering
    #: ``actuator_gain`` by a factor of 1000 changed the total GRF by only 18 % (166,546 ->
    #: 137,428 uN), because a position actuator whose error is large enough pushes at its
    #: limit whatever its gain is.
    #:
    #: A biologically scaled limit is what makes the contact FORCE meaningful; and since the
    #: contact AREA is derived from the force, it is also what makes the area graded instead
    #: of pinned at the tarsus footprint.
    actuator_forcerange: tuple = (-65.0, 65.0)
    add_adhesion: bool = True
    add_tracking_camera: bool = True
    tracking_camera_name: str = "trackcam"
    #: Optional WORLD-FIXED camera looking straight down, so renders can show
    #: WHERE on the ground the fly is rather than only following it.  It must be
    #: added to the MJCF spec BEFORE the world is compiled, which is why it lives
    #: in the config instead of being bolted on afterwards: adding a camera to an
    #: already-compiled model raises "Camera ... not found in the model".
    add_world_camera: bool = False
    world_camera_name: str = "worldcam"
    world_camera_pos_mm: tuple = (60.0, 10.0, 190.0)
    world_camera_fovy: float = 50.0
    #: MuJoCo camera frame: first triple = X axis, second = Y axis; the camera
    #: looks along -Z where Z = X cross Y.  Exposed because the right orientation
    #: had to be established empirically, not assumed.
    world_camera_xyaxes: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    #: ADDITIONAL world-fixed cameras, as plain dicts of MJCF camera keyword
    #: arguments (name, pos, xyaxes, fovy).  They must also be added BEFORE compile
    #: for the same reason as the single world camera above.  ``pos`` is in MM like
    #: everything else in this model.  Used by the recording pipeline for its
    #: close-up, motorised-stage camera; a camera declared here can have its
    #: ``cam_pos`` written at run time to PAN it, which is why the pipeline needs it
    #: in the model rather than bolted on afterwards.
    extra_cameras: tuple = ()
    #: Extra MATERIALS added to the world root BEFORE compile(), as plain dicts of MJCF
    #: material keyword arguments (name, rgba, emission, reflectance, shininess,
    #: specular).  Needed because a geom cannot reference a material that does not exist
    #: at compile time (measured: "material 'marker_emissive' not found in geom 2"), and
    #: the arena's fiducial markers have to be EMISSIVE to be separable from grass that
    #: already saturates the image.
    extra_materials: tuple = ()
    #: ENABLE THE COMPOUND-EYE CAMERAS.  This fly ships with NO cameras at all
    #: (measured: ncam == 0 before any camera is added), which is the concrete
    #: reason the embodied loop had no visual pathway: FlyGym's
    #: Simulation.get_raw_vision() raises "Fly '...' does not have any eye
    #: cameras defined" unless add_vision() ran BEFORE the world was compiled.
    #: Off by default so every previously recorded result keeps its exact model;
    #: vision runs pass True.  CAVEAT carried from FlyGym's own docstring: its
    #: fisheye Retina is calibrated for FlyGym's OWN eye placement, so ommatidia
    #: readouts on the NeuroMechFly body are APPROXIMATE -- approximate optics,
    #: not measured Drosophila optics.
    add_vision: bool = False
    #: Field of view of each eye camera, degrees, AS MEASURED FROM THE COMPILED
    #: MODEL.  NOTE (measured, not assumed): NeuroMechFly's compose class exposes
    #: ``add_vision(draw_sensor_markers=False) -> None`` with NO fovy keyword --
    #: the angle comes from the shipped asset
    #: ``flygym/assets/model/neuromechfly/vision.yaml`` (``fovy_per_eye: 157``),
    #: so it is READ BACK here after compile rather than offered as a knob that
    #: would silently do nothing.  FlyGym's other fly class
    #: (``MusculoskeletalFly``) defaults to 145 deg, which is a DIFFERENT value --
    #: do not carry 145 over to this fly.
    eye_fovy_deg_measured: float = float("nan")
    #: OPTIONAL declarative scene layer (engine.embodied.scene).  None preserves
    #: the historical scene EXACTLY as measured: one infinite flat plane, ONE
    #: uniform friction row [1.0, 0.005, 0.0001] shared by every geom, no
    #: micro-relief and nlight == 0 (no lights in the scene at all).  A preset
    #: name is resolved lazily inside __init__, so this module never hard-depends
    #: on scene.py existing.
    scene_preset: "str | None" = None
    #: Extra STATIC geoms added to the world root BEFORE compile(), given as plain
    #: dicts of MJCF keyword arguments (name, type, size, pos, rgba, contype,
    #: conaffinity).  ``size`` and ``pos`` are in MILLIMETRES like everything else
    #: in this model.  Used to put a visual target in front of the fly; pass
    #: contype=0 and conaffinity=0 so the object CANNOT touch the fly, which keeps
    #: any measured effect purely visual rather than mechanical.
    extra_geoms: tuple = ()
    seed: int = 0
    #: OPTIONAL electrode payload (``electrode_payload.ElectrodePayloadConfig``):
    #: the mass a recording electrode + holder puts on the fly.  Injected into the
    #: MJCF spec BEFORE ``world.add_fly()``/``compile()``, attached as a real child
    #: body at a real offset so its weight produces a torque about the segment
    #: origin.  ``None`` keeps every previously recorded result on a model with
    #: ZERO added mass; the payload module is imported lazily, so this module never
    #: hard-depends on it.
    electrode_payload: "object | None" = None
    #: FLY-BORNE VISUAL MARKERS, as ``{"segments": (...), "radius_m": float,
    #: "material": str|None}``.  Injected into the FLY spec before ``add_fly()``/``compile()``,
    #: which is the ONLY point at which the injection survives -- see the long note at the
    #: injection site.  ``None`` keeps every previously recorded model unchanged.
    #:
    #: THE RADIUS IS AN IDENTITY BUDGET, NOT A COSMETIC CHOICE.  A marker can only be told
    #: apart from its neighbour if its DIAMETER IS SMALLER THAN THE GAP (``tools_identity_
    #: budget.py``, medians over a real episode): the closest non-own marker to a frontal coxa
    #: origin is 0.257 mm away, so the 0.60 mm-diameter marker used previously MERGED with it
    #: on 8 of the 12 leg bones, which is not recoverable at any sensor resolution.  0.26 mm
    #: diameter (0.13 mm radius) makes all 12 separable while staying ~7 px across at the
    #: project's ring and resolution, well above the ~5 px a centroid needs.
    fly_markers: "dict | None" = None
    #: The yardstick the payload's ``load_fraction`` is a fraction OF, in kg.  Must
    #: be the mass of the SAME asset measured on an unloaded model; ``None`` uses
    #: ``electrode_payload.BODY_MASS_KG_FALLBACK``.  Cross-check the compiled
    #: control model with ``electrode_payload.assert_yardstick`` -- a wrong yardstick
    #: changes the load by the same relative error and is otherwise invisible.
    electrode_payload_yardstick_kg: "float | None" = None

    def validate(self):
        if not math.isfinite(self.timestep_s) or self.timestep_s <= 0:
            raise ValueError("timestep_s must be positive and finite")
        if len(self.spawn_position_mm) != 3 or not all(
                math.isfinite(float(v)) for v in self.spawn_position_mm):
            raise ValueError("spawn_position_mm must be three finite numbers")
        if len(self.spawn_quat_wxyz) != 4 or not all(
                math.isfinite(float(v)) for v in self.spawn_quat_wxyz):
            raise ValueError("spawn_quat_wxyz must be four finite numbers")
        _qn = math.sqrt(sum(float(v) ** 2 for v in self.spawn_quat_wxyz))
        if abs(_qn - 1.0) > 1e-6:
            raise ValueError("spawn_quat_wxyz must be a UNIT quaternion (norm %r)" % _qn)
        if not math.isfinite(self.world_half_size_mm) or self.world_half_size_mm <= 0:
            raise ValueError("world_half_size_mm must be positive and finite")
        if not math.isfinite(self.cpg_intrinsic_frequency_hz) or \
                self.cpg_intrinsic_frequency_hz <= 0:
            raise ValueError("cpg_intrinsic_frequency_hz must be positive and finite")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if len(tuple(self.actuator_forcerange)) != 2 or not all(
                math.isfinite(float(v)) for v in self.actuator_forcerange):
            raise ValueError("actuator_forcerange must be two finite numbers")
        if float(self.actuator_forcerange[0]) >= float(self.actuator_forcerange[1]):
            raise ValueError("actuator_forcerange must be (min, max) with min < max")
        if self.add_tracking_camera and not str(self.tracking_camera_name).strip():
            raise ValueError("tracking_camera_name must be a non-empty string")
        if not math.isfinite(self.eye_fovy_deg_measured) and \
                not math.isnan(self.eye_fovy_deg_measured):
            raise ValueError("eye_fovy_deg_measured must be a number")
        if self.scene_preset is not None and not str(self.scene_preset).strip():
            raise ValueError("scene_preset must be None or a non-empty preset name")
        return self

    def units(self):
        return {"spawn_position_mm": MM, "spawn_quat_wxyz": ("unit quaternion (w,x,y,z); "
                "(0,1,0,0) is the fly INVERTED about x"),
                "world_half_size_mm": MM,
                "timestep_s": SECONDS, "cpg_intrinsic_frequency_hz": "Hz",
                "joint_stiffness": "N*mm/rad", "joint_damping": "N*mm*s/rad",
                "actuator_gain": "N*mm/rad", "gravity": "mm/s^2 (= -9810)",
                "actuator_forcerange": "mN (= kg*mm/s^2), per actuator"}


@dataclass
class BodyObservation:
    """One body state.  Units are in the field names; nothing is converted.

    CONTACT SEMANTICS (read from the backend docstring, not guessed):
    ``get_ground_contact_info`` returns a 6-tuple per leg:
        (contact_found (6,), forces (6,3) in the contact frame,
         torques (6,3) in the contact frame, positions (6,3) global,
         normals (6,3) global, tangents (6,3) global)
    ``contact_found`` is the RAW MuJoCo sensor "found" channel.  Measured here:
    its values can exceed 1 (a leg can register several contacts), so
    ``raw > 0`` is the in-contact test and the raw value is kept separate.  An
    earlier version of this file summed the raw channel and reported it as a
    "number of legs in contact", which produced impossible counts (13 of 6 legs)
    and per-leg fractions above 1 -- that is why the two fields are now distinct.
    """
    time_s: float
    joint_angles_rad: np.ndarray
    joint_velocities_rad_s: np.ndarray
    body_positions_mm: np.ndarray
    body_rotations_wxyz: np.ndarray
    thorax_position_mm: np.ndarray
    contact_found_raw: np.ndarray          # (6,) raw sensor channel, may be > 1
    contact_forces: np.ndarray             # (6,3) N, contact frame
    contact_torques: np.ndarray            # (6,3) N*mm, contact frame
    contact_positions_mm: np.ndarray       # (6,3) global
    actuator_forces: np.ndarray            # (42,) N*mm

    @property
    def contact_present(self) -> np.ndarray:
        """Boolean (6,): a leg is in contact iff its raw found channel is > 0."""
        return self.contact_found_raw > 0


class BodyBackend:
    """A pinned FlyGym body+scene behind a minimal, unit-explicit API.

    The class does NOT import flygym at module import time; it imports inside
    ``__init__`` so that the neural environment (which has no MuJoCo) can import
    this package's metadata without failing.
    """

    def __init__(self, config: BodyConfig | None = None, gl_backend: str | None = "egl"):
        self.cfg = (config or BodyConfig()).validate()
        if gl_backend is not None:
            os.environ["MUJOCO_GL"] = gl_backend
        else:
            os.environ.pop("MUJOCO_GL", None)
        self.gl_backend = gl_backend or "default"

        from flygym.compose import ActuatorType, FlatGroundWorld
        from flygym.simulation import Simulation
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain.common import make_locomotion_fly
        import mujoco  # local import: the neural venv has no MuJoCo

        self._ActuatorType = ActuatorType
        self.fly = make_locomotion_fly(
            self.cfg.fly_name,
            joint_stiffness=self.cfg.joint_stiffness,
            joint_damping=self.cfg.joint_damping,
            actuator_gain=self.cfg.actuator_gain,
            actuator_forcerange=tuple(float(v) for v in self.cfg.actuator_forcerange),
            add_adhesion=self.cfg.add_adhesion,
        )
        # Eye cameras are part of the FLY and are created by FlyGym's own
        # add_vision(); without them get_raw_vision()/get_ommatidia_readouts()
        # raise "does not have any eye cameras defined".
        # ORDER MATTERS AND IS NOT DOCUMENTED (measured here): add_vision() must be
        # called BEFORE world.add_fly().  Called afterwards it dies with the
        # misleading "AttributeError: 'NoneType' object has no attribute
        # 'add_body'" because add_vision() looks its parent bodies up via
        # self.mjcf_root.body(name), and once the fly has been re-parented into the
        # world, that lookup returns None.  Verified both ways: with
        # add_vision() first, the compiled model has ncam == 2 and the names
        # nmf/l_eye_cam_camera and nmf/r_eye_cam_camera.
        self.eye_cameras_added = {}
        if self.cfg.add_vision:
            # Signature is add_vision(draw_sensor_markers: bool = False) -> None and
            # the fovy is fixed by the shipped vision.yaml, so the cameras are read
            # back from eyecameraname_to_mjcfcamera, not from a return value.
            self.fly.add_vision()
            self.eye_cameras_added = dict(self.fly.eyecameraname_to_mjcfcamera)

        self.world = FlatGroundWorld(name="flat_world",
                                     half_size=self.cfg.world_half_size_mm)
        # THE ELECTRODE PAYLOAD GOES IN HERE: after the fly is composed (so its
        # bodies exist) and before add_fly()/compile() (so the extra masses are
        # compiled into the model).  It is injected into the FLY spec, not the world
        # spec, so it moves with the thorax/head instead of hanging in world space.
        # Measured and asserted, not assumed: the report records the mass before and
        # after, and the signature shows load_fraction == 0 changes nothing.
        self.payload = None
        self.payload_report = None
        if self.cfg.electrode_payload is not None:
            from electrode_payload import (attach_payload, measure_body_mass_kg,
                                           BODY_MASS_KG_FALLBACK)
            # THE YARDSTICK: the load is expressed as a fraction of the fly's own
            # body mass, so that mass must be known BEFORE any load is added.
            # Two routes were tried and both measured to be broken:
            #   * compiling the WORLD at this point returns mass_sum == 0.0 (an empty
            #     world compiles fine), which silently scaled every payload to zero;
            #   * compiling a throwaway world around a SECOND fly CORRUPTED the real
            #     fly's spec -- measured: "ValueError: repeated name 'c_thorax/...'
            #     in body", i.e. ``NeuroMechFly(name=...)`` handed back a SHARED
            #     spec, so the probe's bodies were already in this fly.
            # So the yardstick is the value MEASURED on this same asset by
            # ``tools_measure_body_mass.py`` (recorded in
            # ``outputs/electrode_payload/body_mass.json``), passed in by the caller
            # and cross-checked against the compiled control model by
            # ``assert_yardstick``.  A default keeps single-call users working.
            _mass_before = float(self.cfg.electrode_payload_yardstick_kg) \
                if getattr(self.cfg, "electrode_payload_yardstick_kg", None) \
                else float(BODY_MASS_KG_FALLBACK)
            _mass_source = ("DECLARED from a previous measurement of this exact asset "
                            "(electrode_payload.BODY_MASS_KG_FALLBACK); verify with "
                            "electrode_payload.assert_yardstick(model)")
            self.payload = attach_payload(self.fly, self.cfg.electrode_payload,
                                         _mass_before)
            _rep = self.payload.as_dict()
            _rep["report"]["body_mass_before_kg"] = float(_mass_before)
            _rep["report"]["body_mass_source"] = _mass_source
            self.payload_report = _rep
            # A payload that silently failed to attach is the failure mode that
            # matters here, so it is promoted to an exception rather than logged.
            if self.cfg.electrode_payload.load_fraction > 0.0 and \
                    self.payload.total_payload_kg <= 0.0:
                raise RuntimeError(
                    "electrode payload requested "
                    f"(load_fraction={self.cfg.electrode_payload.load_fraction}) "
                    "but zero mass was attached; refusing to run an experiment "
                    "that claims a load it does not have")
        # FLY-BORNE VISUAL MARKERS GO IN HERE, for exactly the same reason as the payload:
        # after the fly is composed and BEFORE add_fly()/compile().
        #
        # MEASURED, AND THIS WAS A SILENT FAILURE THAT COST A WHOLE RECORDING ROUND.  Calling
        # ``electrode_payload.attach_marker_bodies(be.fly, ...)`` AFTER ``BodyBackend()`` has
        # returned reports 21 bodies added and has NO effect on the compiled model: by then
        # ``world.add_fly()`` has re-parented the fly's bodies and ``world.compile()`` has
        # already run.  The compiled model was measured to hold 69 bodies, 83 geoms and ZERO
        # geoms or bodies named ``mk_*``, so that six-camera episode contains no fly markers at
        # all and every "measured" fly-marker position in it is really the segment origin read
        # out of the simulator.  Passing the segments through this field instead puts them in
        # before the compile, and the report is verified so the failure cannot recur silently.
        self.marker_report = None
        _mk = getattr(self.cfg, "fly_markers", None)
        if _mk:
            from electrode_payload import attach_marker_geoms, compute_outward_offsets
            _segments = tuple(_mk.get("segments") or ())
            # RADIUS IN MODEL UNITS (mm).  See attach_marker_geoms: passing metres here
            # produced sub-micron spheres and a long hunt for "invisible markers".
            _radius_mm = float(_mk.get("radius_mm", _mk.get("radius_m", 0.13)))
            _mat = _mk.get("material", None)
            if _mat:
                # THE MATERIAL MUST ALSO EXIST IN THE FLY'S OWN SPEC, NOT ONLY THE WORLD'S.
                # MEASURED: FlyGym's ``_rebuild_neutral_keyframe`` compiles the fly SUBTREE
                # STANDALONE inside ``add_fly``, and a material declared only on the world root
                # fails there with "material 'nmf/marker_emissive' not found in geom 5": the
                # reference is namespaced under the fly, so the definition has to live there
                # too.  This is why the previous attempt at fly markers produced no geoms at
                # all rather than an error -- the failure happened after the world had already
                # been built.
                _fly_root = getattr(self.fly, "_mjcf_root", None) \
                    or getattr(self.fly, "mjcf_root", None)
                if _fly_root is None:
                    raise RuntimeError("cannot declare the marker material: the fly spec root "
                                       "was not found")
                for _mspec in (getattr(self.cfg, "extra_materials", ()) or ()):
                    if str(_mspec.get("name")) == str(_mat):
                        _kw = dict(_mspec)
                        _kw.pop("name")
                        _fly_root.add_material(name=str(_mat), **_kw)
                        break
                else:
                    raise RuntimeError(
                        f"fly_markers asks for material {_mat!r} which is not among "
                        "BodyConfig.extra_materials; the fly subtree compiles standalone, so "
                        "the material must be declared there as well")
            # OFFSET: an explicit vector, or "outward" meaning computed per segment from the
            # spec's neutral pose (a marker at a segment ORIGIN is inside the body's mesh and
            # renders as nothing -- MEASURED).
            _off_spec = _mk.get("offset_m", (0.0, 0.0, 0.0))
            if _off_spec == "outward":
                _off = compute_outward_offsets(
                    self.fly, _segments,
                    distance_mm=float(_mk.get("offset_dorsal_mm", 0.18)),
                    torso_distance_mm=float(_mk.get("offset_torso_mm", 0.55)),
                    coxa_distance_mm=float(_mk.get("offset_coxa_mm", 0.50)))
            else:
                _off = tuple(float(v) for v in (_off_spec or (0.0, 0.0, 0.0)))
            _rep = attach_marker_geoms(self.fly, _segments, radius_mm=_radius_mm,
                                       material=_mat, group=int(_mk.get("group", 0)),
                                       offset_m=_off,
                                       rgba=_mk.get("rgba", (1.0, 1.0, 1.0, 1.0)))
            _rep["requested"] = {
                "segments": list(_segments), "radius_mm": _radius_mm,
                "material": _mat,
                "injection": "BodyConfig.fly_markers, applied before world.add_fly()"}
            if len(_rep["geoms_added"]) != len(_segments):
                raise RuntimeError(
                    f"fly markers requested for {len(_segments)} segments but only "
                    f"{len(_rep['geoms_added'])} geoms attached; refusing to record an episode "
                    "whose markers are partly missing")
            self.marker_report = _rep
        self.world.add_fly(
            self.fly,
            spawn_position=[float(v) for v in self.cfg.spawn_position_mm],
            spawn_rotation=Rotation3D("quat",
                                      [float(v) for v in self.cfg.spawn_quat_wxyz]))
        # A tracking camera must be added BEFORE compiling: it is attached
        # inside the fly's root body, and this fly ships with no cameras at all
        # (verified: model.ncam == 0).  Without it Renderer(...) raises
        # "Camera track not found in the model".
        if self.cfg.add_tracking_camera:
            self.fly.add_tracking_camera(name=self.cfg.tracking_camera_name)
        if self.cfg.add_world_camera:
            # MjsBody exposes add_camera(); the generic add() does NOT exist on it.
            self.world.mjcf_root.worldbody.add_camera(
                name=self.cfg.world_camera_name,
                pos=[float(v) for v in self.cfg.world_camera_pos_mm],
                xyaxes=[float(v) for v in self.cfg.world_camera_xyaxes],
                fovy=float(self.cfg.world_camera_fovy))
        # Additional world cameras, same constraint (before compile()).  Recorded in
        # ``extra_camera_names`` so the caller can find them by name afterwards.
        self.extra_camera_names = []
        for cspec in (self.cfg.extra_cameras or ()):
            kwargs = dict(cspec)
            name = str(kwargs.pop("name"))
            kwargs["pos"] = [float(v) for v in kwargs.get("pos", (0.0, 0.0, 100.0))]
            if "xyaxes" in kwargs:
                kwargs["xyaxes"] = [float(v) for v in kwargs["xyaxes"]]
            if "fovy" in kwargs:
                kwargs["fovy"] = float(kwargs["fovy"])
            self.world.mjcf_root.worldbody.add_camera(name=name, **kwargs)
            self.extra_camera_names.append(name)
        # Extra MATERIALS, BEFORE compile() because a geom cannot reference a material
        # that does not exist yet (measured: "material 'marker_emissive' not found in
        # geom 2").  Used by the arena markers, which must be emissive so they are the
        # only thing at the top of the range in the meadow.
        self.extra_material_names = []
        for mspec in (getattr(self.cfg, "extra_materials", ()) or ()):
            kwargs = dict(mspec)
            name = str(kwargs.pop("name"))
            self.world.mjcf_root.add_material(name=name, **kwargs)
            self.extra_material_names.append(name)
        # Eye cameras are part of the FLY and were created by add_vision() ABOVE,
        # before add_fly(); see the ordering note there.
        # The declarative scene layer is applied to the MJCF spec HERE, before
        # compile(), because friction rows, heightfields, lights and objects are
        # all compile-time properties of the model.  None -> untouched, which is
        # the historical (degenerate) scene.
        self.scene_report = None
        self.scene_config = None
        self.scene_ledger = None
        if self.cfg.scene_preset is not None:
            from engine.embodied import scene as _scene_mod
            presets = dict(_scene_mod.SCENE_PRESETS)
            appliers = {p: _scene_mod.apply_scene for p in presets}
            self.scene_ledger = _scene_mod.PARAMETER_LEDGER
            # An OPTIONAL second scene layer lives in its own module so that
            # neither file has to be edited when the other grows.  It is imported
            # defensively: its absence must not break the historical presets, and
            # a broken import must be visible rather than silently swallowed.
            natural_error = None
            try:
                from engine.embodied import natural_scene as _nat_mod
            except ImportError:
                _nat_mod = None
                natural_error = "engine.embodied.natural_scene is not installed"
            except Exception as _exc:  # pragma: no cover - surfaced, not hidden
                _nat_mod = None
                natural_error = (f"engine.embodied.natural_scene failed to import: "
                                 f"{type(_exc).__name__}: {_exc}")
            if _nat_mod is not None:
                _nat_presets = dict(getattr(_nat_mod, "NATURAL_PRESETS", {}) or {})
                presets.update(_nat_presets)
                for _p in _nat_presets:
                    appliers[_p] = getattr(_nat_mod, "apply_natural_scene",
                                           _scene_mod.apply_scene)
                self.scene_ledger = getattr(_nat_mod, "NATURAL_PRESET_LEDGER",
                                            self.scene_ledger)
            if self.cfg.scene_preset not in presets:
                raise KeyError(
                    f"unknown scene preset {self.cfg.scene_preset!r}; available: "
                    f"{sorted(presets)}"
                    + ("" if natural_error is None else f" ({natural_error})"))
            self.scene_config = presets[self.cfg.scene_preset]
            _apply = appliers[self.cfg.scene_preset]
            try:
                self.scene_report = _apply(self.world, self.fly,
                                           self.scene_config,
                                           ledger=self.scene_ledger)
            except TypeError:
                # a scene layer that does not take a ledger keyword
                self.scene_report = _apply(self.world, self.fly,
                                           self.scene_config)
        # Static scene objects go in before compile() for the same reason.
        self.extra_geom_names = []
        if self.cfg.extra_geoms:
            from flygym.utils.mjcf import GEOM_TYPES
            for gspec in self.cfg.extra_geoms:
                kwargs = dict(gspec)
                gtype = kwargs.pop("type", "sphere")
                kwargs["type"] = GEOM_TYPES.get(gtype, gtype)
                self.world.mjcf_root.worldbody.add_geom(**kwargs)
                self.extra_geom_names.append(str(kwargs.get("name", "?")))
        self.model, self.data = self.world.compile()
        self.sim = Simulation(self.world, timestep=self.cfg.timestep_s)
        # DEFECT FIXED HERE (measured, and it had been silently poisoning every
        # render path): Simulation() builds its OWN MjModel/MjData from the world,
        # so the pair returned by world.compile() is a SEPARATE, never-advanced
        # copy.  Measured before this fix:
        #     backend.model is backend.sim.mj_model  -> False
        #     backend.data.qpos[:3]     = [0.496, 0.000, 1.800]   (spawn, frozen)
        #     backend.sim.mj_data.qpos[:3] = [0.735, 0.013, 1.059] (actually stepping)
        # Rendering backend.model with backend.data therefore drew a stale,
        # never-forwarded state and produced a pure-255 blank frame for EVERY
        # camera, while the same camera on sim.mj_model/sim.mj_data rendered
        # correctly (mean 142.2, std 16.7).  That is why an earlier attempt at
        # see-through rendering wrote qpos into backend.data and got a blank image
        # and why changing a geom's alpha appeared to have no effect.  Aliasing the
        # two pairs makes every caller that reaches for backend.model/backend.data
        # touch the LIVE state.
        self.model = self.sim.mj_model
        self.data = self.sim.mj_data
        if self.model is None or self.data is None:  # pragma: no cover
            raise RuntimeError("Simulation did not expose mj_model/mj_data")
        # THE PAYLOAD IS APPLIED HERE, AND THE PLACEMENT IS THE WHOLE BUG THAT MADE IT LOOK
        # BROKEN.  It must come AFTER ``self.model`` is aliased to the LIVE model.
        #
        # MEASURED: with the block above the two aliasing lines, the mass edit was applied to
        # the ``world.compile()`` model -- a SEPARATE object from the one ``Simulation`` builds
        # (a defect this file already documents) -- and then ``self.model`` was overwritten
        # with the live model, so the edit vanished.  Every read-back showed "+0.0000 %" while
        # ``write_verified`` was True, because the write HAD succeeded, on the wrong object.
        #
        # See also electrode_payload.attach_payload: the payload is a MASS EDIT, not extra
        # bodies, because adding bodies silently removed fly bodies at compile time (measured:
        # nbody 69 -> 66 with the head gone for one payload body; 64 and zero eyes for five).
        self.payload_application = None
        if getattr(self, "payload", None) is not None:
            from electrode_payload import apply_payload_to_model
            self.payload_application = apply_payload_to_model(self.model, self.payload)
            if self.payload_report is not None:
                self.payload_report["report"]["application"] = self.payload_application
        self.sim.warmup()

        # The camera is namespaced by the fly name inside the compiled model
        # (measured: the camera added as "trackcam" appears as "nmf/trackcam").
        # set_renderer() needs the QUALIFIED name, so expose it here rather than
        # letting callers guess and hit "Camera ... not found in the model".
        self.camera_spec = None
        self.world_camera_spec = None
        if self.cfg.add_world_camera:
            for i in range(self.model.ncam):
                cname = str(self.model.camera(i).name)
                if cname.endswith(self.cfg.world_camera_name):
                    self.world_camera_spec = cname
                    break
            if self.world_camera_spec is None:
                raise RuntimeError("world camera was requested but is absent from the "
                                   "compiled model")
        if self.cfg.add_tracking_camera:
            for i in range(self.model.ncam):
                cname = str(self.model.camera(i).name)
                if cname.endswith(self.cfg.tracking_camera_name):
                    self.camera_spec = cname
                    break
        # Eye-camera names are namespaced by the fly name exactly like the
        # tracking camera ("l_eye_camera" -> "nmf/l_eye_camera"), so resolve them
        # from the compiled model rather than guessing the prefix.
        self.eye_camera_specs = {}
        if self.cfg.add_vision:
            all_cams = [str(self.model.camera(i).name)
                        for i in range(self.model.ncam)]
            for eye_body in self.eye_cameras_added:
                wanted = f"{eye_body}_camera"
                for cname in all_cams:
                    if cname == wanted or cname.endswith("/" + wanted):
                        self.eye_camera_specs[eye_body] = cname
                        break
            if len(self.eye_camera_specs) != len(self.eye_cameras_added):
                raise RuntimeError(
                    "add_vision() reported "
                    f"{sorted(self.eye_cameras_added)} but only "
                    f"{sorted(self.eye_camera_specs)} were found in the compiled "
                    f"model (ncam={self.model.ncam}, "
                    f"cameras={all_cams})")
            # Read the fovy back from the COMPILED model: the compose API offers no
            # fovy keyword, so this is the only measured value.
            fovys = []
            for cname in self.eye_camera_specs.values():
                cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                        cname)
                fovys.append(float(self.model.cam_fovy[cid]))
            self.cfg.eye_fovy_deg_measured = float(np.mean(fovys))
        self.bodyseg_names = [str(b) for b in self.fly.get_bodysegs_order()]
        self.thorax_index = (self.bodyseg_names.index("c_thorax")
                             if "c_thorax" in self.bodyseg_names else 0)

        from flygym_demo.complex_terrain.common import get_default_locomotion_dof_order
        self.dof_order = get_default_locomotion_dof_order()

        self._controller = None
        self.using_cpg_baseline = False

    # ------------------------------------------------------------------ setup
    def attach_cpg_baseline(self):
        """Attach FlyGym's OWN tripod CPG.  ENGINEERING BASELINE, not the brain."""
        from flygym_demo.complex_terrain.cpg_controller import (
            CPGController, make_tripod_cpg_network)
        from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps
        net = make_tripod_cpg_network(timestep=self.sim.timestep,
                                      intrinsic_frequency=self.cfg.cpg_intrinsic_frequency_hz,
                                      seed=self.cfg.seed)
        self._controller = CPGController(net, PreprogrammedSteps(),
                                         output_dof_order=self.dof_order)
        self.using_cpg_baseline = True
        return self

    def attach_action_source(self, callable_source):
        """Attach any object with ``step() -> LocomotionAction``.

        This is the seam where a connectome-derived motor command will later
        replace the CPG baseline.  The action type is FlyGym's own, so the
        neural side never needs to know about MuJoCo.
        """
        if not hasattr(callable_source, "step"):
            raise TypeError("action source must have a step() method")
        self._controller = callable_source
        self.using_cpg_baseline = getattr(callable_source, "is_cpg_baseline", False)
        return self

    # ------------------------------------------------------------------ loop
    def step(self):
        """Advance one physics step; the attached action source is applied first."""
        if self._controller is None:
            raise RuntimeError("no action source attached: call attach_cpg_baseline() "
                               "or attach_action_source() first")
        from flygym_demo.complex_terrain.common import apply_locomotion_action
        apply_locomotion_action(self.sim, self.fly.name, self._controller.step())
        self.sim.step()

    def observe(self) -> BodyObservation:
        sim, name = self.sim, self.fly.name
        bodies = np.asarray(sim.get_body_positions(name), dtype=float)
        ci = sim.get_ground_contact_info(name)
        if len(ci) != 6:
            raise RuntimeError(f"ground contact info changed shape: got {len(ci)} entries, "
                               f"expected 6 per the backend contract")
        found = np.asarray(ci[0], dtype=float)      # raw channel, can be > 1
        forces = np.asarray(ci[1], dtype=float)     # contact frame
        torques = np.asarray(ci[2], dtype=float)    # contact frame
        positions = np.asarray(ci[3], dtype=float)  # global
        if found.shape != (6,) or forces.shape != (6, 3):
            raise RuntimeError("ground contact arrays are not (6,) and (6,3)")
        return BodyObservation(
            time_s=float(sim.time),
            joint_angles_rad=np.asarray(sim.get_joint_angles(name), dtype=float),
            joint_velocities_rad_s=np.asarray(sim.get_joint_velocities(name), dtype=float),
            body_positions_mm=bodies,
            body_rotations_wxyz=np.asarray(sim.get_body_rotations(name), dtype=float),
            thorax_position_mm=bodies[self.thorax_index].copy(),
            contact_found_raw=found,
            contact_forces=forces,
            contact_torques=torques,
            contact_positions_mm=positions,
            actuator_forces=np.asarray(
                sim.get_actuator_forces(name, self._ActuatorType.POSITION), dtype=float),
        )

    # ------------------------------------------------------------------ meta
    # ---------------------------------------------------------------- vision
    def _require_vision(self):
        if not self.cfg.add_vision:
            raise RuntimeError(
                "vision is OFF for this backend: BodyConfig.add_vision is False, so "
                "the compiled model has no eye cameras and FlyGym's "
                "get_raw_vision() cannot work.  Rebuild with "
                "BodyConfig(add_vision=True).  (This is a real capability gap, not "
                "a numerical result: with add_vision off there is NO visual "
                "pathway at all, which is why the 'notice an object' acceptance "
                "task is recorded as NOT RUN.)")

    def raw_vision(self) -> np.ndarray:
        """Fisheye-corrected RGB from the two compound eyes: (2, H, W, 3).

        Approximate optics -- see the caveat in describe()['vision'].
        """
        self._require_vision()
        return self.sim.get_raw_vision(self.cfg.fly_name)

    def ommatidia_readouts(self) -> np.ndarray:
        """Per-ommatidium yellow/pale channel readings: (2, n_ommatidia, 2).

        Approximate optics -- see the caveat in describe()['vision'].
        """
        self._require_vision()
        return self.sim.get_ommatidia_readouts(self.cfg.fly_name)

    def describe(self):
        return {
            "engine": "FlyGym/NeuroMechFly (Apache-2.0, EPFL Ramdya lab) on MuJoCo",
            "gl_backend": self.gl_backend,
            "units": self.cfg.units(),
            "geometry_note": GEOMETRY_NOTE,
            "fly_config": {k: getattr(self.cfg, k) for k in
                           ("fly_name", "timestep_s", "spawn_position_mm",
                            "spawn_quat_wxyz",
                            "world_half_size_mm", "cpg_intrinsic_frequency_hz",
                            "joint_stiffness", "joint_damping", "actuator_gain",
                            "add_adhesion", "seed")},
            "model": {"n_body": int(self.model.nbody), "n_qpos": int(self.model.nq),
                      "n_qvel": int(self.model.nv), "n_actuator": int(self.model.nu),
                      "n_geom": int(self.model.ngeom),
                      "gravity_mm_s2": list(np.asarray(self.model.opt.gravity, float))},
            "n_bodysegs": len(self.bodyseg_names),
            "n_joint_dof": len(self.dof_order),
            "locomotion_dof_order": [str(d) for d in self.dof_order],
            "n_actuators_model": int(self.model.nu),
            "actuator_note": ("48 = 42 position actuators on the locomotion DOFs "
                              "+ 6 per-leg adhesion actuators (measured)"),
            "thorax_index": self.thorax_index,
            "cameras": [str(self.model.camera(i).name) for i in range(self.model.ncam)],
            "tracking_camera": self.camera_spec,
            "world_camera": self.world_camera_spec,
            "vision": {
                "enabled": bool(self.cfg.add_vision),
                "eye_cameras": dict(self.eye_camera_specs),
                "eye_fovy_deg_measured": float(self.cfg.eye_fovy_deg_measured),
                "n_cameras_total": int(self.model.ncam),
                "caveat": ("FlyGym's fisheye Retina is calibrated for FlyGym's OWN "
                           "eye placement; ommatidia readouts on the NeuroMechFly "
                           "body are APPROXIMATE optics, not measured Drosophila "
                           "optics."),
                "why_it_was_missing": ("this fly is built without add_vision(), so "
                                       "the compiled model has no eye cameras and "
                                       "get_raw_vision() raises; that is the "
                                       "concrete reason the closed loop had no "
                                       "visual pathway"),
            },
            "scene": {
                "preset": self.cfg.scene_preset,
                "applied": (None if self.scene_report is None
                            else getattr(self.scene_report, "as_dict",
                                         lambda: self.scene_report)()),
                "nlight": int(self.model.nlight),
                "nhfield": int(getattr(self.model, "nhfield", 0)),
                "note_baseline": ("scene_preset=None reproduces the historical "
                                  "scene: one flat plane, ONE uniform friction row "
                                  "[1.0, 0.005, 0.0001] shared by every geom, and "
                                  "nlight = 0 (no lights)"),
            },
            "site_positions_note": "get_site_positions returns (0,3): this fly adds no "
                                   "anatomical joint sites; recorded, not hidden",
            "controller": ("FlyGym demo tripod CPG -- ENGINEERING BASELINE, not the "
                           "connectome" if self.using_cpg_baseline else "caller-supplied"),
        }

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass


def summarise_walk(times_s, thorax_mm, contact_present, cfg: BodyConfig,
                   contact_found_raw=None):
    """Reduce a recorded walk to pre-registered engineering metrics.

    The thresholds below are ENGINEERING GATES for this model, not biological
    normal values, and they are declared here (once) rather than tuned per run.
    """
    t = np.asarray(times_s, float)
    p = np.asarray(thorax_mm, float)
    c = np.asarray(contact_present)
    if c.dtype != bool:
        # accept a raw found channel too, but interpret it correctly
        c = c > 0
    c = c.astype(float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("thorax_mm must be (n,3)")
    if t.size != p.shape[0]:
        raise ValueError("times and positions must have equal length")
    disp = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
    dur = float(t[-1] - t[0]) if t.size > 1 else 0.0
    dz = p[:, 2]
    metrics = {
        "duration_s": dur,
        "horizontal_displacement_mm": disp,
        "mean_speed_mm_s": (disp / dur) if dur > 0 else float("nan"),
        "thorax_height_min_mm": float(dz.min()),
        "thorax_height_max_mm": float(dz.max()),
        "thorax_height_end_mm": float(dz[-1]),
        "contact_legs_mean": float(c.sum(axis=1).mean()) if c.size else 0.0,
        "contact_legs_min": float(c.sum(axis=1).min()) if c.size else 0.0,
        "contact_legs_max": float(c.sum(axis=1).max()) if c.size else 0.0,
        "per_leg_contact_fraction": (c.mean(axis=0).tolist() if c.size else []),
        "contact_semantics": "boolean per leg from (raw found channel > 0); "
                             "raw values can exceed 1 and are reported separately",
        "all_finite": bool(np.isfinite(p).all() and np.isfinite(c).all()),
        "units": {"displacement": MM, "speed": MM_PER_S, "height": MM,
                  "contact": "leg count (dimensionless)"},
    }
    if contact_found_raw is not None:
        rf = np.asarray(contact_found_raw, float)
        metrics["contact_found_raw_max"] = float(rf.max()) if rf.size else 0.0
        metrics["contact_found_exceeds_one"] = bool(rf.size and rf.max() > 1.0)
    # pre-registered engineering gates
    metrics["gate_moved"] = bool(disp >= 20.0)          # >= 20 mm in the window
    metrics["gate_upright"] = bool(0.3 <= metrics["thorax_height_end_mm"] <= 3.0)
    metrics["gate_finite"] = bool(metrics["all_finite"])
    metrics["gate_used_legs"] = bool(metrics["contact_legs_mean"] >= 1.0)
    metrics["passed_engineering_gates"] = bool(
        metrics["gate_moved"] and metrics["gate_upright"]
        and metrics["gate_finite"] and metrics["gate_used_legs"])
    return metrics


def select_gl_backend(candidates=GL_BACKENDS):
    """Return the first GL backend that can actually render here.

    Measured, not assumed: on this machine libOSMesa is absent and there is no
    CUDA GPU, so 'egl' is expected to win.  The probe renders one frame.
    """
    import subprocess
    import sys
    src = ("import os,numpy as np;"
           "from flygym.compose import FlatGroundWorld;"
           "from flygym.simulation import Renderer,Simulation;"
           "from flygym.utils.math import Rotation3D;"
           "from flygym_demo.complex_terrain.common import make_locomotion_fly;"
           "f=make_locomotion_fly('p');w=FlatGroundWorld(name='w',half_size=5);"
           "w.add_fly(f,spawn_position=[0,0,0.5],spawn_rotation=Rotation3D('quat',[1,0,0,0]));"
           "m,d=w.compile();s=Simulation(w,timestep=1e-4);s.warmup();"
           "r=Renderer(m,'track');s.set_renderer(r);s.step();"
           "fr=np.asarray(r.render());"
           "print('RENDER_OK' if fr.std()>1.0 else 'RENDER_BLANK')")
    tried = []
    for gl in candidates:
        env = dict(os.environ)
        if gl is None:
            env.pop("MUJOCO_GL", None)
        else:
            env["MUJOCO_GL"] = gl
        try:
            proc = subprocess.run([sys.executable, "-c", src], capture_output=True,
                                  text=True, env=env, timeout=600)
            ok = "RENDER_OK" in proc.stdout
            tried.append({"backend": gl or "default", "ok": ok,
                          "stdout_tail": proc.stdout.strip().splitlines()[-1:] ,
                          "stderr_tail": proc.stderr.strip().splitlines()[-2:]})
            if ok:
                return (gl if gl is not None else None), tried
        except subprocess.TimeoutExpired:
            tried.append({"backend": gl or "default", "ok": False, "error": "timeout"})
    return None, tried
