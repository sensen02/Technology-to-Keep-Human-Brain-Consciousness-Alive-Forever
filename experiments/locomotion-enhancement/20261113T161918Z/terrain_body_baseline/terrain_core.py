"""Explicit derived full-body Core factory; no physical import/construction on import.

planes is deliberately reinterpreted as REGISTERED TERRAIN MASK in this derived
class only. This covers inherited measure/contact_count/NativeRecorder and the
old teacher/PeakRecorder paths. The frozen implementation is never rewritten.
"""
from pathlib import Path
import ast
import importlib
import json
import sys

from prepare import HERE, BASE, XML, verify_prepared, sha
import terrain_metrics


def load_base_physics():
    sys.path.insert(0, str(BASE))
    module = importlib.import_module('physics')
    if Path(module.__file__).resolve() != BASE/'physics.py':
        raise ValueError('Frozen BASE physics import collision')
    return module


def derived_actuator_namespace(base):
    """Clone only the plane-required validation predicate, preserve finite PD code.

    Both load_full_model and IncrementalActuator.__init__ call validation. A
    box-only control would otherwise be rejected even with correct collision.
    The frozen module/globals are never monkeypatched or rewritten.
    """
    tree = ast.parse(Path(base.act_mod.__file__).read_text())
    target = ast.dump(ast.parse('np.any(model.geom_type == mj.mjtGeom.mjGEOM_PLANE)', mode='eval').body)

    class GroundPredicate(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            if ast.dump(node) == target:
                self.count += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='_registered_ground_present',ctx=ast.Load()),
                    args=[ast.Name(id='model',ctx=ast.Load())],keywords=[]),node)
            return self.generic_visit(node)

    transformer = GroundPredicate()
    transformed = transformer.visit(tree)
    if transformer.count != 1:
        raise ValueError('Expected exactly one frozen plane-required validator predicate')
    ast.fix_missing_locations(transformed)
    namespace = dict(__name__='_terrain_registered_ground_actuator',__file__=str(base.act_mod.__file__),
        _registered_ground_present=lambda model:any((model.geom(i).name or '').startswith('terrain_')
            and int(model.geom_bodyid[i]) == 0 and int(model.geom_group[i]) == 5 for i in range(model.ngeom)))
    exec(compile(transformed,str(base.act_mod.__file__)+'::single_registered_ground_predicate','exec'),namespace)
    return namespace


def core_type(base):
    np, mj = base.np, base.mj
    contact_module = terrain_metrics.contact_module()
    actuator_namespace = derived_actuator_namespace(base)

    class TerrainCore(base.Core):
        observation_version = 'TERRAIN_RELATIVE_PAD_CLEARANCE_238_V1'
        metric_version = terrain_metrics.VERSION

        def __init__(self, *, case, recorder=None, adhesion=False):
            if adhesion:
                raise ValueError('No adhesion allowed in terrain baseline')
            frozen = verify_prepared()
            ledger = json.loads((HERE/'PREPARED/MODEL_LEDGER.json').read_text())
            if case not in ledger['cases']:
                raise ValueError('Unregistered terrain case')
            self.terrain_case = case
            self.derived_xml = Path(ledger['cases'][case]['xml'])
            self.terrain_ledger = ledger['cases'][case]
            self._terrain_np, self._terrain_mj = np, mj
            self._terrain_contact_module = contact_module
            self._terrain_registry_ready = False
            self._terrain_frozen = frozen
            derived = self.derived_xml

            class ExplicitDerivedLoader:
                IncrementalActuator = actuator_namespace['IncrementalActuator']

                @staticmethod
                def load_full_model(requested, **kwargs):
                    if Path(requested).resolve() != XML:
                        raise ValueError('Unexpected parent Core XML request')
                    return actuator_namespace['load_full_model'](derived, **kwargs)

            super().__init__(act_mod=ExplicitDerivedLoader(), adhesion=False, recorder=recorder)
            self._ensure_terrain_registry()

        def _ensure_terrain_registry(self):
            if self._terrain_registry_ready:
                return
            names = [self.plant.geom(i).name or '' for i in range(self.plant.ngeom)]
            self.legacy_plane_mask = (self.plant.geom_type == mj.mjtGeom.mjGEOM_PLANE).copy()
            mask = np.asarray([name.startswith('terrain_') for name in names], dtype=bool)
            if not mask.any() or any(int(self.plant.geom_bodyid[i]) != 0 for i in np.flatnonzero(mask)):
                raise ValueError('Requires registered static world terrain geoms')
            self.terrain_ids = set(map(int, np.flatnonzero(mask)))
            self.foot_ids = set(map(int, np.flatnonzero(self.geom_leg >= 0)))
            if len(self.foot_ids) != 18 or any(len(self.pads[leg]) != 3 for leg in base.LEGS):
                raise ValueError('Original three-sphere feet must remain unchanged')
            if any(int(self.plant.geom_group[i]) != 5 for i in self.terrain_ids):
                raise ValueError('Terrain-only rays require terrain geom group5')
            if any(int(self.plant.geom_group[i]) == 5 for i in range(self.plant.ngeom) if i not in self.terrain_ids):
                raise ValueError('Nonterrain geom leaked into ray group5')
            # All frozen inherited loops testing c.planes now see every registered terrain.
            self.planes = mask
            self.ray_group_mask = np.array([0,0,0,0,0,1], dtype=np.uint8)
            maxheight = max(float(item['top_z']) for item in self.terrain_ledger['terrain'])
            self.ray_origin_z = maxheight + max(10., 2.*float(self.plant.stat.extent))
            self._terrain_registry_ready = True

        def measure(self):
            self._ensure_terrain_registry()
            # Same legacy world-z/XY arithmetic and ordering, but all terrain registered.
            value = super().measure()
            registered_mask = self.planes
            try:
                self.planes = self.legacy_plane_mask
                legacy_plane = super().measure()
            finally:
                self.planes = registered_mask
            world_padz = value['padz'].copy()
            relative_padz = np.zeros(6)
            foot_heights = []
            for li, leg in enumerate(base.LEGS):
                bottoms = []
                heights = []
                for geom in self.pads[leg]:
                    point = self.d.geom_xpos[geom]
                    ground, hit = terrain_metrics.height(self, point[:2])
                    bottoms.append(float(point[2]) - float(self.plant.geom_size[geom][0]) - ground)
                    heights.append(dict(geom=int(geom), ground_height=ground, terrain_id=hit))
                relative_padz[li] = min(bottoms)
                foot_heights.append(heights)
            com_world = float(self.d.subtree_com[self.th][2])
            com_ground, com_hit = terrain_metrics.height(self, self.d.subtree_com[self.th][:2])
            normal = terrain_metrics.contacts(self)
            value.update(padz=relative_padz, padz_world=world_padz,
                com=com_world-com_ground, com_world=com_world, com_ground_height=com_ground,
                com_ground_terrain_id=com_hit, pad_ground_samples=foot_heights,
                normal_loads=normal['normal_loads'], tangent_speed=normal['tangent_speed'],
                world_z_force_signed=normal['world_z_force_signed'],
                nonfoot_normal=normal['nonfoot_normal'], legacy_plane=legacy_plane,
                terrain_contact_count=normal['registered_contact_count'],
                legacy_plane_contact_count=normal['old_plane_contact_count'],
                metric_version=self.metric_version, observation_version=self.observation_version)
            return value

        def apply_adhesion(self, loads):
            # Do not silently clear an injected external force and then report zero.
            if self.adhesion_on or np.any(self.d.xfrc_applied) or np.any(self.d.qfrc_applied):
                raise ValueError('Terrain Core forbids adhesion/external forces')
            self._adhesion_wrench[:] = 0.
            self._adhesion_points = np.empty((0,8))
            self._adhesion_reference_body_com = self.d.xipos.copy()
            self._adhesion_refresh_time_s = float(self.d.time)

    return TerrainCore


def actual_identity(core):
    """Capture ACTUAL derived model, not legacy identity.capture's frozen XML path."""
    np = core._terrain_np
    arrays = {}
    for name in dir(core.plant):
        if name.startswith('_'):
            continue
        value = getattr(core.plant, name)
        if isinstance(value, np.ndarray):
            import hashlib
            arrays[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                               sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return dict(kind='ACTUAL_DERIVED_TERRAIN_BODY_IDENTITY_V1',
        xml=str(core.derived_xml), xml_sha256=sha(core.derived_xml),
        case=core.terrain_case, compiled_arrays=arrays,
        source_hashes=core._terrain_frozen['source_hashes'],
        observation_version=core.observation_version, metric_version=core.metric_version,
        planes_attribute_semantics='explicit registered terrain geom mask; not geom type plane',
        actuator_adapter='single AST plane-required validation predicate -> static named terrain group5; all finite PD methods unchanged',
        terrain_ids=sorted(core.terrain_ids), foot_ids=sorted(core.foot_ids),
        adhesion=False, nq=int(core.plant.nq), nv=int(core.plant.nv), nu=int(core.plant.nu),
        nmesh=int(core.plant.nmesh), neq=int(core.plant.neq),
        full_NN_pipeline_ready=False, full_body_locomotion_acceptance_claim=False)
