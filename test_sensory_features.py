"""Deterministic S4 raw-observation diagnostics; not translator/biology validation.

Physical cases use the canonical compiled meshes and actual MuJoCo collision +
constraint solve. Only diagnostic in-memory collision masks/root setup change;
no XML is written. Mock pooling is explicitly NOT a physical contact experiment.
"""
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
from engine import canonical_physics as cp
from engine.embodied.whole_body_sensing import WholeBodySensing, canonical_region
from engine.embodied.sensory_features import SensoryFeatures
from engine.embodied import sensory_channels as sc

REPRESENTATIVES = ['Head_geom', 'Thorax_geom', 'A3_geom', 'LWing_geom'] + [
    leg.upper()+'Tarsus5_geom' for leg in sc.LEGS]


def sensing_for(model, reference=None):
    groups = {leg: {ch: [] for ch in sc.CHANNEL_SPEC} for leg in sc.LEGS}
    with patch('engine.embodied.whole_body_sensing.channel_index_map', return_value=(groups, {})):
        return WholeBodySensing(model, {'n': 0, 'tier': {'root_ids': []}},
                                model.qpos0 if reference is None else reference)


def physical_case(name, velocity=(0., 0., 0.), angular=(0., 0., 0.)):
    m = cp.load_model()
    d = mujoco.MjData(m)
    floor, target = m.geom('floor').id, m.geom(name).id
    # Diagnostic isolation, not the unchanged canonical collision configuration.
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.geom_contype[floor] = 1
    m.geom_conaffinity[target] = 1
    # Nontrivial root orientation exercises body/world transforms.
    quat = np.array([.93, .17, -.21, .24]); quat /= np.linalg.norm(quat)
    d.qpos[3:7] = quat
    mujoco.mj_forward(m, d)
    mesh = m.geom_dataid[target]
    vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
    world = vertices @ d.geom_xmat[target].reshape(3, 3).T + d.geom_xpos[target]
    # One-time root positioning is diagnostic fixture setup, NOT a controller.
    d.qpos[2] -= world[:, 2].min() + .005
    d.qvel[:3] = velocity
    d.qvel[3:6] = angular
    mujoco.mj_forward(m, d)
    return m, d, sensing_for(m)


def state_snapshot(m, d):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, state, spec)
    return state


class SensoryFeatureTests(unittest.TestCase):
    def test_all_42_hinges_signed_position_and_velocity_exact_localization(self):
        m = cp.load_model(); d = mujoco.MjData(m)
        sensing = sensing_for(m); features = SensoryFeatures(sensing)
        self.assertEqual(len(features.joints), 42)
        ids = {n: i for i, n in enumerate(features.names)}
        # Every hinge independently, both signs in each independent dimension.
        for name in features.joints:
            joint = m.joint(name).id
            qa, va = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
            for dp in (-.125, .125):
                for speed in (-2., 2.):
                    with self.subTest(joint=name, deviation=dp, velocity=speed):
                        d.qpos[:] = m.qpos0; d.qvel[:] = 0
                        d.qpos[qa] += dp; d.qvel[va] = speed
                        mujoco.mj_forward(m, d)
                        before = state_snapshot(m, d)
                        states = {r['name']: r for r in sensing.observe(d)['joint_states']}
                        result = features.observe(d)
                        np.testing.assert_array_equal(state_snapshot(m, d), before)
                        for other in features.joints:
                            j = m.joint(other).id; a = m.jnt_qposadr[j]
                            expected = (float(d.qpos[a]), float(d.qpos[a]-m.qpos0[a]), speed if other == name else 0.)
                            for field, value in zip(features.JOINT_FIELDS, expected):
                                self.assertEqual(states[other][field], value)
                                index = ids[f'joint/{other}/{field}']
                                self.assertEqual(result['values'][index], value)
                                self.assertTrue(result['valid'][index])

    def test_actual_localized_mesh_contacts_force_direction_and_read_only(self):
        for name in REPRESENTATIVES:
            with self.subTest(geom=name):
                m, d, sensing = physical_case(name)
                features = SensoryFeatures(sensing)
                before = state_snapshot(m, d)
                obs = sensing.observe(d); result = features.observe(d)
                np.testing.assert_array_equal(state_snapshot(m, d), before)
                rows = obs['contacts']
                self.assertTrue(rows)
                self.assertEqual({r['geom'] for r in rows}, {name})
                self.assertEqual({r['kind'] for r in rows}, {'external'})
                self.assertEqual({r['region'] for r in rows}, {canonical_region(name)[0]})
                self.assertFalse(obs['self_contact_enabled'])
                bid = m.geom_bodyid[m.geom(name).id]; rot = d.xmat[bid].reshape(3, 3)
                world_force = sum((rot @ r['force_body'] for r in rows), np.zeros(3))
                self.assertGreater(world_force[2], 0.)  # ground pushes up, NOT down
                # Independent generalized-force check: internal joint constraints
                # cannot change net free-root translational contact force.
                np.testing.assert_allclose(world_force, d.qfrc_constraint[:3], rtol=1e-10, atol=1e-10)
                for row, contact in zip(rows, [c for c in d.contact if c.efc_address >= 0]):
                    np.testing.assert_allclose(rot @ row['position_body'] + d.xpos[bid], contact.pos, atol=1e-12)
                    np.testing.assert_allclose(rot @ row['normal_body'], [0, 0, 1], atol=1e-12)
                    self.assertGreater(row['normal_force'], 0.)
                for site in features.sites:
                    idx = features.names.index(f'external/{site}/touch')
                    self.assertEqual(result['values'][idx], float(site == name))

    def test_actual_slip_translation_and_angular_velocity_against_point_jacobian(self):
        for velocity, angular in [((1.25, -.75, .4), (0, 0, 0)), ((.2, -.1, .3), (.7, -.4, .8))]:
            m, d, sensing = physical_case('Head_geom', velocity, angular)
            obs = sensing.observe(d); self.assertTrue(obs['contacts'])
            for row, contact in zip(obs['contacts'], [c for c in d.contact if c.efc_address >= 0]):
                body = m.geom_bodyid[m.geom(row['geom']).id]
                jac = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jac, None, contact.pos, body)
                point_velocity = jac @ d.qvel
                expected = point_velocity.copy(); expected[2] = 0.
                rot = d.xmat[body].reshape(3, 3)
                np.testing.assert_allclose(rot @ row['slip_velocity_body'], expected, atol=1e-12)
                self.assertAlmostEqual(row['slip_speed'], np.linalg.norm(expected), places=12)

    def test_mock_pooling_is_nonphysical_and_preserves_site_kind_masks(self):
        m = cp.load_model(); d = mujoco.MjData(m); sensing = sensing_for(m)
        features = SensoryFeatures(sensing); base = sensing.observe(d)
        def row(kind, load, position, slip):
            return dict(geom='Head_geom', kind=kind, normal_force=load, tangent_force=load/2,
                        force_body=[load, 0, 0], position_body=position, normal_body=[0, 0, 1],
                        slip_velocity_body=slip, slip_speed=float(np.linalg.norm(slip)))
        base['contacts'] = [row('external', 1., [1, 2, 3], [2, 0, 0]),
                            row('external', 3., [5, 6, 7], [-2, 0, 0]),
                            row('self', 2., [9, 8, 7], [0, 3, 0])]
        base['self_contact_enabled'] = True
        with patch.object(sensing, 'observe', return_value=base): result = features.observe(d)
        values = dict(zip(features.names, result['values']))
        expected = dict(touch=1., normal_force=4., tangent_force=2., force_x=4., force_y=0., force_z=0.,
                        position_x=4., position_y=5., position_z=6., normal_x=0., normal_y=0., normal_z=1.,
                        slip_velocity_x=-1., slip_velocity_y=0., slip_velocity_z=0., slip_speed=2.)
        for field, value in expected.items(): self.assertEqual(values[f'external/Head_geom/{field}'], value)
        self.assertEqual(values['self/Head_geom/normal_force'], 2.)
        masks = dict(zip(features.names, result['valid']))
        self.assertTrue(masks['external/Thorax_geom/touch'])
        self.assertFalse(masks['external/Thorax_geom/position_x'])
        for modality in ('airflow', 'hair_deflection', 'vibration', 'nociception'):
            self.assertFalse(masks['unmodelled/'+modality])
        base['self_contact_enabled'] = False; base['contacts'] = []
        with patch.object(sensing, 'observe', return_value=base): empty = features.observe(d)
        for i, name in enumerate(features.names):
            if name.startswith('self/'): self.assertFalse(empty['valid'][i])
        self.assertEqual(len(features.names), len(set(features.names)))
        self.assertEqual(len(features.names), len(empty['values']))

    def test_frozen_xml_identity(self):
        self.assertEqual(cp.sha256_file(cp.CANONICAL_XML), cp.CANONICAL_XML_SHA256)


if __name__ == '__main__':
    unittest.main(verbosity=2)
