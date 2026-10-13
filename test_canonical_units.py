"""Regression tests for the unit-semantics correction (mm, not cm) and for the physics anchor.

THE TWO THINGS THAT MUST BOTH HOLD
----------------------------------
1. The LABELS are now right: the model length unit is the millimetre, so 2.04535 model units
   is 2.04535 mm.  Round 39 declared centimetres and "corrected" historical reports by 10x in
   the wrong direction; these tests fail if that claim ever returns.
2. The PHYSICS did not move: `seat_root` seats the body at the same model-space penetration
   the historical code used (-0.05).  Fixing a label by moving the physics would have been a
   second, worse bug, so the seating value is asserted numerically here.

The evidence for (1) is also pinned: if the canonical asset is ever regenerated with different
mesh scaling or a different header, these tests fail rather than letting the claim drift.

Run with the mujoco-capable interpreter:
    venv_body/bin/python test_canonical_units.py
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import engine.canonical_physics as cp  # noqa: E402


class UnitSemanticsTests(unittest.TestCase):
    def test_length_unit_is_millimetre_not_centimetre(self):
        self.assertEqual(cp.MODEL_LENGTH_UNIT_NAME, "mm")
        self.assertEqual(cp.MM_PER_MODEL_LENGTH, 1.0)
        # the exact round-39 mistake, stated as a failing condition
        self.assertNotEqual(cp.MM_PER_MODEL_LENGTH, 10.0,
                            "the centimetre declaration was retracted; 10.0 must not return")

    def test_conversion_is_identity_and_round_trips(self):
        self.assertAlmostEqual(float(cp.model_to_mm(2.04535)), 2.04535, places=12)
        self.assertAlmostEqual(float(cp.mm_to_model(2.04535)), 2.04535, places=12)

    def test_standing_height_reads_as_two_millimetres(self):
        """The decisive number: a Drosophila thorax height is ~2 mm, not ~20 mm."""
        model_value = 2.04535                      # canonical keyframe qpos[2]
        self.assertAlmostEqual(float(cp.model_to_mm(model_value)), 2.045, places=3)
        self.assertLess(float(cp.model_to_mm(model_value)), 5.0,
                        "a >5 mm thorax height would mean the centimetre reading is back")
        # and 2 mm is the right order for the animal, whereas 20 mm is 10x too big
        self.assertGreater(float(cp.model_to_mm(model_value)), 1.0)

    def test_seating_penetration_physics_is_unchanged(self):
        """THE ANCHOR: the model-space penetration must stay at the historically applied -0.05."""
        self.assertAlmostEqual(cp.SEATING_PENETRATION_MODEL, -0.05, places=12)
        self.assertAlmostEqual(cp.SEATING_PENETRATION_MM, -0.05, places=12)
        self.assertAlmostEqual(float(cp.mm_to_model(cp.SEATING_PENETRATION_MM)), -0.05, places=12)

    def test_gravity_is_consistent_with_millimetres(self):
        self.assertAlmostEqual(cp.GRAVITY_MODEL * 1e-3, 9.801, places=6)
        # the centimetre reading would give 98 m/s^2, which is not a planet
        self.assertGreater(abs(cp.GRAVITY_MODEL * 1e-2 - 9.801), 1.0)

    def test_mass_unit_is_reported_as_inferred_not_established(self):
        self.assertEqual(cp.MODEL_MASS_UNIT_STATUS,
                         "INFERRED_FROM_MESH_VOLUME_DENSITY_ARGUMENT_NOT_ESTABLISHED")
        doc = cp.write_result.__doc__ or ""
        self.assertNotIn("mg = 1e-6 kg", doc)
        self.assertAlmostEqual(cp.MODEL_MASS_UNIT_KG, 1e-3, places=12)

    def test_embodied_backend_agrees_with_canonical(self):
        """`body_backend` keeps its own copy and says it must agree; enforce it.

        It previously contradicted ITSELF: one block said centimetres while GEOMETRY_NOTE in
        the same file said millimetres.
        """
        try:
            import engine.embodied.body_backend as bb
        except ImportError as exc:                      # pragma: no cover
            self.skipTest(f"body_backend import failed: {exc}")
        self.assertEqual(bb.MODEL_LENGTH_UNIT_NAME, cp.MODEL_LENGTH_UNIT_NAME)
        self.assertEqual(bb.MM_PER_MODEL_LENGTH, cp.MM_PER_MODEL_LENGTH)


class AssetEvidenceTests(unittest.TestCase):
    """Pin the evidence itself, so the unit claim cannot silently drift."""

    @classmethod
    def setUpClass(cls):
        cls.xml = cp.CANONICAL_XML.read_text()

    def test_asset_hash_unchanged(self):
        self.assertEqual(cp.sha256_file(cp.CANONICAL_XML), cp.CANONICAL_XML_SHA256)

    def test_all_meshes_are_scaled_by_1000(self):
        scales = re.findall(r'<mesh[^>]*scale="([^"]+)"', self.xml)
        self.assertEqual(len(scales), 71, "mesh count changed; re-derive the unit evidence")
        self.assertEqual(set(scales), {"1000 1000 1000"},
                         "mesh scale is the primary evidence that the cached STL (metres) is "
                         "converted to millimetres")

    def test_generator_header_itself_says_mm(self):
        head = self.xml[:2000]
        self.assertIn("z=2.0453 mm", head,
                      "the generator's own comment states the unit; losing it loses the evidence")

    def test_gravity_matches_the_millimetre_family(self):
        self.assertIn('gravity="0 0 -9801"', self.xml)
        upstream_cm = (HERE / "venv_body/lib/python3.12/site-packages/flygym/assets/model/"
                              "flybody/fruitfly.xml")
        if upstream_cm.exists():
            self.assertIn('gravity="0 0 -981"', upstream_cm.read_text(),
                          "sibling asset of the centimetre family; if this changed, "
                          "re-derive the gravity argument")


class StancePhysicsTests(unittest.TestCase):
    """The gate must still seat the round-37 pose, at the same model-space height."""

    def test_gated_seat_is_unchanged_and_passes(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not importable in this interpreter")
        import json
        import numpy as np
        import run_fix_round39 as r39
        model = cp.load_model()
        data = mujoco.MjData(model)
        r39.apply_round37_pose(model, data, json.loads(r39.ROUND37_POSE.read_text()))
        base = data.qpos.copy()
        z = cp.seat_root(model, data, base)
        data.qpos[:] = base
        data.qpos[2] = z
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        facts = cp.stance_facts(model, data)          # raises unless this is a stance
        # the same model-space height the previous (mis-labelled) build reported as "20.449 mm"
        self.assertAlmostEqual(float(z), 2.0449, places=3,
                               msg="seating height moved; the label fix must not move physics")
        self.assertAlmostEqual(float(cp.model_to_mm(z)), 2.0449, places=3)
        self.assertTrue(facts.tarsus_only)
        self.assertGreaterEqual(facts.worst_floor_penetration_model,
                                cp.SEATING_PENETRATION_MODEL - 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
