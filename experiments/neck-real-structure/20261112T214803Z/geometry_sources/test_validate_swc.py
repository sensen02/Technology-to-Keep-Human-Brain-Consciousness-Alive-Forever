"""Synthetic fixture tests only; no actual morphology or biological validity claim."""
from pathlib import Path
import tempfile,unittest,hashlib
from validate_swc import parse_swc,validate,GeometryError
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.swc=self.root/'720575941352057904_skeleton.swc';self.swc.write_text('# toy fixture NOT BIOLOGY\n1 2 1 300000 2 0 -1\n2 2 1 400000 2 0 1\n3 2 1 500000 2 0 2\n');source=self.root/'evidence.txt';source.write_text('SYNTHETIC_EVIDENCE_FOR_SOFTWARE_TEST_ONLY')
  def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
  self.meta=dict(schema='BANC_V888_SWC_GEOMETRY_CHECK_V1',dataset='BANC',materialization=888,root_id='720575941352057904',coordinate_space='BANC_NATIVE',coordinate_units='nm',swc_sha256=sha(self.swc),source_evidence=[dict(path=str(source),sha256=sha(source))],version_identity_evidence=dict(root_888='720575941352057904',verified=True),planes=dict(voxel_size_nm=[4,4,45],voxel_y=[92500,121000],nm_y=[370000.,484000.],source_documentation_url='https://example.invalid/SYNTHETIC_FIXTURE',definition='CURATED_PROFILE_SEED_PLANES_NOT_SYNAPSES'))
 def tearDown(self):self.tmp.cleanup()
 def test_intersections_not_admission(self):
  r=validate(self.swc,self.meta);self.assertTrue(r['each_plane_intersected']);self.assertFalse(r['training_admitted']);self.assertEqual(len(r['plane_intersections']['370000.0']),1)
 def test_duplicate_node(self):
  self.swc.write_text('1 2 0 0 0 0 -1\n1 2 1 1 1 0 -1\n')
  with self.assertRaises(GeometryError):parse_swc(self.swc)
 def test_parent_missing(self):
  self.swc.write_text('1 2 0 0 0 0 -1\n2 2 1 1 1 0 99\n')
  with self.assertRaises(GeometryError):parse_swc(self.swc)
 def test_cycle(self):
  self.swc.write_text('1 2 0 0 0 0 -1\n2 2 1 1 1 0 3\n3 2 1 1 1 0 2\n')
  with self.assertRaises(GeometryError):parse_swc(self.swc)
 def test_nonfinite(self):
  self.swc.write_text('1 2 nan 0 0 0 -1\n')
  with self.assertRaises(GeometryError):parse_swc(self.swc)
 def test_units_not_voxels(self):
  self.meta['coordinate_units']='voxel'
  with self.assertRaises(GeometryError):validate(self.swc,self.meta)
 def test_plane_not_guessed(self):
  self.meta['planes']['nm_y']=[92500,121000]
  with self.assertRaises(GeometryError):validate(self.swc,self.meta)
 def test_root_not_silent_latest(self):
  self.meta['version_identity_evidence']['root_888']='999'
  with self.assertRaises(GeometryError):validate(self.swc,self.meta)
 def test_changed_geometry_rejected(self):
  self.swc.write_text(self.swc.read_text()+'#changed\n')
  with self.assertRaises(GeometryError):validate(self.swc,self.meta)
if __name__=='__main__':unittest.main()
