"""Derived reference gated by explicit scale contract; no physical conversion guessed."""
from pathlib import Path
import json,numpy as np,hashlib
S=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-3d/20261027T150000Z')
O=Path('/home/sensen/Desktop/engine修复验收/real-freewalking-scale/20261027T151500Z')
def main():
 contract=json.loads((O/'SCALE_CONTRACT.json').read_text())
 assert contract['physical_length_scale']['status']=='NOT_ESTABLISHED'
 assert contract['physical_length_scale']['source_to_mm'] is None
 with np.load(S/'REFERENCE.npz') as z:
  arrays={k:z[k].copy() for k in z.files}
  # Preserve original quantities but label them as processed-source coordinates.
  arrays['processed_source_coordinate_change']=arrays.pop('body_displacement')
  np.savez_compressed(O/'REFERENCE_SCALE_GATED.npz',**arrays)
  for k in ('raw_keypoints','foot_world','foot_body','body_basis_world_columns'):
   np.testing.assert_array_equal(arrays[k],z[k])
  center=z['body_reference'];np.testing.assert_array_equal(arrays['processed_source_coordinate_change'],center-center[0])
  inverse=np.matmul(z['foot_body'],np.swapaxes(z['body_basis_world_columns'],1,2))+center[:,None,:]
  np.testing.assert_allclose(inverse,z['foot_world'],atol=1e-12)
  t=z['source_frame']/800.;np.testing.assert_array_equal(t,z['time_from_recording_start_s'])
 limits=dict(body_displacement_mm=None,foot_world_mm=None,foot_body_mm=None,
             physical_distance_status='NOT_ESTABLISHED',time_status='FRAME_DIVIDED_BY_PAPER_RATE_NOT_ACQUISITION_TIMESTAMPS',
             confidence=None,measured_contact=None,step_sequence=None)
 (O/'PHYSICAL_FIELDS.json').write_text(json.dumps(limits,indent=2)+'\n')
 (O/'VERIFICATION.json').write_text(json.dumps(dict(engineering_passed=True,exit_code=0,source_values_unchanged=True,inverse_transform_verified=True,frame_rate_arithmetic_verified=True,physical_units_verified=False,no_unit_conversion=True),indent=2)+'\n')
 print('exit0: source arrays retained, inverse transform verified; physical scale NOT_ESTABLISHED')
if __name__=='__main__':main()
