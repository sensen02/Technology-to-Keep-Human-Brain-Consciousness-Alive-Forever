"""Explicit alternative-branch regression suite; no new full episodes."""
from pathlib import Path
import unittest,io,json,time,hashlib
MODULES=['test_zero_floor_branch','test_activation_timestep_audit','test_fine_clock_branch','test_timestep_constraint_audit','test_clock_refinement','test_fixed_replay_refinement','test_support_audit','test_joint_torque_deficiency','test_channel_transmitter_audit','test_sensory_frontend_balance','test_propagable_input_audit','test_declared_reference_branch','test_motor_output_census','test_cross_audit_consistency','test_model_annotation_gap_audit','test_passive_vs_actuated_support','test_commensurate_normaliser_branch','test_leg_motor_coverage_audit','test_model_provenance_structure_audit','test_campaniform_composition_audit','test_banc_release_check','test_tarsus_dof_structure','test_tarsus_dof_branch','test_flybody_anatomy_audit','test_anatomical_joint_axis_audit']
def main():
 root=Path(__file__).resolve().parent;start=time.monotonic();log=io.StringIO()
 result=unittest.TextTestRunner(stream=log,verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(MODULES))
 hashes={str(root/(m+'.py')):hashlib.sha256((root/(m+'.py')).read_bytes()).hexdigest() for m in MODULES}
 report=dict(status='PASS_BRANCH_TESTS_NOT_STANDING' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=time.monotonic()-start,modules=MODULES,test_source_hashes=hashes,log=log.getvalue(),scope='Existing branch evidence and small fixture checks, no new autonomous episodes; no standing or convergence claim.')
 p=root/'outputs/model_branches/branch_regressions.json';p.write_text(json.dumps(report,indent=2)+'\n');print(log.getvalue());print(p)
 return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
