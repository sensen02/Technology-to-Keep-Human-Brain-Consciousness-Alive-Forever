"""Bounded explicit lightweight regression suite; no full neural/body episodes.

Individual audit tests may read captured artifacts; PASS is not standing.
"""
from pathlib import Path
import io,json,unittest,time,hashlib
MODULES='test_closure_clock test_closure_bundle test_channel_connectivity test_perturbation_inference test_neural_command_audit test_motor_manifest_validation test_neural_muscle_output test_sensory_channel_validation test_sensory_leg_assignment test_sensory_physics_observation test_initial_pose_contract test_audit_provenance test_touch_path_audit test_motor_state_audit test_connected_feedback_audit test_early_feedback_probe test_neural_pose_comparison test_neural_duration_probe test_spike_filter_reconstruction engine.embodied.test_source_snapshot'.split()
def main():
    started=time.monotonic();base=Path(__file__).resolve().parent
    suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES);log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=1).run(suite)
    sources={str(base/(name.replace('.','/')+'.py')):hashlib.sha256((base/(name.replace('.','/')+'.py')).read_bytes()).hexdigest() for name in MODULES}
    doc=dict(status='PASS_REGRESSIONS_NOT_AUTONOMY' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),modules=MODULES,test_source_hashes=sources,wall_s=time.monotonic()-started,log=log.getvalue(),limitations='Lightweight explicit subset, not all repository tests; no standing acceptance; no full neural replay.')
    dest=base/'outputs/neural_muscle_closure/consolidated_regressions.json';dest.write_text(json.dumps(doc,indent=2)+'\n');print(log.getvalue());print(dest)
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
