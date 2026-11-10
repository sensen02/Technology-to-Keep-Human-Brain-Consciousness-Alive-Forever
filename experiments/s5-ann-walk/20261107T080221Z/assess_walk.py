"""Add explicit heading requirements without editing the frozen physics verifier.
Teacher feasibility is kept separate from final teacher-free neural success.
"""
import importlib.util
import json
import sys
from pathlib import Path
import math
VERIFY=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T073258Z/verify_walk.py')
spec=importlib.util.spec_from_file_location('frozen_walk_verifier',VERIFY)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def main():
    v.main(sys.argv[1:])  # Return2 is an honest failed assessment, not a crash.
    folder=Path(sys.argv[1]);report=json.loads((folder/'VERIFICATION.json').read_text())
    g=report['gait']['locomotion'];seconds=report['release_seconds']
    forward=g['cycle_mean_body_forward_travel']
    lateral_ratio=abs(g['cycle_mean_body_lateral_travel'])/max(forward,1e-12)
    heading=dict(mean_yaw_drift_bounded=abs(g['heading_change_rad'])/seconds<=math.radians(2),
                 heading_range_bounded=g['heading_range_rad']<=.5,
                 straightness=g['straightness']>=.9,
                 lateral_to_forward=lateral_ratio<=.2)
    physical=dict(native_support=report['support']['passed'],six_leg_steps=report['gait']['all_legs_stepping'],
                  body_forward=g['sustained_forward_travel'],policy_integrator=report['policy']['passed'],
                  replay=bool(report['physics_replay']['performed'] and report['physics_replay']['passed']),
                  material_slip=report['material_slip_passed'] is True,
                  sample_duration=seconds>=3.-1e-9)
    screen=bool(all(physical.values()) and all(heading.values()))
    final=bool(report['verified_neural_locomotion'] and all(heading.values()))
    out=dict(status='VERIFIED_NEURAL_STRAIGHT_WALK' if final else 'NOT_COMPLETE',
        controller=report['controller'],release_seconds=seconds,
        heading_checks=heading,physical_screen_checks=physical,
        physical_screen_passed=screen,
        teacher_feasibility_only=screen and report['controller']=='teacher',
        final_neural_goal_passed=final,
        thresholds=dict(mean_absolute_yaw_drift_deg_per_s=2.,heading_range_rad=.5,
                        straightness=.9,lateral_to_forward=.2,
                        original_support_and_slip='unchanged frozen verifier'),
        observed=dict(heading_change_rad=g['heading_change_rad'],straightness=g['straightness'],
                      lateral_to_forward=lateral_ratio,forward_speed=g['cycle_mean_forward_speed']),
        provenance=dict(verifier=str(VERIFY),verifier_sha256=v.sha(VERIFY),
                        input=str(folder/'VERIFICATION.json'),input_sha256=v.sha(folder/'VERIFICATION.json'),
                        assessment_source=str(Path(__file__)),assessment_sha256=v.sha(__file__)),
        limitations='Heading thresholds are disclosed engineering targets, not biological calibration. Teacher movement never counts as neural success.')
    path=folder/'GOAL_ASSESSMENT.json'
    if path.exists():path.read_text()
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)
    return 0 if final else 2


if __name__=='__main__':raise SystemExit(main())
