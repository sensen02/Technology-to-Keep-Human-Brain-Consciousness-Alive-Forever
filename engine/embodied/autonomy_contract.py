"""Fail-closed reporting contract; not proof that a runtime has no hidden controls.

The manifest must be backed by executable source/current/control audits. Unknown
fields cannot become evidence merely because a caller omitted a prohibited route.
"""
REQUIRED_FALSE = ('external_locomotion_permission', 'target_error_to_motor_current',
                  'independent_gait_controller', 'independent_posture_controller',
                  'scheduled_motor_commands')
REQUIRED_TRUE = ('sensory_inputs_from_body_environment_only', 'all_active_commands_from_neural_output',
                 'runtime_command_origin_audited', 'sensory_mapping_evidence_verified',
                 'motor_mapping_evidence_verified', 'neural_pathway_interventions_verified',
                 'functional_closed_loop_verified')


def assess_autonomy(manifest):
    failures=[]
    for key in REQUIRED_FALSE:
        if manifest.get(key) is not False:
            failures.append(key + ': must be explicitly false and source-audited')
    for key in REQUIRED_TRUE:
        if manifest.get(key) is not True:
            failures.append(key + ': missing or unverified evidence')
    return {'autonomous_goal_status': 'DECLARED_EVIDENCE_GATES_MET_REQUIRES_REVIEW' if not failures else 'NOT_ACHIEVED',
            'unmet_requirements': failures,
            'limitation': 'Manifest gating alone does not prove source correctness or biological autonomy.'}
