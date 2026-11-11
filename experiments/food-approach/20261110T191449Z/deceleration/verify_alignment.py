"""Additional saved-evidence checks; no physics or source edits."""
from pathlib import Path
import json
import hashlib
import importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
    checks={};original=HERE.parent/'teacher'/'left_half'
    for name in ('halfspeed_straight','halfspeed_smallleft'):
        a=load(HERE/name/'TRACE_1MS.npz');s=load(HERE/name/'SAMPLES_5MS.npz')
        checks[name+'_all5_action_ticks_match_each_decision']=bool(np.array_equal(a['action'][1000:],np.repeat(s['Y42'],5,axis=0)))
        checks[name+'_previous_actual_action_in_X238']=bool(np.array_equal(s['X238'][:,195:237],np.vstack([np.zeros((1,42)),s['Y42'][:-1]])))
        checks[name+'_every_sample_array_has4800rows']=all(len(v)==4800 for v in s.values())
        checks[name+'_trace_has_expected_full_body_shape']=a['qpos'].shape==(25000,49) and a['qvel'].shape==(25000,48) and a['action'].shape==(25000,42)
        if name=='halfspeed_smallleft':
            olda=load(original/'TRACE_1MS.npz');olds=load(original/'SAMPLES_5MS.npz')
            checks['smallleft_first15p4s_original_half_failure_trace_bitexact']=all(np.array_equal(a[k][:len(v)],v) for k,v in olda.items())
            checks['smallleft_first2880_teacher_samples_bitexact']=all(np.array_equal(s[k][:2880],olds[k]) for k in ('X238','Y42','actual_time','source_phase','post_prime'))
    spec=importlib.util.spec_from_file_location('_deceleration_curriculum',HERE/'curriculum.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    verification=json.loads((HERE/'VERIFICATION.json').read_text())
    for name in module.CASES:
        admitted=True
        try: samples=module.load_verified_samples(name)
        except ValueError: admitted=False
        expected=verification['all_integrity_checks_passed'] and verification['recomputed'][name]['all_halfspeed']
        checks[name+'_curriculum_admission_matches_independent_gates']=admitted==expected
        if admitted:
            checks[name+'_curriculum_returns4800and3840']=samples['X238'].shape==(4800,238) and module.load_verified_samples(name,post_only=True)['X238'].shape==(3840,238)
    result=dict(all_passed=all(checks.values()),checks=checks,
        known_runner_failure_path_limitation='Frozen run.py does not persist accumulated in-memory arrays after arbitrary exceptions; if encountered, no retry/substitution permitted and incomplete evidence must be reported. This limitation does not affect normally completed trajectories.',
        curriculum='Preferred loader curriculum.py authenticates independently recomputed success and RESULT/SAMPLES hashes; frozen api.py weaker loader is not parent admission API.')
    (HERE/'ALIGNMENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['all_passed']
if __name__=='__main__':main()
