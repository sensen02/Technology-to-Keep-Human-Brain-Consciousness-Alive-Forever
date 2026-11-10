"""Deterministic numerical verification only: no neural or physical execution."""
from pathlib import Path
import copy
import csv
import inspect
import json
import sys
import numpy as np
from encoder import SensoryEncoder, SOURCE, MAPPING, CLASSES, DISABLED, NERVES, qualify_rows, sha256

OUT = Path(__file__).resolve().parent


def main():
    prereg = OUT / 'PREREG.json'
    assert prereg.exists(), 'preregistration must precede tests'
    checks = []
    def check(name, condition, **details):
        if not condition:
            raise AssertionError(name)
        checks.append(dict(name=name, passed=True, **details))
    e = SensoryEncoder()
    contract = json.loads((OUT / 'CONTRACT.json').read_text())
    check('saved_contract_matches_runtime', contract['qualified_records'] == e.qualified_records and contract['source_contract'] == e.source_contract and contract['runtime_provenance']['encoder_sha256'] == sha256(OUT / 'encoder.py'))
    check('only_explicit_input_arguments', tuple(inspect.signature(e.encode).parameters) == ('q42_rad', 'qvel42_rad_s', 'foot_load6_native'))
    check('no_neural_simulator_imports', not any(x == 'mujoco' or x.startswith('engine.') or x in ('torch', 'tensorflow') for x in sys.modules))
    with (MAPPING / 'current42_candidates.csv').open() as f:
        joint_rows = list(csv.DictReader(f))
    source_names = [r['current_joint'] for r in joint_rows]
    bounds = np.array([json.loads(r['range_rad']) for r in joint_rows])
    check('all42_source_names_and_bounds', tuple(source_names) == e.joint_names and np.array_equal(bounds, e.joint_ranges_rad))
    check('actual_RF_coxa_order', e.joint_names[21:24] == ('joint_RFCoxa_roll','joint_RFCoxa_yaw','joint_RFCoxa_pitch') and e.coxa_indices[3].tolist() == [22,23,21])
    check('explicit_foot_order', e.foot_order == ('LF','LM','LH','RF','RM','RH'))
    with np.load(SOURCE / 'LOCAL_MODULE_CANDIDATE.npz', allow_pickle=False) as a:
        roots, cache_indices = a['root_ids'], a['cache_indices']
    with np.load(SOURCE / 'GROUP_CACHE_INDICES.npz', allow_pickle=False) as a:
        groups = {k:a[k] for k in a.files}
    with np.load(e.source_contract['cache']['path'], allow_pickle=False) as a:
        cache = {k:a[k] for k in ('root_ids','ann_class','ann_super_class','ann_nerve')}
    check('all20000_exact_root_identity', np.array_equal(roots, e.root_ids) and np.array_equal(cache['root_ids'][cache_indices], roots))
    check('ports_unique_and_source_qualified', len(set(e.local_indices.tolist())) == e.qualified_count and set(e.local_indices) == {r['local_index_0based'] for r in e.qualified_records})
    for r in e.qualified_records:
        ci, li = r['cache_index_0based'], r['local_index_0based']
        assert int(r['root_888']) == roots[li] == cache['root_ids'][ci]
        assert ci in groups['leg_sensory_nerve_union']
        assert ci in groups['leg_sensory_' + r['leg'].lower()]
        nervelegs = [leg for leg, nerve in NERVES.items() if nerve in str(cache['ann_nerve'][ci]).split(',')]
        assert nervelegs == [r['leg']]
        assert r['cache_class'] == CLASSES[r['channel']]
        assert r['channel'] not in DISABLED
    check('every_qualified_source_identity_class_nerve_and_group', True, count=e.qualified_count)
    lo, hi = bounds.T
    mid = (lo + hi) / 2
    z42, z6 = np.zeros(42), np.zeros(6)
    zero = e.encode(z42,z42,z6)
    baseline = e.encode(mid,z42,z6)
    check('zero_inputs_finite_and_port_confined', zero.shape == (20000,) and np.isfinite(zero).all() and np.all(zero >= 0) and np.all(zero <= .06) and np.all(zero[~e.allowed_mask] == 0), nonzero_cells=int(np.count_nonzero(zero)))
    check('zero_is_coordinate_diagnostic_not_valid_body_pose', bool(np.any((z42 < lo) | (z42 > hi))))
    angle_ports = sorted(r['local_index_0based'] for r in e.qualified_records if r['channel'] in ('claw_position','hairplate_angle'))
    check('baseline_only_qualified_angle_ports', np.flatnonzero(baseline).tolist() == angle_ports and np.allclose(baseline[angle_ports], .03, atol=1e-16, rtol=0))
    check('zero_and_midpose_differ', not np.array_equal(zero,baseline))
    def ports(leg, ch):
        return set(e.channel_local_indices[leg][ch].tolist())
    # Independently derive each feature's input coordinate from literal joint names.
    isolation = []
    for ji,name in enumerate(source_names):
        q=mid.copy(); q[ji]=hi[ji]
        got=e.encode(q,z42,z6)
        leg=name[6:8]
        expected=ports(leg,'claw_position') if 'Tibia_pitch' in name else ports(leg,'hairplate_angle') if 'Coxa_' in name else set()
        changed=set(np.flatnonzero(got != baseline).tolist())
        check('pose_coordinate_'+name, changed == expected)
        if expected:
            delta=.02 if 'Tibia_pitch' in name else .02/3
            check('pose_gain_'+name, np.allclose(got[list(expected)]-baseline[list(expected)],delta,rtol=0,atol=2e-17))
        v=z42.copy(); v[ji]=10
        got=e.encode(mid,v,z6)
        expected_v=ports(leg,'hook_direction') if 'Tibia_pitch' in name else set()
        check('velocity_coordinate_'+name,set(np.flatnonzero(got != baseline).tolist()) == expected_v)
        check('velocity_sign_not_fabricated_'+name,np.array_equal(got,e.encode(mid,-v,z6)))
        isolation.append(dict(joint=name,pose_changed=len(changed),speed_changed=len(expected_v)))
    for fi,leg in enumerate(e.foot_order):
        load=z6.copy();load[fi]=24.446354755878005/6
        got=e.encode(mid,z42,load)
        expected=ports(leg,'campaniform_strain')
        check('load_foot_'+leg,set(np.flatnonzero(got != baseline).tolist()) == expected and np.allclose(got[list(expected)],.05*np.tanh(1),rtol=0,atol=1e-17))
    check('range_normalization_low',np.array_equal(e.normalize_angles(lo),z42))
    check('range_normalization_high',np.array_equal(e.normalize_angles(hi),np.ones(42)))
    check('range_normalization_mid',np.allclose(e.normalize_angles(mid),.5,rtol=0,atol=4*np.finfo(float).eps), max_error=float(np.max(np.abs(e.normalize_angles(mid)-.5))))
    check('range_normalization_clamping',np.array_equal(e.normalize_angles(lo-100),z42) and np.array_equal(e.normalize_angles(hi+100),np.ones(42)))
    check('negative_load_clamps_zero',np.array_equal(baseline,e.encode(mid,z42,-np.ones(6)*1e300)))
    extreme = np.finfo(np.float64).max
    with np.errstate(all='raise'):
        for sign in (-1,1):
            current=e.encode(np.full(42,sign*extreme),np.full(42,sign*extreme),np.full(6,sign*extreme))
            assert np.isfinite(current).all() and np.all(current >= 0) and np.all(current <= .06)
            assert np.all(current[~e.allowed_mask] == 0)
    check('extreme_finite_values_bounded_without_overflow',True)
    # No random data, history or phase: fixed deterministic arithmetic stress sweep.
    for k in range(31):
        q=lo+(hi-lo)*(k/15-0.5)
        v=np.arange(42,dtype=float)*(k-15)
        load=np.arange(6,dtype=float)*(k-8)
        a=e.encode(q,v,load)
        b=SensoryEncoder().encode(q,v,load) if k in (0,15,30) else e.encode(q,v,load)
        assert np.array_equal(a,b)
        assert not np.shares_memory(a,b)
        assert np.isfinite(a).all() and np.all((a>=0)&(a<=.06)) and np.all(a[~e.allowed_mask]==0)
        for leg in e.foot_order:
            for ch in CLASSES:
                pp=e.channel_local_indices[leg][ch]
                assert len(pp)==0 or np.all(a[pp]==a[pp[0]])
    check('31_deterministic_sweeps_and_identical_within_class_tuning',True)
    invalid_count=0
    for arg,n in enumerate((42,42,6)):
        for bad in (np.zeros(n+1),np.zeros((1,n)),np.array(0.),np.full(n,np.nan),np.full(n,np.inf),np.full(n,-np.inf),np.full(n,1+2j),np.full(n,'1')):
            inputs=[mid,z42,z6]; inputs[arg]=bad
            try:
                e.encode(*inputs)
            except ValueError:
                invalid_count+=1
            else:
                raise AssertionError('invalid input accepted')
    check('invalid_shapes_nonfinite_and_nonreal_rejected',invalid_count==24,count=invalid_count)
    review=json.loads((SOURCE/'SENSORY_CHANNEL_REVIEW.json').read_text())
    ambiguous={r['root_id_string'] for r in review['ambiguous_records']}
    excluded_by_root={r['root_888']:r for r in e.exclusions}
    check('review4_saved_as_explicit_exclusions',all('review4_modality_ambiguity' in excluded_by_root[root]['reasons'] for root in ambiguous))
    excluded_local={r['local_index_0based'] for r in e.exclusions if r['local_index_0based'] is not None}
    check('all_excluded_local_cells_remain_zero',not (excluded_local & set(e.local_indices.tolist())) and np.all(current[list(excluded_local)] == 0))
    # Qualification mutation fixtures are derived from real admitted records.
    # They never enter CONTRACT or encoder output; no synthetic cells are created.
    base={k:v for k,v in e.qualified_records[0].items() if k not in ('local_index_0based','cache_nerve','reasons')}
    base['leg']=base['leg'].lower()
    def rejected(name, altered_rows, expected, altered_cache=None, altered_groups=None, local_r=roots, local_i=cache_indices, amb=ambiguous):
        admitted,rejections=qualify_rows(altered_rows,altered_cache or cache,altered_groups or groups,local_r,local_i,amb)
        check('qualification_reject_'+name,not admitted and any(expected in x['reasons'] for x in rejections))
    a=copy.deepcopy(base);a['cache_index_0based']=-1
    rejected('missing_cache',[a],'missing_cache_identity')
    rejected('review_ambiguity',[base],'review4_modality_ambiguity',amb=ambiguous|{base['root_888']})
    rejected('duplicate_reused_identity',[base,copy.deepcopy(base)],'reused_source_identity_or_cache_index')
    a=copy.deepcopy(base);a['leg']='rh' if a['leg']!='rh' else 'lf'
    rejected('source_leg_contradiction',[a],'cache_nerve_source_leg_mismatch_or_ambiguity')
    a=copy.deepcopy(base);a['cache_class']='motor_neuron'
    rejected('class_contradiction',[a],'channel_source_class_contradiction')
    a=copy.deepcopy(base);a['root_888']=str(int(a['root_888'])+1)
    rejected('root888_mismatch',[a],'root888_cache_identity_mismatch')
    ci=int(base['cache_index_0based'])
    keep=cache_indices!=ci
    rejected('outside_candidate',[base],'outside_local_candidate',local_r=roots[keep],local_i=cache_indices[keep])
    g=dict(groups);g['leg_sensory_nerve_union']=groups['leg_sensory_nerve_union'][groups['leg_sensory_nerve_union']!=ci]
    rejected('outside_union',[base],'outside_leg_sensory_nerve_union',altered_groups=g)
    c={k:v.copy() for k,v in cache.items()};c['ann_nerve'][ci]=NERVES[base['leg'].upper()]+','+NERVES['RH' if base['leg']!='rh' else 'LF']
    rejected('multiple_leg_nerves',[base],'cache_nerve_source_leg_mismatch_or_ambiguity',altered_cache=c)
    check('source_manifest_hashes_still_match',all(sha256(item['path']) == item['sha256'] for item in e.source_contract.values()))
    result={
        'status':'PASS_NUMERICAL_ENGINEERING_ONLY', 'checks_passed':len(checks), 'checks':checks,
        'qualified_count':e.qualified_count, 'source_rows':len(e.qualified_records)+len(e.exclusions),
        'excluded_rows':len(e.exclusions), 'zero_input_nonzero_cells':int(np.count_nonzero(zero)),
        'midpose_nonzero_cells':int(np.count_nonzero(baseline)), 'joint_isolation':isolation,
        'channel_leg_counts':contract['channel_leg_counts'], 'exclusion_reason_counts':contract['exclusion_reason_counts'],
        'prereg_sha256':sha256(prereg), 'encoder_sha256':sha256(OUT/'encoder.py'), 'test_sha256':sha256(__file__),
        'runtime':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__},
        'neural_steps':0,'physical_steps':0,'mujoco_forward_calls':0,'training_updates':0,
        'limitations':'Static/numerical engineering contract only, no biological calibration or neural/physical behavior claim.'}
    (OUT/'VERIFICATION.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_passed','qualified_count','excluded_rows','exclusion_reason_counts')}))


if __name__=='__main__':
    main()
