"""Freeze addendum: measured movement deficiency + hash chain to the round-15 freeze.

Independent checks only: the previous freeze manifest and its verification are re-hashed,
the quoted verifier numbers are re-read from the verifier file and compared with the
transcribed constants, and the deficiency arithmetic is recomputed.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
HERE = Path(__file__).resolve().parent
PREV = HERE.parent / '20261108T001200Z'
VERIFIER = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/evaluation/ann_residual_u32/VERIFICATION.json')
checks = []


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(name, ok, **d):
    checks.append(dict(name=name, passed=bool(ok), **d)); return ok


def main():
    spec = importlib.util.spec_from_file_location('md', HERE / 'movement_deficiency.py')
    md = importlib.util.module_from_spec(spec); sys.modules['md'] = md; spec.loader.exec_module(md)
    prev_manifest = PREV / 'FREEZE_MANIFEST.json'
    prev_verif = PREV / 'VERIFICATION.json'
    check('prev_freeze_exists', prev_manifest.exists() and prev_verif.exists(),
          prev_freeze_sha256=sha(prev_manifest), prev_verification_sha256=sha(prev_verif))
    man = json.loads(prev_manifest.read_text())
    # re-hash every frozen artefact from the previous manifest
    bad = []
    for e in man['artefacts']:
        p = Path(e['path'])
        if not p.exists():
            bad.append([e['role'], 'missing']); continue
        if sha(p) != e['sha256']:
            bad.append([e['role'], 'changed'])
    check('frozen_artefacts_unchanged', bad == [], mismatches=bad)
    # quoted verifier numbers must match the verifier file itself
    v = json.loads(VERIFIER.read_text())
    check('verifier_hash_matches_quote', sha(VERIFIER) == md.SOURCE_SHA256, quoted=md.SOURCE_SHA256, actual=sha(VERIFIER))
    mc = v['physics_replay']['material_contact_velocity']
    check('mean_speed_transcription', list(np.round(mc['force_weighted_mean_tangential_speed'], 12)) == list(np.round(md.MEAN_TANGENTIAL_SPEED, 12)))
    check('rms_speed_transcription', list(np.round(mc['force_weighted_rms_tangential_speed'], 12)) == list(np.round(md.RMS_TANGENTIAL_SPEED, 12)))
    check('slip_ratio_transcription', list(np.round(mc['slip_to_forward_speed_ratio'], 12)) == list(np.round(md.SLIP_RATIO, 12)))
    check('loaded_samples_transcription', list(mc['loaded_samples']) == list(md.LOADED_SAMPLES))
    check('verifier_reported_failure', v['material_slip_passed'] is False and v['max_material_slip_to_forward_speed_ratio'] == md.THRESHOLD)
    check('ratio_consistent_with_speed', np.allclose(np.asarray(mc['force_weighted_mean_tangential_speed']) / v['gait']['locomotion']['cycle_mean_forward_speed'],
                                                    np.asarray(mc['slip_to_forward_speed_ratio']), rtol=1e-9))
    s = md.summary()
    check('failing_count_is_five', s['failing_count'] == 5 and s['passing_count'] == 1, failing=s['legs_failing'], passing=s['legs_passing'])
    check('max_ratio_above_threshold', s['max_ratio'] > md.THRESHOLD and s['max_ratio'] < 1.05)
    check('accepted_is_false', s['accepted'] is False)
    addendum = dict(schema='pre-injury-platform-freeze-addendum/1', status='ADDENDUM_VERIFIED',
                    parent_freeze=dict(path=str(prev_manifest), sha256=sha(prev_manifest),
                                       verification=str(prev_verif), verification_sha256=sha(prev_verif),
                                       artefacts=len(man['artefacts']), artefacts_unchanged=bad == []),
                    movement_deficiency=s,
                    consequence=('the frozen platform may be exercised for state/feeding/selector logic only; it may not be '
                                 'reported as accepted walking, and no injury experiment may use it as a healthy baseline'),
                    checks_passed=sum(c['passed'] for c in checks), checks_total=len(checks), checks=checks,
                    physics_steps=0, neural_steps=0)
    (HERE / 'FREEZE_ADDENDUM.json').write_text(json.dumps(addendum, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=addendum['status'] if addendum['checks_passed'] == addendum['checks_total'] else 'ADDENDUM_FAILED',
                          checks_passed=addendum['checks_passed'], checks_total=addendum['checks_total'],
                          failing=len(s['legs_failing']), max_ratio=round(s['max_ratio'], 4),
                          mean_ratio=round(s['mean_ratio'], 4))))
    for c in checks:
        if not c['passed']:
            print('FAILED:', c)


if __name__ == '__main__':
    main()
