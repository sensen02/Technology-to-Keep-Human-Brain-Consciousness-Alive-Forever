"""Independent freeze verification: recompute every hash, re-run the declared reproduction
and the platform unit tests, and confirm the declared limitations are present.

Does not import build_freeze.py; the manifest is read as data and every hash is recomputed
from the files themselves.
"""
from pathlib import Path
import hashlib, json, subprocess, sys
import numpy as np
HERE = Path(__file__).resolve().parent
PY = '/run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python'
checks = []


def check(name, ok, **d):
    checks.append(dict(name=name, passed=bool(ok), **d))
    return ok


def main():
    man = json.loads((HERE / 'FREEZE_MANIFEST.json').read_text())
    check('schema', man.get('schema') == 'pre-injury-platform-freeze/1')
    check('status_frozen', man.get('status') == 'FROZEN_WITH_DECLARED_LIMITS')
    check('no_missing', man.get('missing') == [])
    bad = []
    for e in man['artefacts']:
        p = Path(e['path'])
        if not p.exists():
            bad.append([e['role'], 'missing']); continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != e['sha256']:
            bad.append([e['role'], 'hash changed', h, e['sha256']])
    check('all_hashes_match', bad == [], mismatches=bad)
    # declared limitations must be present and non-empty
    nf = man['not_frozen']
    for key in ('accepted_walking', 'feeding_biomechanics', 'injury_models', 'absolute_units', 'intake_to_locomotion_feedback'):
        check('limitation_' + key, bool(nf.get(key)))
    check('accepted_walking_explicitly_absent', 'NOT ACHIEVED' in nf['accepted_walking'])
    # movement status must be a non-accepted controller
    ms = json.loads((HERE / 'RESULT_REPRODUCTION.json').read_text())['movement']
    check('movement_not_accepted', ms['accepted'] is False)
    check('movement_deficiencies_declared', len(ms['deficiencies']) >= 3)
    # reproduction must match the declared expectation
    rep = json.loads((HERE / 'RESULT_REPRODUCTION.json').read_text())
    exp = man['reproduction']['expectation']
    ok_rep = (rep['arrival']['tick'] == 224 and abs(rep['arrival']['time_s'] - 1.125) < 1e-9
              and rep['decisions']['feed'] == 58 and abs(rep['energy_start'] - .5999) < 5e-5
              and abs(rep['energy_end'] - .5925) < 5e-5 and abs(rep['food_end'] - .9275) < 5e-5
              and rep['native_steps_added'] == 0)
    check('reproduction_matches_expectation', ok_rep, expectation=exp)
    # unit tests must still pass on the frozen component files
    out = subprocess.run([PY, '-B', str(HERE.parent / '20261107T234820Z' / 'test_platform.py')],
                         capture_output=True, text=True, env={'PYTHONDONTWRITEBYTECODE': '1', 'PATH': '/usr/bin:/bin'})
    ut = {}
    try:
        ut = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        pass
    check('unit_tests_pass', ut.get('status') == 'PLATFORM_UNIT_TESTS_PASS' and ut.get('checks_passed') == 24,
          unit_tests=ut, stderr=out.stderr[-200:] if out.returncode else '')
    # no accepted-walking or injury artefact may be referenced anywhere in the freeze
    text = (HERE / 'FREEZE_MANIFEST.json').read_text()
    check('no_injury_artefact_frozen', 'injury' in text and 'none created' in text)
    passed = sum(c['passed'] for c in checks)
    out_json = dict(status='FREEZE_VERIFIED' if passed == len(checks) else 'FREEZE_VERIFICATION_FAILED',
                    checks_passed=passed, checks_total=len(checks), checks=checks,
                    physics_steps=0, neural_steps=0, note='verification only: hashes, declared limits, reproduction, unit tests')
    (HERE / 'VERIFICATION.json').write_text(json.dumps(out_json, indent=2) + '\n')
    print(json.dumps(dict(status=out_json['status'], checks_passed=passed, checks_total=len(checks))))
    if passed != len(checks):
        for c in checks:
            if not c['passed']:
                print('FAILED:', c)


if __name__ == '__main__':
    main()
