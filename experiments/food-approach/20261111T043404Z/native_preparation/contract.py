"""Fail-closed admission and identities; imports only the Python standard library."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python')
BASE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z')
BANC = BASE.parent
OWN_SOURCE_NAMES = ('contract.py','execute_native.py','pipeline.py','verify_native.py','render_selected.py','check_prepared.py','METHODS_DRAFT.json')
PINS = {
    BANC/'20261110T054510Z/native_acceptance.py': '379707f72a231e5e2ff6c5c827971cb29572a65cf3cccc5c7cd63df707b83f8c',
    BANC/'20261110T152725Z/final_native_audit.py': '363e37eea682d3d0ead64fa7edc8d1d3066d0243e4d8b2cac47355800123141c',
    BASE/'physics.py': 'a5effcae6c5e369852b1bb08ec3556f87a2d3c62e468e931a57350a6f3479530',
    ROOT.parent/'20261110T235830Z/evaluation/verify_actual.py': '1293cf1db7b61a90eeb13500a18eb53796b4ff8891a38502dbfac3f94e5e7245',
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def save_new(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(HERE), 'output outside native preparation scope')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

def pinned():
    for path, digest in PINS.items():
        require(sha(path) == digest, 'frozen native method/source changed: '+str(path))

def identity(document):
    payload = {key:value for key,value in document.items() if key != 'identity_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == document['identity_sha256'], 'execution identity self hash mismatch')
    for name, item in document['files'].items():
        require(Path(name).is_absolute() and sha(name) == item['sha256'], 'execution dependency drift: '+name)
    require(document['physics'] == dict(nq=49,nv=48,nu=42,timestep=12.5e-6,gravity=[0.,0.,-9801.]), 'body physics changed')
    require(document['adhesion']['enabled'] is False, 'adhesion enabled')

def qualification(summary_path, checkpoint, digest):
    """Inspect actual CEM summary, immutable selection and every matched raw case."""
    summary_path = Path(summary_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    require(summary_path == ROOT/'FINAL_SUMMARY.json', 'exact parent final summary required')
    require(checkpoint.is_relative_to(ROOT) and checkpoint.name == 'candidate.pt', 'parent candidate required')
    require(sha(checkpoint) == digest, 'candidate SHA mismatch')
    s = load(summary_path)
    require(s['status'] == 'COMPLETE' and s['selected_all_three_pass'] is True, 'same-model full three engineering cases not passed')
    require(s['selected'] == checkpoint.parent.name and Path(s['selected_checkpoint']).resolve() == checkpoint and s['selected_checkpoint_sha256'] == digest, 'summary selected checkpoint mismatch')
    selection_path = Path(s['selected_proof']).resolve()
    require(selection_path == ROOT/'SELECTED.json', 'immutable selection proof path differs')
    selection = load(selection_path)
    require(selection['checkpoint_sha256'] == digest and Path(selection['checkpoint']).resolve() == checkpoint and selection['candidate'] == s['selected'], 'selection proof model mismatch')
    require(selection['engineering_pass'] is True and selection['integrity_all_pass'] is True, 'selection proof not passed')
    for prefix in ('proposal','frozen_candidates'):
        require(sha(selection[prefix+'_path']) == selection[prefix+'_sha256'], 'immutable optimization proof changed: '+prefix)
    cases = [s['selected_left']] + list(s['confirmations'])
    require(len(cases) == 3 and {r['condition'] for r in cases} == {'neutral','left','right'}, 'three distinct matched conditions required')
    bound = {str(summary_path):sha(summary_path),str(selection_path):sha(selection_path),str(checkpoint):digest}
    normalized = {}
    for row in cases:
        condition = row['condition']
        require(row['engineering_pass'] is True and row['integrity_all_pass'] is True and row['sha256'] == digest and Path(row['checkpoint']).resolve() == checkpoint, 'unqualified case/model association')
        rp, ap = Path(row['result']).resolve(), Path(row['audit']).resolve()
        require(rp.is_relative_to(ROOT) and ap.is_relative_to(ROOT), 'qualification artifact outside experiment')
        require(sha(rp) == row['result_sha256'] and sha(ap) == row['audit_sha256'], 'case proof hash changed')
        result, audit = load(rp), load(ap)
        require(result['engineering_pass'] is True and audit['integrity_all_pass'] is True and audit['engineering_pass'] is True, 'raw case/audit not fully passed')
        require(result['condition'] == condition and result['checkpoint_sha256'] == digest, 'raw case identity mismatch')
        require(all(result['autonomous']['gates'].values()) and result['autonomous_cycles']['all_legs_completed'], 'original gates or six cycles not passed')
        require(result['autonomous']['nonfoot_force_max_native'] == 0., 'strict nonfoot-zero missing')
        if condition != 'neutral':
            require(result['yaw_change_rad']*(1 if condition == 'left' else -1) >= .01, 'original signed yaw gate failed')
        ident_path = rp.parent/'IDENTITY.json'
        ident = load(ident_path); identity(ident)
        require(ident['identity_sha256'] == result['identity_sha256'], 'case/result runtime identity mismatch')
        for key, name in [('trace','trace_1ms.npz'),('neural_decisions','neural_decisions.npz'),('geometry','geometry_1ms.npz'),('runtime_geometry','runtime_geometry.npz')]:
            target = rp.parent/name
            require(sha(target) == result[key]['sha256'], 'case data changed: '+name)
            bound[str(target)] = sha(target)
        bound.update({str(rp):sha(rp),str(ap):sha(ap),str(ident_path):sha(ident_path)})
        normalized[condition] = dict(row)
    require(selection['left_result_sha256'] == normalized['left']['result_sha256'] and selection['left_audit_sha256'] == normalized['left']['audit_sha256'], 'selection left evidence differs')
    return dict(summary=s, selection=selection, cases=normalized, hashes=bound)

def admitted(plan_path, checkpoint, digest):
    plan_path = Path(plan_path).resolve()
    require(plan_path == HERE/'RUN_PLAN.json', 'explicit scoped RUN_PLAN.json required')
    p = load(plan_path)
    require(p['status'] == 'REGISTERED_SELECTED_LEFT_NATIVE', 'no actual native registration')
    require(p['native_runs'] == 1 and p['condition'] == 'left' and p['native_samples'] == 1232000, 'only one full selected LEFT native run permitted')
    require(Path(p['checkpoint']).resolve() == Path(checkpoint).resolve() and p['checkpoint_sha256'] == digest, 'plan selected SHA differs')
    require(p['checkpoint_sha256'] == p['engineering_checkpoint_sha256'], 'engineering/native model differs')
    require(sha(p['engineering_summary']) == p['engineering_summary_sha256'], 'engineering summary drift')
    require(sha(HERE/'METHODS_DRAFT.json') == p['methods_sha256'], 'methods changed after registration')
    require({str(HERE/name) for name in OWN_SOURCE_NAMES}.issubset(p['preparation_source_hashes']), 'complete native method/entry/source hash closure required before launch')
    for name, expected in p['preparation_source_hashes'].items():
        require(Path(name).name not in ('CURRENT_STATE.json','LIVE_PROGRESS.json'), 'mutable pointer must not enter immutable native source seal')
        require(Path(name).resolve().is_relative_to(HERE) and sha(name) == expected, 'native preparation source drift')
    q = qualification(p['engineering_summary'], checkpoint, digest)
    require(sha(ROOT/'SELECTED.json') == p['selection_sha256'], 'selection certificate drift')
    pinned()
    return p, q
