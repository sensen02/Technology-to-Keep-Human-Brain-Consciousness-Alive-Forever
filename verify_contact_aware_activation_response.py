"""S130 independent OFFLINE evidence verifier; no simulator imports or calls.

Only saved bytes are read. This does not independently rerun the contact solver,
certify a trajectory gradient, produce candidate weights or change a checkpoint.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/sensen/Desktop/engine修复验收')
REFERENCE = BASE/'s121-constraint-row-recording/20261024T072500Z/init80ms'
S124 = BASE/'s124-muscle-reconstructability/20261024T075500Z'
S125 = BASE/'s125-local-translation-candidate/20261024T080500Z'
STATES = [77, 100, 105]
ACTUATORS = [0, 1, 2, 4, 5]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def manifest(path):
    data = js(path)
    if 'before' in data:
        assert data['before'] == data['after'], str(path)+' frozen input mutation'
        data = data['before']
    assert isinstance(data, dict) and data, str(path)+' empty manifest'
    for name, digest in data.items():
        assert Path(name).is_absolute(), 'Manifest requires absolute paths: '+name
        assert isinstance(digest, str) and len(digest) == 64
        assert sha(name) == digest, 'Hash mismatch: '+name
    return {str(Path(name).resolve()): digest for name, digest in data.items()}


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, passed, **details):
        self.rows.append(dict(name=name, passed=bool(passed), **details))
        return bool(passed)

    def equal(self, name, actual, expected, atol=1e-10, rtol=1e-10):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            return self.add(name, False, actual_shape=list(a.shape), expected_shape=list(b.shape))
        finite = np.isfinite(a).all() and np.isfinite(b).all()
        residual = float(np.max(np.abs(a-b), initial=0))
        return self.add(name, finite and np.all(np.abs(a-b) <= atol+rtol*np.abs(b)),
                        max_abs=residual, atol=atol, rtol=rtol)

    def exact(self, name, a, b):
        return self.add(name, np.array_equal(a, b))


def independent_gain(length, velocity, prm):
    """Literal scalar FLV law, independent of the producer's muscle helper."""
    g = [float(v) for v in prm['gainprm'].split()]
    lo, hi = [float(v) for v in prm['lengthrange'].split()]
    assert g[2] > 0, 'Negative force scale needs unavailable acc0; do not infer it'
    scale = (hi-lo)/max(1e-15, g[1]-g[0])
    length = g[0]+(length-lo)/max(1e-15, scale)
    velocity /= max(1e-15, scale*g[6])
    left, right = (g[4]+1)/2, (1+g[5])/2
    if length < g[4] or length > g[5]:
        fl = 0.
    elif length <= left:
        fl = .5*((length-g[4])/max(1e-15, left-g[4]))**2
    elif length <= 1:
        fl = 1-.5*((1-length)/max(1e-15, 1-left))**2
    elif length <= right:
        fl = 1-.5*((length-1)/max(1e-15, right-1))**2
    else:
        fl = .5*((g[5]-length)/max(1e-15, g[5]-right))**2
    y = g[8]-1
    if velocity <= -1:
        fv = 0.
    elif velocity <= 0:
        fv = (1+velocity)**2
    elif velocity <= y:
        fv = g[8]-(y-velocity)**2/max(1e-15, y)
    else:
        fv = g[8]
    return -g[2]*fl*fv


def activation_tangent(reference, initialization, parameters):
    """Rebuild frozen historical FIR/clip and 25us RK4 tangent, no producer imports."""
    W, pool = reference['weights'], reference['actuator_pool']
    features = np.concatenate((initialization['features'], reference['features']))
    assert features.shape == (166, 66, 5)
    source = js(REFERENCE.parent/'PREREG.json')['source_hashes']
    manifests = [p for p in source if p.endswith('/MANIFEST.json') and 's4-endpoints' in p]
    assert len(manifests) == 1
    assert sha(manifests[0]) == source[manifests[0]]
    groups = [tuple(sorted(row['local_indices'])) for row in js(manifests[0])['rows']]
    pools = sorted(set(g for g in groups if g))
    lookup = {g: j for j, g in enumerate(pools)}
    assert np.array_equal(pool, [lookup.get(g, -1) for g in groups])
    history = np.zeros((80, 66)); reconstructed = []
    for spikes in np.concatenate((initialization['spikes'], reference['spikes'])):
        rates = np.array([np.mean(spikes[list(g)])/.0005/100 for g in pools])
        history = np.vstack((rates, history))[:80]
        reconstructed.append(np.stack([history[lo:hi].mean(0) for lo, hi in ((0,5),(5,10),(10,20),(20,40),(40,80))], axis=1))
    assert np.max(np.abs(np.array(reconstructed)-features)) <= 1e-12
    assert np.array_equal(pool[:6], [38,38,36,-1,15,3])
    assert np.array_equal(np.flatnonzero(pool == 38), [0,1])
    columns = [(p, b) for p in (3, 15, 38) for b in range(5) if W[p, b] > 0]
    assert len(columns) == 15
    raw = np.sum(features*W, axis=2)
    ctrl = np.zeros((166, 90)); dc = np.zeros((166, 90, 15))
    for a, p in enumerate(pool):
        if p < 0:
            continue
        ctrl[:, a] = np.clip(raw[:, p], 0, 1)
        branch = (raw[:, p] > 0) & (raw[:, p] < 1)
        for j, (pp, b) in enumerate(columns):
            if p == pp:
                dc[:, a, j] = features[:, p, b]*branch
    tau = np.zeros((90, 10))
    for a, node in enumerate(parameters):
        values = list(map(float, node['resolved']['dynprm'].split()))
        tau[a, :len(values)] = values
    assert np.all(tau[:, 2] == 0), 'This independent hard-switch tangent requires dynprm width zero'
    a = initialization['body_act'].copy(); tangent = np.zeros((90, 15))
    states = [a.copy()]; derivatives = [tangent.copy()]; dt = .000025

    def stage(x, D, u, U):
        v = .5+1.5*np.clip(x, 0, 1)
        delta = np.clip(u, 0, 1)-x
        rising = delta > 0
        t = np.where(rising, tau[:, 0]*v, tau[:, 1]/v)
        tp = np.where(rising, 1.5*tau[:, 0], -1.5*tau[:, 1]/v**2)
        tp *= (x > 0) & (x < 1)
        f = delta/t
        return f, (-1/t-delta*tp/t**2)[:, None]*D+U/t[:, None]

    for step in range(120):
        tick = 160+step//20
        u, U = ctrl[tick], dc[tick]
        f1, d1 = stage(a, tangent, u, U)
        f2, d2 = stage(a+dt*f1/2, tangent+dt*d1/2, u, U)
        f3, d3 = stage(a+dt*f2/2, tangent+dt*d2/2, u, U)
        f4, d4 = stage(a+dt*f3, tangent+dt*d3, u, U)
        a += dt*(f1/6+f2/3+f3/3+f4/6)
        tangent += dt*(d1/6+d2/3+d3/3+d4/6)
        states.append(a.copy()); derivatives.append(tangent.copy())
    return np.array(columns), ctrl, np.array(states), np.array(derivatives)


def contact_load(raw, frame, geoms, metadata):
    """Recompute signed external floor/foot forces; never abs(normal force)."""
    raw, frame, geoms = np.asarray(raw), np.asarray(frame), np.asarray(geoms)
    assert raw.shape == (len(geoms), 6) and frame.shape == (len(geoms), 3, 3)
    force2 = np.einsum('cij,ci->cj', frame, raw[:, :3])
    foot = np.zeros((6, 3)); external = np.zeros(3)
    geom_map = {int(k): v for k, v in metadata['geom_map'].items()}
    bodyid = metadata['geom_bodyid']; floor = set(metadata['floor_geom_ids'])
    assert metadata['foot_order'] == ['lf','lm','lh','rf','rm','rh']
    signs = []
    for c, (g1, g2) in enumerate(geoms.astype(int)):
        fixed1, fixed2 = g1 in floor or bodyid[g1] == 0, g2 in floor or bodyid[g2] == 0
        sign = (1. if fixed1 else -1.) if fixed1 != fixed2 else 0.
        signs.append(sign)
        external += sign*force2[c]
        if fixed1 != fixed2:
            other = g2 if fixed1 else g1
            if other in geom_map:
                foot[metadata['foot_order'].index(geom_map[other])] += sign*force2[c]
    return foot, external, force2, np.array(signs)


OUTPUTS = ['qacc','qfrc_constraint','qfrc_actuator','actuator_force','foot_fz','load_ratio','contact_force','efc_force']
FD_TOL = {k: (1e-6, 1e-5) for k in OUTPUTS}
FD_TOL['qacc'] = (1e-3, 1e-5)


def criterion(a, b, absolute, relative):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False
    residual = np.abs(a-b)
    return bool(np.max(residual, initial=0) <= absolute and
                np.max(residual/np.maximum(1., np.maximum(np.abs(a),np.abs(b))), initial=0) <= relative)


def main(out):
    out = out.resolve(); target = out/'PARENT_INDEPENDENT_CHECK_FINAL.json'
    assert not target.exists(), 'Never overwrite prior independent evidence'
    assert 'mujoco' not in sys.modules
    inputs = manifest(out/'INPUT_HASHES.json'); outputs = manifest(out/'OUTPUT_HASHES.json')
    expected_files = {str(p.resolve()) for p in out.iterdir() if p.is_file() and p.name not in {'OUTPUT_HASHES.json', 'PARENT_INDEPENDENT_CHECK.json', 'PARENT_INDEPENDENT_CHECK_FINAL.json'}}
    assert expected_files == set(outputs), 'Output manifest lacks coverage or contains extraneous paths'
    required = [REFERENCE/'TRAJECTORY.npz', REFERENCE/'CONSTRAINT_ROWS.npz',
                REFERENCE/'INITIALIZATION.npz', REFERENCE/'COM_CONTACT_METADATA.json',
                S125/'LOCAL_CHAIN.npz', S124/'ACTUATOR_XML_RESOLUTION.json']
    for p in required:
        assert inputs.get(str(p.resolve())) == sha(p), 'Required source not committed: '+str(p)
    c = Checks(); pre = js(out/'PREREG.json')
    assert pre['samples'] == STATES and pre['actuators'] == ACTUATORS
    assert pre['h'] == 1e-5 and pre['half_h'] == 5e-6
    assert pre['planned'] == dict(model_load=1,baseline_forward=3,perturb_forward=60,maximum_forward=63,step=0,neural_ticks=0)
    assert pre['baseline_tolerance'] == dict(absolute=1e-8,scaled_relative=1e-6,both_required=True,geometry_absolute=1e-10,geometry_relative=1e-10)
    assert pre['fd_tolerance'] == {k:dict(absolute=v[0],scaled_relative=v[1],both_required=True) for k,v in FD_TOL.items()}
    for key in ('analytic_derivative_tolerance','differential_dynamics_tolerance'):
        assert pre[key] == dict(absolute=1e-6,scaled_relative=1e-5)
    assert pre['force_reconstruction_tolerance'] == dict(absolute=1e-10,scaled_relative=1e-10)
    z = archive(REFERENCE/'TRAJECTORY.npz'); historical = archive(REFERENCE/'CONSTRAINT_ROWS.npz')
    raw = archive(out/'RAW_PROBES.npz'); chain = archive(out/'FIXED_STATE_CHAIN.npz')
    oldchain = archive(S125/'LOCAL_CHAIN.npz'); parameters = js(S124/'ACTUATOR_XML_RESOLUTION.json')['all_actuators']
    columns, ctrl, acts, da = activation_tangent(z, archive(REFERENCE/'INITIALIZATION.npz'), parameters)
    c.equal('independent reference activation', acts, z['act'],1e-12,0)
    c.equal('independent S125 reference activation', acts, oldchain['reference_act'],1e-12,0)
    c.equal('independent command',ctrl[160:],z['applied'],1e-12,0)
    c.equal('independent S125 reference tangent',da,oldchain['activation_derivative'],1e-10,1e-10)
    c.exact('S125 reference weight indices',columns,oldchain['weight_indices'])
    c.exact('S130 reference weight indices',columns,chain['weight_indices'])
    c.exact('reference weights unchanged',oldchain['reference_weights'],z['weights'])
    c.equal('pool36 frozen activation tangent',da[:,2],np.zeros((121,15)),0,0)
    c.equal('unmapped actuator3 tangent',da[:,3],np.zeros((121,15)),0,0)
    c.equal('shared pool38 tangent',da[:,0],da[:,1],1e-12,0)
    c.exact('all unprobed activation tangents zero',np.delete(da,ACTUATORS,axis=1),np.zeros((121,85,15)))
    c.exact('chain samples',chain['samples'],STATES); c.exact('chain actuators',chain['actuators'],ACTUATORS)
    meta = js(REFERENCE/'COM_CONTACT_METADATA.json'); footmap = js(out/'FOOT_MAP.json')
    assert footmap['order'] == meta['foot_order'] == ['lf','lm','lh','rf','rm','rh']
    assert {int(g):meta['foot_order'].index(leg) for g,leg in meta['geom_map'].items()} == {int(g):li for g,li in footmap['geom_leg'].items()}
    assert meta['floor_geom_ids'] == [footmap['floor']]
    weight = float(z['body_mass'][0].sum()*np.linalg.norm(meta['gravity_world']))
    c.equal('weight from original body mass and gravity',footmap['weight_native'],weight,1e-12,1e-12)
    model = js(out/'MODEL_AUDIT.json'); provenance = js(REFERENCE/'MODEL_PROVENANCE.json')
    for phase in ('before','after'): c.add('original full model '+phase,model[phase] == provenance[phase])
    assert not model['warmstart_saved'] and 'qacc_warmstart' not in z
    saved_gates = js(out/'BASELINE_GATES.json'); saved_fd = js(out/'FD_CHECKS.json')
    assert [g['sample'] for g in saved_gates] == STATES
    fd_lookup = {(r['sample'],r['actuator']):r for r in saved_fd}
    assert len(fd_lookup) == len(saved_fd)
    identity = ['nefc','ncon','efc_type','efc_id','efc_state','contact_geom1','contact_geom2','contact_dim','contact_efc_address']
    base_ok = []; verified = []; expected_keys = set()

    def block(prefix):
        result = {k[len(prefix)+1:]:v for k,v in raw.items() if k.startswith(prefix+'_')}
        assert result, 'Missing raw probe '+prefix
        expected_keys.update(prefix+'_'+k for k in result)
        return result

    def raw_checks(label, probe, baseline, mass=None):
        for field,n in [('qacc',48),('qfrc_constraint',48),('qfrc_actuator',48),('qfrc_passive',48),('qfrc_bias',48),('act',90),('actuator_force',90)]:
            c.add(label+' full shape '+field,probe[field].shape == (n,) and np.isfinite(probe[field]).all())
        c.equal(label+' cold warmstart',probe['warmstart_before'],np.zeros(48),0,0)
        J = probe['efc_J_dense']; c.add(label+' full J shape',J.shape == (int(probe['nefc']),48))
        c.equal(label+' J transpose force',J.T@probe['efc_force'],probe['qfrc_constraint'],1e-9,1e-10)
        geoms = np.stack((probe['contact_geom1'],probe['contact_geom2']),axis=1)
        foot, external, force2, signs = contact_load(probe['contact_force'],probe['contact_frame'],geoms,meta)
        c.equal(label+' six signed foot Fz',probe['foot_fz'],foot[:,2],1e-10,1e-10)
        c.equal(label+' load divided by native weight',probe['load_ratio'],foot[:,2].sum()/weight,1e-12,1e-12)
        c.equal(label+' contact frame orthonormal',probe['contact_frame']@np.swapaxes(probe['contact_frame'],-1,-2),np.tile(np.eye(3),(len(geoms),1,1)),1e-10,1e-10)
        c.equal(label+' fixed state moment',probe['actuator_moment'],baseline['actuator_moment'],1e-12,1e-12)
        if mass is not None:
            c.equal(label+' delta dynamics',mass@(probe['qacc']-baseline['qacc']),probe['qfrc_actuator']-baseline['qfrc_actuator']+probe['qfrc_constraint']-baseline['qfrc_constraint'],1e-9,1e-9)
            for field in ('qfrc_passive','qfrc_bias','actuator_length','actuator_velocity','efc_J_dense'):
                c.equal(label+' mechanical state fixed '+field,probe[field],baseline[field],1e-12,1e-12)

    for index,i in enumerate(STATES):
        p = f'sample{i:03d}'; base = block(p+'_baseline'); checks = {}
        for key in ['qacc','qfrc_constraint','qfrc_actuator','qfrc_passive','qfrc_bias','actuator_force','actuator_length','actuator_velocity','actuator_moment']:
            checks[key] = criterion(base[key],z[key][i],1e-8,1e-6)
        checks['warnings'] = np.array_equal(base['warnings'],z['warnings'][i]) and not base['warnings'].any()
        checks['nefc'] = int(base['nefc']) == int(historical['nefc'][i])
        checks['ncon'] = int(base['ncon']) == int(z['contact_ncon'][i])
        for key in ['efc_type','efc_id','contact_geom1','contact_geom2','contact_dim','contact_efc_address']:
            checks[key] = np.array_equal(base[key],historical[p+'_'+key])
        for key in ('contact_pos','contact_frame'):
            checks[key] = criterion(base[key],historical[p+'_'+key],1e-10,1e-10)
        checks['efc_force'] = criterion(base['efc_force'],historical[p+'_efc_force'],1e-8,1e-6)
        checks['efc_J_dense'] = criterion(base['efc_J_dense'],historical[p+'_efc_J_dense'],1e-10,1e-10)
        checks['J_transpose_force'] = criterion(base['efc_J_dense'].T@base['efc_force'],base['qfrc_constraint'],1e-10,1e-10)
        checks['contact_force'] = criterion(base['contact_force'],z['contact_force_raw6'][i,:int(base['ncon'])],1e-8,1e-6)
        checks['foot_fz'] = criterion(base['foot_fz'],z['foot_force'][i,:,2],1e-8,1e-6)
        checks['load_ratio'] = criterion(base['load_ratio'],z['load_ratio'][i],1e-8,1e-6)
        gains = np.array([independent_gain(base['actuator_length'][a],base['actuator_velocity'][a],parameters[a]['resolved']) for a in range(90)])
        bias = np.zeros(90)
        for a,node in enumerate(parameters):
            prm=node['resolved']; b=list(map(float,prm['biasprm'].split())); lo,hi=map(float,prm['lengthrange'].split())
            assert b[2] > 0
            scale=(hi-lo)/max(1e-15,b[1]-b[0]); length=b[0]+(base['actuator_length'][a]-lo)/max(1e-15,scale)
            midpoint=(1+b[5])*.5
            if length <= 1: value=0.
            elif length <= midpoint: value=.5*((length-1)/max(1e-15,midpoint-1))**2
            else: value=.5+(length-midpoint)/max(1e-15,midpoint-1)
            bias[a]=-b[2]*b[7]*value
        forces = gains*base['act']+bias
        checks['analytic_force'] = criterion(forces,base['actuator_force'],1e-10,1e-10)
        ok = all(checks.values()); base_ok.append(ok)
        c.add(p+' baseline gate result',ok == saved_gates[index]['passed'])
        assert set(checks) == set(saved_gates[index]['checks'])
        for key,value in checks.items(): c.add(p+' baseline gate '+key,bool(value) == saved_gates[index]['checks'][key]['passed'])
        raw_checks(p+'_baseline',base,base)
        c.equal(p+' original J',base['efc_J_dense'],historical[p+'_efc_J_dense'],1e-12,1e-12)
        c.exact(p+' original activation',base['act'],z['act'][i])
        if not ok:
            assert not any(k.startswith(p+'_act') or k.startswith(p+'_mechanics') for k in raw)
            assert not any(k.startswith(p+'_') for k in chain)
            continue
        mechanics = block(p+'_mechanics'); mass = mechanics['mass']
        c.add(p+' full mass',mass.shape == (48,48) and np.isfinite(mass).all())
        c.equal(p+' symmetric mass',mass,mass.T,1e-12,1e-12)
        c.add(p+' positive definite mass',np.linalg.eigvalsh(mass).min() > 0)
        c.equal(p+' independent gain',mechanics['gain'],gains,1e-10,1e-10)
        c.equal(p+' independent bias',mechanics['bias'],bias,1e-10,1e-10)
        jacs = {key:[] for key in OUTPUTS}; valids = []
        for a in ACTUATORS:
            prefix = p+f'_act{a}'; probes=[]
            for label,delta in [('plus_h',1e-5),('minus_h',-1e-5),('plus_half',5e-6),('minus_half',-5e-6)]:
                probe=block(prefix+'_'+label); probes.append(probe)
                expected=z['act'][i].copy(); expected[a]+=delta
                c.exact(prefix+'_'+label+' activation-only signed probe',probe['act'],expected)
                raw_checks(prefix+'_'+label,probe,base,mass)
                c.equal(prefix+'_'+label+' scalar force',probe['actuator_force'],gains*expected+bias,1e-10,1e-10)
            stable=all(all(np.array_equal(probe[k],base[k]) for k in identity) for probe in probes)
            warnings=all(not probe['warnings'].any() for probe in probes)
            numerical={}; dh={}; half={}; rich={}; saved=block(prefix+'_derivatives')
            for key in OUTPUTS:
                if all(probe[key].shape == base[key].shape for probe in probes):
                    dh[key]=(probes[0][key]-probes[1][key])/2e-5
                    half[key]=(probes[2][key]-probes[3][key])/1e-5
                    rich[key]=(4*half[key]-dh[key])/3
                    numerical[key]=criterion(dh[key],half[key],*FD_TOL[key])
                    for tag,values in [('h',dh),('half',half),('richardson',rich)]:
                        c.equal(prefix+' saved '+tag+' '+key,saved[tag+'_'+key],values[key],1e-12,1e-12)
                else:
                    numerical[key]=False; rich[key]=np.full_like(base[key],np.nan)
            tau=base['actuator_moment'][a]*gains[a]; force=np.zeros(90); force[a]=gains[a]
            numerical['analytic_tau']=criterion(half['qfrc_actuator'],tau,1e-6,1e-5)
            numerical['analytic_force']=criterion(half['actuator_force'],force,1e-6,1e-5)
            numerical['differential_dynamics']=criterion(mass@half['qacc'],half['qfrc_actuator']+half['qfrc_constraint'],1e-6,1e-5)
            numerical['probe_force']=all(criterion(gains[None,:]*probe['act'][None,:]+bias[None,:],probe['actuator_force'][None,:],1e-10,1e-10) for probe in probes)
            passed=stable and warnings and all(numerical.values()); valids.append(passed)
            detail=fd_lookup[(i,a)]; verified.append(dict(sample=i,actuator=a,passed=passed))
            c.add(prefix+' gate result',passed == detail['passed'] == bool(saved['valid']))
            c.add(prefix+' branch identity gate',stable == detail['active_set_stable'])
            c.add(prefix+' warning gate',warnings == detail['warnings_zero'])
            c.add(prefix+' classification',detail['label'] == ('VERIFIED_LOCAL' if passed else ('NONSMOOTH' if not stable else 'FD_OR_ANALYTIC_GATE_FAILED')))
            assert set(numerical) == set(detail['checks'])
            for key,value in numerical.items(): c.add(prefix+' gate '+key,value == detail['checks'][key]['passed'])
            c.equal(prefix+' independent analytic torque',saved['analytic_tau'],tau,1e-10,1e-10)
            c.equal(prefix+' independent analytic force',saved['analytic_force'],force,1e-10,1e-10)
            c.equal(prefix+' fixed constraint direct mass response',mass@saved['fixed_constraint_direct_qacc'],tau,1e-10,1e-10)
            for key in OUTPUTS: jacs[key].append(rich[key])
        c.equal(p+' independent selected activation tangent',chain[p+'_activation_derivative'],da[i,ACTUATORS,:],1e-10,1e-10)
        c.add(p+' chain validity',bool(chain[p+'_valid']) == all(valids))
        for key in OUTPUTS:
            if all(valids):
                J=np.stack(jacs[key],axis=-1)
                c.equal(p+' activation Jacobian '+key,chain[p+'_activation_J_'+key],J,1e-12,1e-12)
                c.equal(p+' independent instantaneous weight chain '+key,chain[p+'_weight_J_'+key],J@da[i,ACTUATORS,:],1e-9,1e-10)
            else:
                c.add(p+' no invalid chain '+key,p+'_activation_J_'+key not in chain and p+'_weight_J_'+key not in chain)
    assert set(raw) == expected_keys
    assert set(fd_lookup) == {(r['sample'],r['actuator']) for r in verified}
    ledger=js(out/'CALL_LEDGER.json'); nbase=sum(base_ok)
    expected_ledger=dict(model_load=1,forward=3+20*nbase,baseline_forward=3,perturb_forward=20*nbase,step=0,neural_ticks=0)
    c.add('fixed actual call ledger',ledger == expected_ledger)
    summary=js(out/'SUMMARY.json'); count=sum(r['passed'] for r in verified)
    c.add('summary counts',summary['baseline_passed'] == nbase and summary['baseline_count'] == 3 and summary['verified_activation_columns'] == count and summary['planned_columns'] == 15 and summary['ledger'] == ledger)
    for key in ('full_closed_loop_gradient','new_weights','physical_preheat'): c.add('no claim '+key,summary[key] is False)
    c.add('units remain unestablished',summary['units'] == 'NOT_ESTABLISHED')
    c.add('no candidate weights file',not any('CANDIDATE' in p.name.upper() or 'WEIGHTS' in p.name.upper() for p in out.iterdir()))
    # Producer counts literal committed paths; manifest() deduplicates canonical aliases.
    c.add('frozen before after reported',js(out/'FROZEN_BEFORE_AFTER.json') == dict(passed=True,inputs=len(js(out/'INPUT_HASHES.json'))))
    assert inputs == manifest(out/'INPUT_HASHES.json') and outputs == manifest(out/'OUTPUT_HASHES.json')
    assert 'mujoco' not in sys.modules
    result=dict(passed=all(r['passed'] for r in c.rows),scope='OFFLINE saved-byte integrity and independent local FD/force/mass/contact/load/15-column chain arithmetic; not independent solver execution',
                checks=c.rows,baseline_verified=nbase,verified_activation_columns=count,derivatives=verified,
                model_load=0,forward=0,step=0,neural_ticks=0,new_weights=False,full_closed_loop_gradient=False,physical_preheat=False,standing_or_walking_claim=False,units='NOT_ESTABLISHED',
                verifier_sha256=sha(__file__),input_files=len(inputs),output_files=len(outputs),input_output_hashes_rechecked=True,
                caveat='Historical warmstart absent. Frozen mechanical-state instantaneous chain omits later body/contact/neural/sensory state evolution; no training or behavioral acceptance claim.')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'},ensure_ascii=False))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path,help='Existing S130 result directory; no new simulation')
    main(parser.parse_args().output)
