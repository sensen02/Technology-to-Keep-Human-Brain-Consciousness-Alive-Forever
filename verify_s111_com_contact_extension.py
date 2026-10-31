"""P1/S2: saved-array-only verification of the S111 10 ms COM extension.

No simulator/neural imports or execution. Only this verifier's three report
artifacts are written into the explicitly supplied absolute output directory.
Scientific failures retain reports/plot and return exit status 1.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import sys
import traceback
import numpy as np
import verify_com_contact_probe as old

ROOT = Path(__file__).resolve().parent
HISTORY = Path('/home/sensen/Desktop/engine修复验收/s111-pool-synergy-response/20261024T054520Z')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')


def literal_copy(function, patches, namespace):
    """Derive a separate function; neither old file nor old globals are edited."""
    source = inspect.getsource(function)
    for before, after in patches:
        if before not in source:
            raise ValueError('verifier literal anchor missing: ' + before)
        source = source.replace(before, after)
    scope = dict(namespace)
    exec(compile(source, str(Path(__file__).resolve()) + '::' + function.__name__, 'exec'), scope)
    return scope[function.__name__]


# Requested literal adaptation of the original reconstruct function only.
# The prefix0..17 impulse and FD gates and all original tolerances stay intact.
reconstruct = literal_copy(old.reconstruct, [('31', '101'),
    ('window_ms=[.1,2.9]', 'window_ms=[.1,9.9]'),
    ('interior indices1..29', 'interior indices1..99')], vars(old))


def inherited_verify(out):
    patches = [
        ("str(ROOT/'run_com_contact_probe.py') in hashes", "str(ROOT/'run_s111_com_contact_extension.py') in hashes"),
        ('forward=62,step=60,neural_ticks=172', 'forward=202,step=200,neural_ticks=200'),
        ("pre['physical_duration_ms']==3", "pre['physical_duration_ms']==10"),
        ('np.arange(31)', 'np.arange(101)'), ('(31,49)', '(101,49)'),
        ('(31,48)', '(101,48)'), ('(6,17099)', '(20,17099)'),
        ('(6,66,5)', '(20,66,5)'), ('(6,90)', '(20,90)'),
        ("checks.eq(label+' actual sensory held current',a['current'],np.repeat(init['current'][None],6,axis=0))",
         "checks.eq(label+' initial 5ms sensory held current',a['current'][:10],np.repeat(init['current'][None],10,axis=0)); checks.eq(label+' refreshed 5ms sensory hold',a['current'][10:],np.repeat(a['current'][10:11],10,axis=0))"),
        ("schedule six rows',len(schedule)==6", "schedule twenty rows',len(schedule)==20"),
        ("('NEURAL_FINAL_META.json',86,43)", "('NEURAL_FINAL_META.json',100,50)"),
        ("'OBSERVATION_00.json','SCHEDULE.json'", "'OBSERVATION_00.json','OBSERVATION_10.json','SCHEDULE.json'"),
    ]
    scope = dict(vars(old), reconstruct=reconstruct)
    return literal_copy(old.verify, patches, scope)(out)


def raw_contact_checks(a, meta, checks, label, rec):
    """Reconstruct six tarsal loads/counts and every body-side leg identity."""
    names = [(name or '').lower() for name in meta['body_names']]
    geom_names = [(name or '').lower() for name in meta['geom_names']]
    geom_body = np.asarray(meta['geom_bodyid'])
    leg_of_body = np.asarray([next((i for i, leg in enumerate(LEGS) if name.startswith(leg)), -1) for name in names])
    valid = a['contact_valid_mask'].astype(bool)
    bodies = a['contact_body_ids']
    checks.add(label+' contact_leg_ids integer', np.issubdtype(a['contact_leg_ids'].dtype, np.integer))
    checks.eq(label+' raw body-name contact_leg_ids ownership', a['contact_leg_ids'][valid], leg_of_body[bodies[valid]])
    checks.add(label+' contact leg metadata order', len(names)==73)
    geom_leg = {}
    for g, gn in enumerate(geom_names):
        bn = names[geom_body[g]]
        for i, leg in enumerate(LEGS):
            if 'tars' in gn+bn and (gn.startswith(leg) or bn.startswith(leg)):
                geom_leg[g] = i
    # Original six-foot recorder uses the plane itself (not arbitrary self or
    # non-tarsal contacts). The metadata contains that unique plane/floor ID.
    floors = meta['floor_geom_ids']
    old.require(len(floors)==1, 'unique floor required for six-foot recorder')
    floor = floors[0]
    force = np.zeros((101,6,3)); counts = np.zeros((101,6), dtype=int)
    depth = np.zeros(101)
    world = np.einsum('tcji,tcj->tci', a['contact_frame'].reshape(101,-1,3,3), a['contact_force_raw6'][...,:3])
    for t, c in np.argwhere(valid):
        g1, g2 = a['contact_geom_ids'][t,c]
        if floor not in (g1,g2):
            continue
        foot = g2 if g1==floor else g1
        if foot in geom_leg:
            leg = geom_leg[foot]
            force[t,leg] += (1 if g1==floor else -1)*world[t,c]
            counts[t,leg] += int(a['contact_dist'][t,c] < 0)
        else:
            depth[t] = max(depth[t], -float(a['contact_dist'][t,c]))
    checks.eq(label+' independent raw wrench six-foot force', a['foot_force'], force, 2e-12)
    checks.eq(label+' independent penetrating six-foot contact counts', a['contacts'], counts)
    checks.eq(label+' independent nonfoot depth', a['nonfoot_depth'], depth, 2e-12)
    rhs = a['qfrc_actuator']+a['qfrc_passive']+a['qfrc_constraint']+a['qfrc_applied']-a['qfrc_bias']
    checks.eq(label+' independently summed generalized RHS', a['RHS'], rhs, 2e-12)
    checks.eq(label+' recorded Ma equals independent RHS', a['Ma'], rhs, 1e-8)
    checks.eq(label+' generalized residual identity', a['residual'], a['Ma']-rhs, 2e-12)
    # Full-duration impulse is an additional explicitly preregistered gate;
    # never replace the original prefix0..17 gate with this aggregate.
    impulse = np.vstack((np.zeros(3), np.cumsum(.5*(rec['net_force'][1:]+rec['net_force'][:-1])*np.diff(a['time'])[:,None],axis=0)))
    delta = rec['mass']*(rec['velocity']-rec['velocity'][0])
    err = float(np.abs(impulse-delta).max())
    scale = max(float(rec['mass']*np.linalg.norm(rec['velocity'],axis=1).max()),1e-12)
    checks.add(label+' full10ms impulse all inclusive prefixes0..100', err<=1e-4 and err/scale<=.01,
        max_abs_residual=err, relative_residual=err/scale, scale=scale,
        absolute_tolerance=1e-4, relative_tolerance=.01, window_ms=[0,10],
        impulse=impulse, mass_delta_velocity=delta, residual=impulse-delta)


def historical_checks(out, plotted, checks):
    comparison = {}
    for label, a, rec, result in plotted:
        path = HISTORY/label/'TRAJECTORY.npz'
        archived = old.npz(path)
        shared = sorted(set(a)&set(archived))
        missing = sorted(set(archived)-set(a)); extra = sorted(set(a)-set(archived))
        shapes = {}; compared = []
        for key in shared:
            x,y = a[key],archived[key]
            if x.shape!=y.shape:
                shapes[key] = dict(current=list(x.shape), historical=list(y.shape))
            elif x.dtype.kind in 'biufc' and y.dtype.kind in 'biufc':
                # Equality, not allclose: no tolerance, no prefix slicing.
                exact = bool(np.array_equal(x,y))
                error = float(np.max(np.abs(x.astype(np.complex128)-y.astype(np.complex128)),initial=0))
                checks.add(label+' S111 exact shared numeric '+key, exact,
                    shape=list(x.shape), max_abs_error=error, absolute_tolerance=0)
                compared.append(key)
        checks.add(label+' S111 comparison nonvacuous', len(compared)>0)
        comparison[label] = dict(path=str(path),sha256=old.sha(path), compared_numeric_fields=compared,
            missing_historical_fields=missing, extra_new_fields=extra, different_shape_not_compared=shapes,
            policy='All shared same-shape numeric arrays, value error exactly zero. New-only fields are not failures; no prefix comparison.')
    return comparison


def provenance_checks(out, checks):
    pre = old.js(out/'PREREG.json')
    historical = old.js(HISTORY/'INPUT_HASHES.json')
    drift = {}
    for name, previous in historical.items():
        current = old.sha(name)
        if current!=previous:
            drift[name] = dict(historical=previous,current=current)
    checks.add('exactly four historical source drifts openly declared',len(drift)==4 and pre['source_drift']==drift,
        source_drift=drift, historical_hashes_not_reanchored=True)
    checks.add('source patches drift agreement',old.js(out/'SOURCE_PATCHES.json')['source_drift']==drift)
    checks.add('new prereg baseline runner SHA',pre['base_runner']==str(ROOT/'run_com_contact_probe.py') and old.sha(pre['base_runner'])==pre['base_runner_sha256'])
    checks.add('new prereg derived runner SHA',old.sha(out/'DERIVED_RUNNER.py')==pre['derived_source_sha256'])
    checks.add('historical reference exact',pre['historical_reference']==str(HISTORY))
    checks.add('new prereg no tolerance relaxation',pre['tolerance_relaxation'] is False)
    checks.add('new prereg extension timing',pre['physical_steps_per_arm']==100 and pre['physical_samples_per_arm']==101 and pre['physical_neural_ticks_per_arm']==20)
    checks.add('leg order exact',pre['leg_order']==list(LEGS))
    criteria = pre['closure_criteria']
    required = dict(instantaneous_force_absolute_max_native=1e-8,instantaneous_force_relative_max=1e-6,
        impulse_absolute_max_native=1e-4,impulse_relative_max=.01,pointwise_force_absolute_max_native=1.,
        pointwise_force_relative_max=.05,com_velocity_fd_absolute_max_native=.1,com_velocity_fd_relative_max=.01)
    checks.add('original preregistered scientific tolerances unchanged', all(criteria.get(k)==v for k,v in required.items()) and criteria['window_s']==[0,.0017])
    checks.add('extended impulse prereg bounds',pre['extended_impulse_scope_s']==[0,.01] and pre['extended_impulse_absolute_max_native']==1e-4 and pre['extended_impulse_relative_max']==.01)
    return dict(historical_reference=str(HISTORY),historical_source_drift=drift,
        prereg_sha256=old.sha(out/'PREREG.json'), derived_runner_sha256=old.sha(out/'DERIVED_RUNNER.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    if not out.is_absolute() or not out.is_dir():
        parser.error('--output must be an existing absolute runner output directory')
    report = dict(passed=False,verifier=str(Path(__file__).resolve()),verifier_sha256=old.sha(__file__),
        old_verifier_sha256=old.sha(old.__file__),new_physics_calls=0,new_neural_ticks=0,
        physical_units='NOT_ESTABLISHED',behavior_claim='NONE',selection=False)
    try:
        old.require('mujoco' not in sys.modules,'physics imported')
        checks, plotted, identities = inherited_verify(out)
        report.update(identities)
        # Render before extra gates, so ordinary scientific failures never
        # suppress the requested full-duration diagnostic figure or pair data.
        plot = literal_copy(old.plot, [('ax.set_xlim(0,3)','ax.set_xlim(0,10)'),
            ('S112 independent saved-array COM/contact diagnostic','S111 10ms independent saved-array COM/contact diagnostic')], vars(old))
        report['comparison_png'] = str(plot(out,plotted))
        details = old.pair_details(plotted)
        for label,a,rec,result in plotted:
            details['contact_geometry'][label]['contact_leg_ids'] = a['contact_leg_ids']
            raw_contact_checks(a,old.js(out/label/'COM_CONTACT_METADATA.json'),checks,label,rec)
        details['historical_reproduction'] = historical_checks(out,plotted,checks)
        details['leg_order'] = list(LEGS)
        old.write_json(out/'COM_PAIR_DETAILS.json',details)
        report['pair_details'] = str(out/'COM_PAIR_DETAILS.json')
        report.update(provenance_checks(out,checks))
        checks.add('simulator never imported', 'mujoco' not in sys.modules)
        report.update(passed=checks.passed, checks=checks.rows,
            historical_reproduction=details['historical_reproduction'],
            caveats=['Saved Jacobians and Ma are solver exports, not independently regenerated physics.',
                'Historical source drift is disclosed, not claimed to match old hashes.',
                'Finite-difference gates retain original indices1..17 and original tolerances; full force balance covers101 samples.',
                '10ms is a diagnostic, not standing/walking or validated absolute force units.'])
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        if 'checks' in locals():
            report['checks'] = checks.rows
    old.write_json(out/'VERIFICATION.json',report)
    print(json.dumps(dict(passed=report['passed'], report=str(out/'VERIFICATION.json'),
        failed_checks=[r['name'] for r in report.get('checks',[]) if not r['passed']],error=report.get('error')),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
