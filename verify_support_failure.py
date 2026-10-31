"""S170 independent saved-array verifier. No engine, optimization or helper imports.
Run exactly once after producer: --output /absolute/S170/run/root.
Only writes INDEPENDENT_CHECK.json, refusing to overwrite existing evidence.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

BASE = Path('/home/sensen/Desktop/engine修复验收')
S169 = BASE/'s169-near-initial-support/20261024T233428554259Z'
S167 = BASE/'s167-support-decomposition/20261024T231837641256212Z'
S165 = BASE/'s165-static-support-corrected/20261024T231400000000Z'
SOURCE = BASE/'s154-activation-rate/20261024T215751775489Z/init80ms'
MAP = BASE/'s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_90_MUSCLE_MAPPING.csv'
CHECKS = {}
INPUTS = set()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    INPUTS.add(Path(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def js(path):
    INPUTS.add(Path(path))
    return json.loads(Path(path).read_text())


def rows(path):
    INPUTS.add(Path(path))
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def close(a, b, label, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    err = float(np.max(np.abs(a-b), initial=0.))
    assert err <= tol, (label, err, tol)
    CHECKS[label] = {'shape': list(a.shape), 'maxabs_error': err, 'tolerance': tol}
    return err


def muscle_zero(length, gain, bias, lengthrange):
    # Independent direct scalar static muscle law; velocity is exactly zero.
    lo, hi = gain[4:6]
    scale = (lengthrange[1]-lengthrange[0])/(gain[1]-gain[0])
    l = gain[0]+(length-lengthrange[0])/scale
    mlo, mhi = (lo+1)/2, (hi+1)/2
    if l < lo or l > hi:
        active = 0.
    elif l <= mlo:
        active = .5*((l-lo)/(mlo-lo))**2
    elif l <= 1:
        active = 1-.5*((1-l)/(1-mlo))**2
    elif l <= mhi:
        active = 1-.5*((l-1)/(mhi-1))**2
    else:
        active = .5*((hi-l)/(hi-mhi))**2
    bs = (lengthrange[1]-lengthrange[0])/(bias[1]-bias[0])
    lb = bias[0]+(length-lengthrange[0])/bs
    mid = (1+bias[5])/2
    if lb <= 1:
        passive = 0.
    elif lb <= mid:
        passive = -bias[2]*bias[7]*.5*((lb-1)/(mid-1))**2
    else:
        passive = -bias[2]*bias[7]*(.5+(lb-mid)/(mid-1))
    return -gain[2]*active, passive


def affine(s, native, label):
    assert not s['qvel'].any() and not s['actuator_velocity'].any()
    assert not s['qfrc_applied'].any() and not s['xfrc_applied'].any()
    gain, bias = np.array([muscle_zero(s['actuator_length'][j], native['actuator_gainprm'][j], native['actuator_biasprm'][j], native['actuator_lengthrange'][j]) for j in range(90)]).T
    moment = s['actuator_moment']
    assert moment.shape == (90, 48)
    B = moment.T*gain
    base = moment.T@bias+s['qfrc_passive']-s['qfrc_bias']
    close(s['actuator_force'], gain*s['act']+bias, label+'.native_muscle')
    close(s['qfrc_actuator'], moment.T@s['actuator_force'], label+'.native_qfrc')
    return dict(B=B, base=base, J=s['efc_J_dense'], gain=gain, bias=bias, moment=moment)


def contact(s, edges, meta, label):
    pairs = s['contact_geom'] if 'contact_geom' in s else s['contact_geom_ids']
    n = len(pairs)
    assert pairs.shape == (n, 2) and s['contact_pos'].shape == (n, 3)
    assert n == 18 and s['efc_J_dense'].shape == (72, 48)
    assert np.all(s['contact_dim'] == 3) and np.all(s['contact_row_count'] == 4)
    assert np.all(s['efc_type'] == 6) and np.all(s['contact_valid_mask'])
    assert np.all(edges >= 0.) and edges.shape == (72,)
    addresses = s['contact_efc_address'].astype(int)
    assert np.array_equal(np.sort(np.concatenate([np.arange(a,a+4) for a in addresses])), np.arange(72))
    floors = set(map(int, meta['floor_geom_ids']))
    legs = list(meta['foot_order'])
    geomleg = {int(k): legs.index(v) for k,v in meta['geom_map'].items()}
    forces = np.zeros((6,3)); counts = np.zeros(6, int)
    penetrating = np.zeros(6, int); loaded = np.zeros(6, int)
    occurrence = {}; details = []; used = []
    for ci,(g1,g2) in enumerate(pairs.astype(int)):
        a = addresses[ci]
        assert np.array_equal(np.flatnonzero(s['efc_id'] == ci), np.arange(a,a+4))
        used.extend(range(a,a+4))
        assert (g1 in floors) != (g2 in floors)
        foot = g2 if g1 in floors else g1
        assert foot in geomleg
        li = geomleg[foot]
        mu = s['contact_friction'][ci,:2]
        assert np.all(mu > 0.) and np.isfinite(mu).all()
        frame = s['contact_frame'][ci]
        close(frame@frame.T, np.eye(3), label+f'.contact{ci}.frame')
        e = edges[a:a+4]
        local = np.array([e.sum(), mu[0]*(e[0]-e[1]), mu[1]*(e[2]-e[3])])
        margin = local[0]-np.sum(np.abs(local[1:]/mu))
        assert margin >= -1e-9
        world = (1 if g1 in floors else -1)*(frame.T@local)
        dist = float(s['contact_dist'][ci]); assert np.isfinite(dist)
        key = (int(g1),int(g2)); occ = occurrence.get(key,0); occurrence[key] = occ+1
        forces[li] += world; counts[li] += 1
        penetrating[li] += dist < 0.; loaded[li] += world[2] > 0.
        details.append(dict(state=label, contact_index=ci, geom1=int(g1), geom2=int(g2), occurrence=occ, leg=legs[li], position=s['contact_pos'][ci].tolist(), signed_distance=dist, penetration=max(0.,-dist), row_address=int(a), row_count=4, friction=mu.tolist(), edge_forces=e.tolist(), local_force=local.tolist(), world_force=world.tolist(), pyramid_margin=float(margin)))
    assert sorted(used) == list(range(72))
    return dict(foot_order=legs, leg_force_world=forces.tolist(), leg_contact_counts=counts.tolist(), leg_penetrating_counts=penetrating.tolist(), leg_loaded_counts=loaded.tolist(), contacts=details, matching_policy='Native ordered geom pair plus occurrence is only an aggregate bookkeeping key, NOT physical same-point identity; positions retained separately in each state; no cross-pose index alignment.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    out = p.parse_args().output
    assert out.is_absolute() and out.is_dir()
    product = out/'INDEPENDENT_CHECK.json'
    assert not product.exists(), 'Refuse to overwrite evidence'
    native = load(SOURCE/'NATIVE_MODEL_ARRAYS.npz')
    meta = js(SOURCE/'COM_CONTACT_METADATA.json')
    mapping = sorted(rows(MAP), key=lambda r:int(r['actuator_index']))
    assert [int(r['actuator_index']) for r in mapping] == list(range(90))
    members = [tuple(sorted(map(int,r['local_indices'].split(';')))) if r['local_indices'] else () for r in mapping]
    unique = sorted(set(v for v in members if v)); assert len(unique) == 66
    pool = np.array([unique.index(v) if v else -1 for v in members])
    assert np.array_equal(pool,[int(r['pool_id']) for r in mapping])
    mapped, missing = np.flatnonzero(pool>=0), np.flatnonzero(pool<0)
    assert (len(mapped), len(missing)) == (74,16)
    T = np.zeros((90,66)); T[mapped,pool[mapped]] = .5
    Tf = np.column_stack((T,np.zeros((90,16)))); Tf[missing,66+np.arange(16)] = .5
    s = load(S169/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz')
    best = affine(s,native,'best')
    for k in ('B','base','J','gain','bias'):
        close(s[k],best[k],'best.saved_'+k)
    close(s['T'],T,'best.mapping',0.)
    assert np.all(s['pools']>=0.) and np.all(s['pools']<=1.)
    abest = T@s['pools']; assert not abest[missing].any()
    close(s['act'],abest,'best.actual_activation',0.)
    close(s['ctrl'],abest,'best.actual_control',0.)
    assert np.all(s['efc_D']>0.) and not s['efc_frictionloss'].any()
    f0 = np.where(s['efc_type']==0,s['efc_D']*s['efc_aref'],np.maximum(0.,s['efc_D']*s['efc_aref']))
    close(s['f0'],f0,'best.native_f0_law',0.)
    rbest = best['B']@abest+best['base']+best['J'].T@f0
    close(s['residual48'],rbest,'best.residual48')
    bc = contact(s,f0,meta,'best')
    close(s['foot_load'],bc['leg_force_world'],'best.foot_load')
    rr = rows(S169/'BEST_RESIDUAL48.csv'); assert [int(r['dof']) for r in rr] == list(range(48))
    close([float(r['best']) for r in rr],rbest,'S169.csv_best_residual')
    orig = load(S169/'ORIGINAL_POSE_OPTIMIZED_POOLS_DIAGNOSTIC.npz')
    close([float(r['original']) for r in rr],orig['residual48'],'S169.csv_original_residual')
    lr = rows(S169/'BEST_LEG_SUPPORT.csv'); assert [r['leg'].lower() for r in lr] == [v.lower() for v in meta['foot_order']]
    close([[float(r[k]) for k in ('Fx','Fy','Fz')] for r in lr],bc['leg_force_world'],'S169.csv_leg_load')
    results = []
    for i in (0,15,83):
        label = f'CASE_{i:03d}'
        z = load(S167/(label+'_fill.npz'))
        refstate = load(S165/f'ZERO_V_{i:03d}.npz')
        frozen = load(S165/f'STATIC_AFFINE_{i:03d}.npz')
        ref = affine(refstate,native,label)
        for k in ('B','base','J','gain','bias','moment'):
            close(frozen[k],ref[k],label+'.S165_'+k)
        close(z['T'],Tf,label+'.fill_mapping',0.)
        assert z['x'].shape == (154,) and np.isfinite(z['x']).all()
        assert np.all(z['x']>=0.) and np.all(z['x'][:82]<=1.)
        afill = Tf@z['x'][:82]; lam = z['x'][82:]
        close(z['activation'],afill,label+'.activation',0.)
        close(z['Aeq'],np.column_stack((ref['B']@Tf,ref['J'].T)),label+'.Aeq')
        close(z['beq'],-ref['base'],label+'.beq')
        rfill = ref['B']@afill+ref['base']+ref['J'].T@lam
        close(z['residual'],rfill,label+'.rfill')
        close(rfill,np.zeros(48),label+'.fill_feasibility',1e-8)
        mappedterm = best['B'][:,mapped]@(abest[mapped]-afill[mapped])
        missingterm = -best['B'][:,missing]@afill[missing]
        geometryterm = (best['B']-ref['B'])@afill+(best['base']-ref['base'])
        contactterm = best['J'].T@f0-ref['J'].T@lam
        total = mappedterm+missingterm+geometryterm+contactterm
        delta = rbest-rfill
        close(total,delta,label+'.decomposition_all48',1e-9)
        pose_contact = geometryterm+contactterm
        close(mappedterm+missingterm+pose_contact,delta,label+'.three_classes_all48',1e-9)
        rc = contact(refstate,lam,meta,label+'_fill')
        rootzero = bool(np.count_nonzero(best['B'][:6]) == 0)
        if rootzero:
            close(mappedterm[:6],np.zeros(6),label+'.root_mapped_zero',0.)
            close(missingterm[:6],np.zeros(6),label+'.root_missing_zero',0.)
        arrays = dict(rbest=rbest, rfill=rfill, delta=delta, mappedterm=mappedterm, missingterm=missingterm, geometryterm=geometryterm, contactterm=contactterm, pose_contact=pose_contact, total=total, abest=abest, afill=afill, lambda_fill=lam)
        results.append(dict(sample_index=i, arrays={k:v.tolist() for k,v in arrays.items()}, reference_contact=rc, root_actuator_matrix_exactly_zero=rootzero, root_translation_unweighted_L2_lower_bound_if_Brootzero=float(np.linalg.norm(rbest[:3])) if rootzero else None))
    # Product-schema comparisons are specified below independently of producer imports.
    verify_products(out,results,bc,rbest)
    answer = dict(status='PASS', conclusion='TWO_FACTORS_NOT_DISTINGUISHABLE', scope='Saved witnesses only; no interventions optimized, no necessity proof from selected witness, no ranking by mixed generalized-DOF norm. Native units, SI NOT_ESTABLISHED.', call_ledger=dict(engine=0, optimization=0, model_load=0, forward=0, step=0, neural=0), mapping=dict(pool_count=66,mapped_actuator_count=74,zero_missing_count=16), checks=CHECKS, best_contact=bc, references=results, input_sha256={str(p):digest(p) for p in sorted(INPUTS,key=str)}, verifier_sha256=digest(__file__))
    with product.open('x') as f:
        json.dump(answer,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=answer['status'],output=str(product),check_count=len(CHECKS),conclusion=answer['conclusion'])))


def verify_products(out,results,bc,rbest):
    summary = js(out/'SUMMARY.json')
    assert summary['classification'] == '两者无法由现有保存数据区分'
    assert summary['inputs_unchanged'] is True
    assert all(v == 0 for v in summary['calls'].values())
    prereg = js(out/'PREREG.json')
    assert all(v == 0 for v in prereg['calls'].values())
    hashes = js(out/'INPUT_HASHES.json')
    assert all(digest(p) == h for p,h in hashes.items())
    frozen = sorted((out/'INPUT_FREEZE').iterdir())
    assert len(frozen) == len(hashes)
    for j,(p,h) in enumerate(hashes.items()):
        assert digest(out/'INPUT_FREEZE'/f'{j:03d}_{Path(p).name}') == h
    dr = rows(out/'RESIDUAL_DECOMPOSITION48.csv')
    lr = rows(out/'LEG_SUPPORT_DIFFERENCE.csv')
    cr = rows(out/'CONTACT_EDGES_GAP_FRICTION.csv')
    assert len(dr) == 144 and len(lr) == 18 and len(cr) == 288
    contacts = {'BEST':bc}
    colmap = dict(best_residual='rbest',reference_residual='rfill',mapped_activation_difference='mappedterm',missing_zero_difference='missingterm',pose_muscle_bias_passive_gravity_difference='geometryterm',native_soft_vs_rigid_contact_difference='contactterm',pose_contact_class_sum='pose_contact')
    for result in results:
        i = result['sample_index']; label = f'CASE_{i:03d}'
        a = {k:np.array(v) for k,v in result['arrays'].items()}
        z = load(out/f'DECOMPOSITION_{i:03d}.npz')
        fields = dict(mapped='mappedterm',missing='missingterm',geometry='geometryterm',contact='contactterm',best_residual='rbest',reference_residual='rfill',reference_activation='afill',best_activation='abest')
        assert set(z) == set(fields)|{'closure'}
        for k,v in fields.items():
            close(z[k],a[v],label+'.product_'+k)
        close(z['closure'],a['delta']-a['total'],label+'.product_closure')
        selected = [r for r in dr if int(r['reference_state']) == i]
        assert [int(r['dof']) for r in selected] == list(range(48))
        for col,key in colmap.items():
            close([float(r[col]) for r in selected],a[key],label+'.CSV48_'+col)
        close([float(r['closure']) for r in selected],z['closure'],label+'.CSV48_closure')
        selectedlegs = [r for r in lr if int(r['reference_state']) == i]
        assert [r['leg'].lower() for r in selectedlegs] == [v.lower() for v in bc['foot_order']]
        rc = result['reference_contact']; contacts[str(i)] = rc
        for side, values in [('best',bc['leg_force_world']),('reference',rc['leg_force_world'])]:
            close([[float(r[side+'_'+c]) for c in ('Fx','Fy','Fz')] for r in selectedlegs],values,label+'.CSVlegs_'+side)
        close([float(r['delta_Fz']) for r in selectedlegs],np.array(bc['leg_force_world'])[:,2]-np.array(rc['leg_force_world'])[:,2],label+'.CSVlegs_delta')
        ss = [r for r in summary['contrasts'] if r['reference_state']==i]
        assert len(ss)==1 and 'NOT causal ranking' in ss[0]['warning']
        close(ss[0]['closure_maxabs'],float(np.max(np.abs(z['closure']))),label+'.summary_closure')
        close(ss[0]['reference_maxabs'],float(np.max(np.abs(a['rfill']))),label+'.summary_reference')
        close(ss[0]['best_maxabs'],float(np.max(np.abs(rbest))),label+'.summary_best')
        for name,key in [('mapped','mappedterm'),('missing','missingterm'),('geometry','geometryterm'),('contact','contactterm')]:
            close(ss[0]['diagnostic_maxabs_by_term'][name],float(np.max(np.abs(a[key]))),label+'.summary_'+name)
    for state,c in contacts.items():
        selected = [r for r in cr if r['state']==state]
        assert len(selected)==72
        assert [(int(r['contact']),int(r['edge'])) for r in selected] == [(ci,e) for ci in range(18) for e in range(4)]
        for row in selected:
            ci,e = int(row['contact']),int(row['edge']); v = c['contacts'][ci]
            assert int(row['geom1'])==v['geom1'] and int(row['geom2'])==v['geom2']
            assert int(row['pair_occurrence'])==v['occurrence']+1
            assert row['leg'].lower()==v['leg'].lower() and 'not tracked material contact' in row['matching_policy']
            expected = dict(edge_force=v['edge_forces'][e],distance=v['signed_distance'],mu1=v['friction'][0],mu2=v['friction'][1],Fn=v['local_force'][0],Ft1=v['local_force'][1],Ft2=v['local_force'][2],pyramid_margin=v['pyramid_margin'],min_edge=min(v['edge_forces']),Fx=v['world_force'][0],Fy=v['world_force'][1],Fz=v['world_force'][2])
            for k,value in expected.items():
                close(float(row[k]),value,f'product.contacts.{state}.{ci}.{e}.{k}')
    gaps = js(out/'GAP_FRICTION_SUMMARY.json')
    assert [str(r['state']) for r in gaps] == ['BEST','0','15','83']
    for g in gaps:
        c = contacts[str(g['state'])]
        ds = [r['signed_distance'] for r in c['contacts']]
        close(g['min_distance'],min(ds),'product.gap_min.'+str(g['state']))
        close(g['max_distance'],max(ds),'product.gap_max.'+str(g['state']))
        assert g['contacts']==18 and g['rows']==72
    # Explicitly quantify ambiguity: repeated identical native pairs, positions
    # differ across states; occurrence keys are never material point tracking.
    for result in results:
        ref = result['reference_contact']['contacts']
        best = bc['contacts']
        counts = {}
        for c in best:
            key = str((c['geom1'],c['geom2']))
            counts[key] = counts.get(key,0)+1
        result['contact_identity_audit'] = dict(best_pair_multiplicity=counts,ambiguous_native_pairs=[k for k,v in counts.items() if v>1],best_and_reference_have_same_ordered_pair_list=all((a['geom1'],a['geom2'])==(b['geom1'],b['geom2']) for a,b in zip(best,ref)),same_physical_point_identity_established=False,position_matching_performed=False,reason='Native pairs repeat; occurrence only disambiguates saved records, not a physical same point between poses. All positions stored above for inspection.')


if __name__ == '__main__':
    main()
