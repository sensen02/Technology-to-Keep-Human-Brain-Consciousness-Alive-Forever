"""S169/P1-S2 saved-only native SOFT bounded-search verification.

No engine or producer imports, no optimization, no model/forward/step/neural
calls. Numerical search failure is not an impossibility certificate. Independent
engine reload of an accepted candidate is a separate parent-authorized action.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import numpy as np

BASE = Path('/home/sensen/Desktop/engine修复验收')
S154 = BASE/'s154-activation-rate/20261024T215751775489Z/init80ms'
S165 = BASE/'s165-static-support-corrected/20261024T231400000000Z'
MAPPING = BASE/'s163-sensory-readout-audit/20261024T224712953805523Z/GRAPH_90_MUSCLE_MAPPING.csv'


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, label, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (label, a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all(), label
    err = float(np.max(np.abs(a-b), initial=0.))
    assert err <= tol, (label, err, tol)
    return err


def mapping():
    with MAPPING.open(newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r['actuator_index']))
    assert [int(r['actuator_index']) for r in rows] == list(range(90))
    members = [tuple(sorted(map(int, r['local_indices'].split(';')))) if r['local_indices'] else () for r in rows]
    assert all(len(v) == len(set(v)) for v in members)
    unique = sorted(set(v for v in members if v))
    assert len(unique) == 66
    pool = np.array([unique.index(v) if v else -1 for v in members])
    assert np.array_equal(pool, [int(r['pool_id']) for r in rows])
    assert np.count_nonzero(pool < 0) == 16
    T = np.zeros((90, 66))
    mapped = np.flatnonzero(pool >= 0)
    T[mapped, pool[mapped]] = .5
    return T, pool


def muscle_function():
    # Extract just a pure numerical definition: never execute helper module.
    path = Path(__file__).with_name('newverify_static_support_corrected.py')
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'muscle']
    assert len(nodes) == 1
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(nodes[0]))
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns['muscle']


def contact_check(s, f0, meta):
    """Decode ALL native pyramids, including nonfoot/nonfloor diagnostics."""
    J = s['efc_J_dense']
    nrow = len(s['efc_type']); ncon = len(s['contact_dist'])
    assert J.shape == (nrow, 48) and f0.shape == (nrow,)
    assert np.all(np.isin(s['efc_type'], [0,3,4,5,6])), 'unsupported native row'
    assert not np.any(s['efc_frictionloss'])
    assert int(s['ncon']) == ncon
    assert np.all(np.isin(s['contact_dim'], [1,3]))
    assert np.all(s['contact_row_count'] == np.where(s['contact_efc_address'] < 0, 0, np.where(s['contact_dim'] == 1, 1, 4)))
    assert np.all(s['contact_valid_mask'])
    floors = set(map(int, meta['floor_geom_ids']))
    legs = meta['foot_order']; assert len(legs) == 6
    geomleg = {int(k): legs.index(v) for k, v in meta['geom_map'].items()}
    addresses = s['contact_efc_address'].astype(int)
    used = np.concatenate([np.arange(a, a+int(s['contact_row_count'][ci])) for ci,a in enumerate(addresses) if a >= 0]) if np.any(addresses >= 0) else np.zeros(0, int)
    assert np.array_equal(np.sort(used), np.flatnonzero(np.isin(s['efc_type'], [5,6])))
    force = np.zeros((6, 3)); counts = np.zeros(6, int)
    penetrating = np.zeros(6, int); loaded = np.zeros(6, int)
    max_pen = 0.; nonfoot_max = 0.; nonfoot_count = 0; details = []
    assert np.isfinite(s['contact_dist']).all()
    for ci, addr in enumerate(addresses):
        count = int(s['contact_row_count'][ci])
        expected_rows = np.arange(addr, addr+count) if addr >= 0 else np.zeros(0,int)
        assert np.array_equal(np.flatnonzero((s['efc_id'] == ci)&np.isin(s['efc_type'],[5,6])), expected_rows)
        edges = f0[addr:addr+count] if addr >= 0 else np.zeros(0)
        assert np.all(edges >= 0.)
        mu = s['contact_friction'][ci, :2]
        assert np.isfinite(mu).all()
        frame = s['contact_frame'][ci]
        close(frame@frame.T, np.eye(3), 'orthonormal native frame')
        if count == 4:
            assert np.all(mu > 0.) and np.all(s['efc_type'][expected_rows] == 6)
            local = np.array([edges.sum(), mu[0]*(edges[0]-edges[1]), mu[1]*(edges[2]-edges[3])])
            margin = float(local[0]-np.sum(np.abs(local[1:]/mu)))
        else:
            assert count in (0,1)
            if count: assert s['efc_type'][addr] == 5
            local = np.array([edges.sum(),0.,0.]); margin = float(local[0])
        assert margin >= -1e-9
        g1, g2 = map(int, s['contact_geom'][ci])
        isfloor = (g1 in floors) != (g2 in floors)
        other = g2 if g1 in floors else g1
        isfoot = isfloor and other in geomleg
        penetration = max(0., -float(s['contact_dist'][ci]))
        max_pen = max(max_pen, penetration)
        world = (1 if g1 in floors else -1)*(frame.T@local)
        if isfoot:
            li = geomleg[other]; counts[li] += 1; penetrating[li] += penetration > 0
            loaded[li] += world[2] > 0; force[li] += world
        else:
            nonfoot_count += 1; nonfoot_max = max(nonfoot_max, penetration)
        details.append(dict(contact=ci, geom1=g1, geom2=g2, foot=bool(isfoot),
                            leg=legs[geomleg[other]] if isfoot else None,
                            signed_dist=float(s['contact_dist'][ci]), penetration=penetration,
                            friction=mu.tolist(), pyramid_margin=margin,
                            force_world=world.tolist()))
    return dict(contact_count=ncon, row_count=nrow, foot_order=legs,
                leg_contact_counts=counts.tolist(), leg_penetrating_counts=penetrating.tolist(),
                leg_positive_Fz_counts=loaded.tolist(), leg_force_world=force.tolist(),
                max_penetration=max_pen, nonfoot_contact_count=nonfoot_count,
                max_nonfoot_penetration=nonfoot_max, contacts=details)


def reconstruct(s, pools, native, T, meta):
    assert pools.shape == (66,) and np.isfinite(pools).all()
    assert np.all(pools >= 0.) and np.all(pools <= 1.)
    assert not s['qvel'].any() and not s['actuator_velocity'].any()
    assert not s['act'].any() and not s['ctrl'].any(), 'pose probe must be act=ctrl=0'
    assert not s['qfrc_applied'].any() and not s['xfrc_applied'].any()
    scalar = muscle_function()
    gain, bias = np.array([scalar(s['actuator_length'][j], 0., 0., native['actuator_gainprm'][j],
                                 native['actuator_biasprm'][j], native['actuator_lengthrange'][j])[1:]
                           for j in range(90)]).T
    moment = s['actuator_moment']; assert moment.shape == (90, 48)
    close(s['actuator_force'], bias, 'zero-act native muscle bias')
    close(s['qfrc_actuator'], moment.T@bias, 'zero-act muscle generalized force')
    B = moment.T*gain[None, :]
    base = moment.T@bias+s['qfrc_passive']-s['qfrc_bias']
    D, aref = s['efc_D'], s['efc_aref']
    assert np.isfinite(D).all() and np.all(D > 0.)
    assert np.isfinite(aref).all()
    f0 = np.where(s['efc_type'] == 0, D*aref, np.maximum(D*aref, 0.))
    contact = contact_check(s, f0, meta)
    act = T@pools
    residual = B@act+base+s['efc_J_dense'].T@f0
    assert residual.shape == (48,) and np.isfinite(residual).all()
    return dict(B=B, base=base, gain=gain, bias=bias, f0=f0, activation=act,
                residual=residual, contact=contact)


def plot_saved(out, original, best, original_recon, best_recon, meta, accepted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 10))
    names = meta['body_names']
    structure=json.loads((out/'BODY_STRUCTURE.json').read_text())
    parents=structure['body_parentid']
    for panel, axes in enumerate(((0, 1), (0, 2))):
        ax = fig.add_subplot(2, 2, panel+1)
        for s, label, color in ((original, 'Original probe', 'tab:gray'), (best, 'Best saved probe', 'tab:blue')):
            xyz = s.get('xpos', s.get('body_xpos')); assert xyz.shape == (len(names), 3)
            # Every fly body is a saved-position marker, not a rendered surface.
            ids = np.array([j for j,n in enumerate(names) if n not in ('world', 'ground')])
            ax.scatter(xyz[ids, axes[0]], xyz[ids, axes[1]], s=16, alpha=.7, label=label, c=color)
            for j in ids:
                parent=int(parents[j])
                if parent in ids:
                    ax.plot(xyz[[j,parent],axes[0]],xyz[[j,parent],axes[1]],color=color,lw=.6,alpha=.45)
            for j in ids:
                if names[j] in ('Head', 'Thorax', 'A1A2', 'A3', 'A4', 'A5', 'A6', 'LWing', 'RWing'):
                    ax.annotate(names[j], (xyz[j, axes[0]], xyz[j, axes[1]]), fontsize=6, color=color)
        ax.set_xlabel('xyz'[axes[0]]+' (native)'); ax.set_ylabel('xyz'[axes[1]]+' (native)')
        ax.set_aspect('equal', adjustable='datalim'); ax.legend(fontsize=8)
        ax.set_title('Saved full-body position markers (not mesh render)')
    ax = fig.add_subplot(2,2,3)
    f_orig=np.asarray(original_recon['contact']['leg_force_world'])[:,2]
    f_best=np.asarray(best_recon['contact']['leg_force_world'])[:,2]
    x=np.arange(6); ax.bar(x-.18, f_orig,.36,label='Original f0'); ax.bar(x+.18,f_best,.36,label='Best f0')
    ax.set_xticks(x,meta['foot_order']); ax.set_ylabel('Native static f0 world Fz'); ax.legend()
    ax.set_title('Prescribed native SOFT f0, not arbitrary contact forces')
    ax=fig.add_subplot(2,2,4)
    ax.plot(np.arange(48),original_recon['residual'],label='Original best-pools residual')
    ax.plot(np.arange(48),best_recon['residual'],label='Best residual'); ax.axhline(0,c='k',lw=.5)
    ax.set_xlabel('All 48 generalized DOFs'); ax.set_ylabel('Native equilibrium residual'); ax.legend()
    classification='SAVED CANDIDATE — independent reload not performed here' if accepted else 'BEST NUMERICAL FAILURE — no impossibility claim'
    fig.suptitle('S169 bounded near-initial native SOFT search: '+classification+'\nNo dynamics, no neural/FIR attainability; SI NOT_ESTABLISHED',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93)); path=out/'SAVED_BEST_COMPARISON.png'
    assert not path.exists(), 'Refuse to overwrite plot'
    fig.savefig(path,dpi=160); plt.close(fig)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); assert args.output.is_absolute()
    out = args.output.resolve(); product = out/'S169_SAVED_INDEPENDENT_CHECK.json'
    assert not product.exists(), 'Refuse to overwrite independent evidence'
    prereg = json.loads((out/'PREREG.json').read_text())
    summary = json.loads((out/'SUMMARY.json').read_text())
    ledger = json.loads((out/'CALL_LEDGER.json').read_text())
    records = json.loads((out/'EVALUATIONS.json').read_text())
    frozen = load(out/'FROZEN_SEARCH.npz')
    hashes = json.loads((out/'INPUT_HASHES.json').read_text())
    assert hashes == json.loads((out/'INPUT_HASHES_AFTER.json').read_text())
    assert all(sha(p) == h for p,h in hashes.items())
    manifest = json.loads((out/'SOURCE_FREEZE_MANIFEST.json').read_text())
    assert all(sha(v['copy']) == hashes[p] == v['sha256'] for p,v in manifest.items())
    outputhash = json.loads((out/'OUTPUT_HASHES.json').read_text())
    assert all(sha(p) == h for p,h in outputhash.items())
    T,pool = mapping(); close(frozen['T'],T,'actual CSV T',0.)
    z = load(S154/'TRAJECTORY.npz'); native = load(S154/'NATIVE_MODEL_ARRAYS.npz')
    meta = json.loads((S154/'COM_CONTACT_METADATA.json').read_text())
    body = json.loads((out/'BODY_STRUCTURE.json').read_text())
    assert body['body_names'] == meta['body_names'] and len(body['body_names']) == 73
    assert body['original_xml_unmodified'] and body['rendering_not_performed']
    q0=z['qpos'][0]; close(frozen['qpos0'],q0,'single S154 index0',0.)
    indices=np.r_[2,np.arange(7,49)]; assert np.array_equal(frozen['indices'],indices)
    # Independently read original native joint bounds from XML, not search output.
    import xml.etree.ElementTree as ET
    xml = ET.parse(prereg['model']['xml'])
    joints=list(xml.getroot().find('worldbody').iter('joint'))
    assert len(joints)==43 and joints[0].get('type')=='free'
    ranges=np.array([list(map(float,j.get('range').split())) for j in joints[1:]])
    radius=np.minimum(.15,.1*np.diff(ranges,axis=1).ravel())
    lower=np.r_[.95*q0[2],np.maximum(q0[7:]-radius,ranges[:,0])]
    upper=np.r_[1.15*q0[2],np.minimum(q0[7:]+radius,ranges[:,1])]
    close(frozen['joint_ranges'],ranges,'original native joint range',0.)
    close(frozen['lower'],lower,'preregistered lower',0.); close(frozen['upper'],upper,'preregistered upper',0.)
    close(prereg['bounds_values']['lower'],lower,'JSON lower',0.)
    close(prereg['bounds_values']['upper'],upper,'JSON upper',0.)
    assert prereg['maximum_actual_forwards']==200 and prereg['maximum_pose_evaluations']==99
    prior=load(S165/'STATIC_AFFINE_000.npz'); zero=load(S165/'ZERO_V_000.npz')
    scale=np.maximum(np.max(abs(prior['B']@T),axis=1)+abs(prior['base'])+abs(prior['J'].T@prior['f0']),1e-12)
    close(frozen['row_scale'],scale,'frozen row scale',0.)
    depthcap=float(np.maximum(-zero['contact_dist'],0).max(initial=0))
    close(frozen['original_foot_depth'],depthcap,'original depth cap',0.)
    assert prereg['accept']['foot_depth_cap']==depthcap
    files=sorted(out.glob('FORWARD_*.npz'))
    assert ledger==summary['ledger']
    assert ledger['model_load']==1 and ledger['forward']==len(files)<=200
    assert ledger['baseline_forward']==1
    assert ledger['probe_forward']==ledger['optimized_state_forward']==ledger['pose_evaluations']==len(records)<=99
    assert ledger['forward']==1+2*len(records)
    assert all(ledger[k]==0 for k in ('constraintUpdate','step','neural_ticks'))
    assert [int(p.name.split('_')[1]) for p in files]==list(range(1,len(files)+1))
    baseline=load(out/'FORWARD_001_baseline_forward.npz')
    for k in ('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied','qacc','qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_constraint','actuator_force','actuator_length','actuator_velocity','actuator_moment'):
        close(baseline[k],z[k][0],'baseline S154 '+k,1e-9)
    close(baseline['qacc_warmstart_input'],z['native_qacc_warmstart'][0],'baseline warmstart',0.)
    checked=[]; reconstructions={}; states={}; accepted=[]
    qscale=np.r_[np.full(3,9801.),np.full(45,9801./q0[2])]
    fixed=np.setdiff1d(np.arange(49),indices)
    for eid,row in enumerate(records,1):
        assert row['evaluation']==eid and row['forwards']==1+2*eid
        e=load(out/f'EVAL_{eid:03d}.npz')
        assert int(e['probe_forward_id'])==2*eid and int(e['actual_forward_id'])==2*eid+1
        probe=load(out/f'FORWARD_{2*eid:03d}_probe_forward.npz')
        actual=load(out/f'FORWARD_{2*eid+1:03d}_optimized_state_forward.npz')
        q=e['qpos']; assert q.shape==(49,)
        close(q[fixed],q0[fixed],'fixed rootxy/quaternion/nonleg',0.)
        assert np.all(q[indices]>=lower) and np.all(q[indices]<=upper)
        for state in (probe,actual):
            close(state['qpos'],q,'same pose',0.)
            assert not state['qvel'].any() and not state['qacc_warmstart_input'].any()
            assert not state['qfrc_applied'].any() and not state['xfrc_applied'].any()
            assert not state['actuator_velocity'].any()
            closure=state['M']@state['qacc']-(state['qfrc_actuator']+state['qfrc_passive']+state['qfrc_constraint']-state['qfrc_bias'])
            close(state['dynamic_closure'],closure,'forward dynamic closure',0.)
            close(state['qfrc_constraint'],state['efc_J_dense'].T@state['efc_force'],'native J.T force',1e-8)
        r=reconstruct(probe,e['pools'],native,T,meta)
        for key,other in [('B','B'),('base','base'),('gain','gain'),('bias','bias'),('f0','f0'),('activation','act'),('residual','residual48')]:
            close(e[other],r[key],'independent '+key,1e-9)
        close(e['T'],T,'evaluation T',0.); close(e['J'],probe['efc_J_dense'],'J',0.)
        close(actual['act'],r['activation'],'best actual mapped act',0.)
        close(actual['ctrl'],actual['act'],'constant ctrl=act',0.)
        assert not actual['act'][pool<0].any() and not actual['act_dot'].any()
        assert np.all(actual['act']>=0) and np.all(actual['act']<=.5)
        close(e['actual_forward_qacc'],actual['qacc'],'saved actual qacc',0.)
        close(e['actual_forward_efc_force'],actual['efc_force'],'saved actual native force not f0',0.)
        close(e['probe_forward_qacc'],probe['qacc'],'saved probe qacc',0.)
        ranges_ctrl=np.asarray(json.loads((S165/'AFFINITY_PROOF.json').read_text())['ctrlrange'])
        assert np.all(actual['ctrl']>=ranges_ctrl[:,0]) and np.all(actual['ctrl']<=ranges_ctrl[:,1])
        for pi in range(66):
            assert np.all(actual['act'][pool==pi]==.5*e['pools'][pi])
        for k in ('actuator_length','actuator_moment','qfrc_passive','qfrc_bias','efc_D','efc_aref','efc_type','efc_J_dense','contact_geom','contact_dist'):
            close(actual[k],probe[k],'pose geometry invariant '+k,1e-10)
        close(actual['actuator_force'],r['gain']*actual['act']+r['bias'],'actual muscle force')
        close(actual['qfrc_actuator'],actual['actuator_moment'].T@actual['actuator_force'],'actual muscle generalized')
        c=r['contact']; footdepth=max((x['penetration'] for x in c['contacts'] if x['foot']),default=0.)
        bearing={x['leg'] for x in c['contacts'] if x['foot'] and x['signed_dist']<=0 and x['force_world'][2]>0}
        # Producer counts positive normal force; use native local normal sum exactly.
        bearing=set()
        for ci,x in enumerate(c['contacts']):
            addr=int(probe['contact_efc_address'][ci]); count=int(probe['contact_row_count'][ci])
            if x['foot'] and x['signed_dist']<=0 and addr>=0 and r['f0'][addr:addr+count].sum()>0: bearing.add(x['leg'])
        margin=min((x['pyramid_margin'] for x in c['contacts'] if int(probe['contact_efc_address'][x['contact']])>=0),default=0.)
        maxres=float(abs(r['residual']).max()); qnorm=float(abs(actual['qacc']/qscale).max())
        limits=bool(np.all(q[7:]>=ranges[:,0]-1e-8) and np.all(q[7:]<=ranges[:,1]+1e-8))
        controlok=bool(np.array_equal(actual['act'],actual['ctrl']) and not actual['act_dot'].any())
        passed=bool(maxres<=1e-8 and abs(actual['dynamic_closure']).max()<=1e-8 and qnorm<=1e-6 and footdepth<=depthcap+1e-12 and c['max_nonfoot_penetration']<=1e-12 and len(bearing)>=3 and margin>=-1e-10 and limits and controlok)
        x=(q[indices]-lower)/(upper-lower); x0=(q0[indices]-lower)/(upper-lower)
        penalties=np.array([max(footdepth-depthcap,0.)/q0[2],c['max_nonfoot_penetration']/q0[2],max(3-len(bearing),0.),max(-margin,0.)])
        vector=np.r_[r['residual']/scale,penalties,1e-8*(x-x0)]
        close(e['objective_vector'],vector,'objective vector',1e-9)
        objective=float(vector@vector); close(row['objective'],objective,'objective',1e-8)
        close(row['max_abs_native_residual'],maxres,'residual max',1e-9)
        close(row['normalized_qacc'],qnorm,'normalized qacc',1e-9)
        close(row['foot_depth'],footdepth,'footdepth',1e-12)
        close(row['nonfoot_depth'],c['max_nonfoot_penetration'],'nonfoot depth',1e-12)
        assert row['contact_legs']==len(bearing) and row['accepted']==passed
        close(e['foot_load'],c['leg_force_world'],'native soft perleg forces',1e-9)
        if passed: accepted.append(eid)
        checked.append(dict(evaluation=eid,objective=objective,max_abs_residual=maxres,normalized_qacc=qnorm,accepted=passed,contact=c))
        reconstructions[eid]=r; states[eid]=actual
    assert records
    for row in records:
        geometry_ok=bool(row['foot_depth']<=depthcap+1e-12 and row['nonfoot_depth']<=1e-12 and row['contact_legs']>=3 and row['friction_margin']>=-1e-10 and row['control_ok'])
        assert row['geometry_ok']==geometry_ok
    bestid=min(range(1,len(records)+1),key=lambda i:(not records[i-1]['geometry_ok'],records[i-1]['max_abs_native_residual'],records[i-1]['objective']))
    assert summary['best_diagnostic']['evaluation']==bestid
    assert sha(out/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz')==sha(out/f'EVAL_{bestid:03d}.npz')
    if accepted:
        assert accepted==[len(records)] and summary['accepted']['evaluation']==accepted[0]
        assert summary['status']=='ACCEPTED_STATIC_DIAGNOSTIC_ONLY'
        assert sha(out/'ACCEPTED_CANDIDATE.npz')==sha(out/f'EVAL_{accepted[0]:03d}.npz')
        classification='SAVED_STATIC_CANDIDATE_PENDING_SEPARATE_INDEPENDENT_MODEL_RELOAD'
    else:
        assert summary['accepted'] is None and summary['status']=='SEARCH_FAILED'
        assert not (out/'ACCEPTED_CANDIDATE.npz').exists()
        classification='NUMERICAL_BOUNDED_SEARCH_FAILED_NOT_IMPOSSIBILITY'
    plot=plot_saved(out,baseline,states[bestid],reconstructions[1],reconstructions[bestid],meta,bool(accepted))
    report=dict(verified=True,scope='S169/P1-S2 saved-only all evaluations, all48 residuals and complete ledger',
                classification=classification,separation='NOT_APPLICABLE: nonlinear pose search failure is not global infeasibility',
                independent_engine_reload='NOT_PERFORMED: parent explicit candidate acceptance required',
                actual_verifier_calls=dict(engine_import=0,model_load=0,forward=0,step=0,neural_ticks=0,optimization=0),
                producer_ledger=ledger,best_evaluation=bestid,accepted_evaluations=accepted,evaluations=checked,
                plot=plot,units='native; SI NOT_ESTABLISHED',limitations='No dynamics/standing/walking/neural-FIR attainability claim',
                input_hashes_unchanged=all(sha(p)==h for p,h in hashes.items()),
                output_hashes_unchanged=all(sha(p)==h for p,h in outputhash.items()),verifier_sha256=sha(__file__))
    assert report['input_hashes_unchanged'] and report['output_hashes_unchanged']
    product.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True,classification=classification,best_evaluation=bestid,producer_ledger=ledger,report=str(product),plot=plot)))


if __name__=='__main__':
    main()
