"""S165 corrected P1/S2 static-support diagnostic. IMPLEMENTATION ONLY until parent review.
No integration, neural imports, endpoint forwards, or application of LP solutions.
Run once by parent: venv_body/bin/python -B this_file --output NEW_ABSOLUTE_TIMESTAMP.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import csv
import hashlib
import json
import shutil
import traceback
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = Path('/home/sensen/Desktop/engine修复验收')
SOURCE = BASE/'s154-activation-rate/20261024T215751775489Z'
MAP = BASE/'s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
POINTS = (0, 15, 83)
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
GATE_ATOL, LP_ATOL = 1e-9, 1e-8
REQUIRED = ('time', 'qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'native_qacc_warmstart')
OPTIONAL = ('mocap_pos', 'mocap_quat', 'eq_active', 'plugin_state', 'userdata')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def csvout(path, records):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def extract(path, names):
    tree = ast.parse(path.read_text())
    return '\n'.join(ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)) for name in names)+'\n'


def check(a, b, exact=False):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return dict(passed=False, shape_a=list(a.shape), shape_b=list(b.shape))
    error = float(np.max(np.abs(a.astype(float)-b.astype(float)), initial=0))
    return dict(passed=bool(np.array_equal(a, b) if exact else np.isfinite(error) and error <= GATE_ATOL), max_abs=error)


def solve_lp(folder, A, b, upper, rigid, linprog):
    """Absolute native residual decides feasibility, never HiGHS status alone.
    Independent support-function separation: y.b > sup_box/nonnegative y.A.x.
    A recession coefficient must be <=0 EXACTLY before a certificate is accepted.
    No certificate => status 2 means numerical infeasibility only, not incapacity.
    """
    n = A.shape[1]
    ub = np.r_[np.full(90, upper), np.full(n-90, np.inf)]
    cost = np.r_[np.ones(90), np.zeros(n-90)]
    bounds = np.column_stack((np.zeros(n), ub))
    scale = np.maximum(np.max(np.abs(A), axis=1), 1e-12)
    result = linprog(cost, A_eq=A/scale[:, None], b_eq=b/scale,
                     bounds=[(0., None if np.isinf(u) else float(u)) for u in ub],
                     method='highs', options={'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})
    x = np.full(n, np.nan) if result.x is None else result.x.copy()
    residual = np.full(48, np.nan) if result.x is None else A@x-b
    feasible = bool(result.x is not None and np.isfinite(x).all() and np.max(np.abs(residual)) <= LP_ATOL
                    and np.min(x) >= -LP_ATOL and np.all(x[:90] <= upper+LP_ATOL))
    saved = dict(Aeq=A, beq=b, bounds=bounds, cost=cost, row_scale=scale, x=x,
                 residual=residual, lower_margin=x, upper_margin=ub-x)
    for key in ('eqlin','ineqlin','lower','upper'):
        block = getattr(result,key,None)
        if block is not None:
            for field in ('residual','marginals'):
                value = getattr(block,field,None)
                if value is not None: saved[key+'_'+field] = value
    # Phase I: min t subject to |Ax-b|<=t in native units, not a feasibility proof by itself.
    phase = linprog(np.r_[np.zeros(n),1.], A_ub=np.vstack((np.column_stack((A,-np.ones(48))),np.column_stack((-A,-np.ones(48))))),
                    b_ub=np.r_[b,-b], bounds=[(0., None if np.isinf(u) else float(u)) for u in ub]+[(0.,None)], method='highs')
    saved['phaseI_x'] = np.array([]) if phase.x is None else phase.x
    certificate = dict(accepted=False, interpretation='No independently validated separation; solver infeasibility is numerical/model-only, not muscle incapacity.')
    if not feasible:
        # y in [-1,1]^48, t_j >= max(Aact[:,j].y,0).
        sep_A = np.column_stack((A[:,:90].T, -np.eye(90)))
        sep_rhs = np.zeros(90)
        if rigid:
            sep_A = np.vstack((sep_A,np.column_stack((A[:,90:].T,np.zeros((n-90,90))))))
            sep_rhs = np.r_[sep_rhs,np.zeros(n-90)]
        sep = linprog(np.r_[-b,np.full(90,upper)],A_ub=sep_A,b_ub=sep_rhs,
                      bounds=[(-1.,1.)]*48+[(0.,None)]*90,method='highs')
        if sep.x is not None:
            y = sep.x[:48]
            # Re-evaluate independently in extended precision; save every term.
            coeff = A.astype(np.longdouble).T@y.astype(np.longdouble)
            dot = b.astype(np.longdouble)@y.astype(np.longdouble)
            support = np.longdouble(upper)*np.maximum(coeff[:90],0).sum()
            gap = dot-support
            recession_ok = bool(np.all(coeff[90:] <= 0))
            accepted = bool(recession_ok and gap > LP_ATOL*np.sum(np.abs(y)))
            saved.update(certificate_y=y, certificate_ATy=np.asarray(coeff,float), certificate_bTy=np.asarray(float(dot)),
                         certificate_support=np.asarray(float(support)), certificate_gap=np.asarray(float(gap)))
            certificate = dict(accepted=accepted, gap=float(gap), bTy=float(dot), support=float(support),
                               recession_nonpositive_exact=recession_ok, residual_tolerance_budget=float(LP_ATOL*np.abs(y).sum()),
                               interpretation='Independent support-function separation in this linear model only.' if accepted else 'Separation not certified; numerical infeasibility only.')
    np.savez_compressed(Path(str(folder)+'.npz'), **saved)
    info = dict(solver_status=int(result.status), solver_message=result.message, feasible_absolute_native=feasible,
                max_abs_residual=None if result.x is None else float(np.max(np.abs(residual))),
                phaseI_status=int(phase.status), phaseI_min_infinity_residual=None if phase.x is None else float(phase.x[-1]),
                objective=None if not feasible else float(cost@x), objective_meaning='minimum activation sum diagnostic, not unique or neural', certificate=certificate)
    dump(Path(str(folder)+'.json'), info)
    return info, x, residual


def main(out):
    assert out.is_absolute() and not out.exists(), 'Fresh absolute timestamp directory required; never retry existing output'
    out.mkdir(parents=True)
    freeze = out/'SOURCE_FREEZE'
    freeze.mkdir()
    ledger = dict(model_load=0, forward=0, dynamic_forward=0, zeroV_forward=0, step=0, neural_ticks=0)
    inputs = {}
    gates = []
    def reserve(kind):
        ledger[kind] += 1
        if kind in ('dynamic_forward','zeroV_forward'): ledger['forward'] += 1
        assert ledger['model_load'] <= 1 and ledger['forward'] <= 6
        dump(out/'CALL_LEDGER.json',ledger)
    def finish(status):
        unchanged = {p:sha(p)==h for p,h in inputs.items()}
        dump(out/'INPUT_HASHES_AFTER.json', {p:sha(p) for p in inputs})
        dump(out/'SUMMARY.json',dict(status=status,ledger=ledger,baseline_gates=gates,all_inputs_unchanged=all(unchanged.values()),
                                   units='native; physical units NOT_ESTABLISHED',neural_controller=False,trajectory=False,integration=False,solutions_applied=False))
        dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='OUTPUT_HASHES.json'})
        assert all(unchanged.values()), 'Input identity changed; preserve failure, no retry'
    dump(out/'CALL_LEDGER.json',ledger)
    try:
        paths = [Path(__file__).resolve(), XML, MAP, ROOT/'run_fixed_state_sensitivity.py',
                 SOURCE/'INPUT_HASHES.json', SOURCE/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py']
        paths += [SOURCE/'init80ms'/n for n in ('TRAJECTORY.npz','CONSTRAINT_ROWS.npz','MODEL_PROVENANCE.json','NATIVE_MODEL_ARRAYS.npz','COM_CONTACT_METADATA.json')]
        specs = [(ROOT/'run_com_contact_probe.py',('com_contact_record',)),
                 (ROOT/'run_constraint_row_recording.py',('constraint_record',)),
                 (ROOT/'run_matched_state_counterfactual.py',('snapshot',)),
                 (ROOT/'run_local_pool_response.py',('moment_record',)),
                 (ROOT/'verify_muscle_reconstruction.py',('fl','evaluate'))]
        paths += [p for p,_ in specs]
        original = json.loads((SOURCE/'INPUT_HASHES.json').read_text())
        original = original.get('hashes',original)
        for p,h in original.items():
            if isinstance(h,str) and len(h)==64:
                assert sha(p)==h, 'Original binding changed: '+p
                inputs[p]=h
        for p in paths: inputs[str(p)]=sha(p)
        dump(out/'INPUT_HASHES.json',inputs)
        # Freeze all original bound inputs individually, including assets; no parent/new verifier discovery.
        for index,(p,h) in enumerate(sorted(inputs.items())):
            target=freeze/f'{index:04d}_{Path(p).name}'
            shutil.copyfile(p,target)
            assert sha(target)==h
        text='\n'.join(extract(p,names) for p,names in specs)
        (freeze/'INERT_EXTRACTED_FUNCTIONS.py').write_text(text)
        dump(out/'SOURCE_FREEZE_MANIFEST.json',{p:dict(sha256=h,copy=str(freeze/f'{index:04d}_{Path(p).name}')) for index,(p,h) in enumerate(sorted(inputs.items()))})
        dump(out/'PREREG.json',dict(scope='S165/P1/S2 offline fixed-q static support diagnostic ONLY',source=str(SOURCE),selected_indices=POINTS,
             selected_ms=[0.,.375,2.075],model_load=1,forwards=6,first_dynamic_forwards=3,then_zeroV_forwards=3,steps=0,neural_ticks=0,
             stop='Any model, missing-state, or baseline failure stops immediately without retry; all three dynamic gates must pass before zeroV',
             model=dict(xml=str(XML),mujoco='3.9.0',tau_act=.001,tau_deact=.0004,integrator='RK4',dt=.000025),
             baseline_atol=GATE_ATOL,relative_tolerance=None,LP_absolute_native_atol=LP_ATOL,row_scale='max(max(abs(Aeq row)),1e-12)',
             soft='qacc_desired=0, f0=max(D*aref,0), B=moment.T*gain, base=moment.T*bias+passive-qfrc_bias; B act=-base-J.T f0',
             rigid='[B,J.T] [act,lambda]=-base; lambda>=0, native 72 pyramidal edge rows; relaxed rigid upper bound only',
             activation_upper_bounds=[.5,1.],objective='min sum(act), diagnostic only, not unique',
             force_affinity='compiled muscle gain/bias, no force/joint/tendon actuator limits, scalar evaluate(act=0), verify actual force reconstruction',
             control='offline possible ctrl=act with native clamp and rate=0 proof only; never applied',
             no_endpoint_forwards=True,no_solution_application=True,no_qacc_overwrite=True,no_integration=True,no_neural_import=True,
             original_binding_count=sum(isinstance(h,str) and len(h)==64 for h in original.values()),bound_input_count=len(inputs),
             historical_invalid_output='untouched and not imported',units='native; physical units NOT_ESTABLISHED'))
        z=load(SOURCE/'init80ms/TRAJECTORY.npz')
        rows=load(SOURCE/'init80ms/CONSTRAINT_ROWS.npz')
        proof=json.loads((SOURCE/'init80ms/MODEL_PROVENANCE.json').read_text())
        native=load(SOURCE/'init80ms/NATIVE_MODEL_ARRAYS.npz')
        meta=json.loads((SOURCE/'init80ms/COM_CONTACT_METADATA.json').read_text())
        mapping=json.loads(MAP.read_text())
        ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
        exec(compile(text,str(freeze/'INERT_EXTRACTED_FUNCTIONS.py'),'exec'),ns)
        # Engine and solver imports occur ONLY after immutable input hashes, source freeze, prereg.
        import mujoco as mj
        from scipy.optimize import linprog
        assert mj.__version__=='3.9.0'
        for name in ('control','act_dyn','act_gain','act_bias','passive','sensor'):
            getter=getattr(mj,'get_mjcb_'+name,None)
            assert getter is None or getter() is None
        reserve('model_load')
        m=mj.MjModel.from_xml_path(str(XML))
        before=ns['snapshot'](m)
        assert before==proof['before'], 'S154 prepatch model mismatch'
        m.actuator_dynprm[:,0]=.001
        m.actuator_dynprm[:,1]=.0004
        m.opt.integrator=mj.mjtIntegrator.mjINT_RK4
        m.opt.timestep=.000025
        after=ns['snapshot'](m)
        assert after==proof['after'] and np.all(m.actuator_dynprm[:,1]==.0004)
        assert (m.nv,m.nu,m.na)==(48,90,90)
        dump(out/'MODEL_PROOF.json',dict(before=before,after=after,exact_S154_before=True,exact_S154_after=True))
        # Native archive arrays are prepatch; compare every available model/opt field, with only declared patch transformed.
        native_checks={}
        for key,value in native.items():
            name=key.removeprefix('model.')
            obj=m.opt if key.startswith('opt.') else m
            name=key.split('.',1)[1] if '.' in key else name
            if hasattr(obj,name):
                expected=value.copy()
                if obj is m and name=='actuator_dynprm': expected[:,0]=.001
                if obj is m.opt and name=='integrator': expected=np.asarray(int(mj.mjtIntegrator.mjINT_RK4))
                if obj is m.opt and name=='timestep': expected=np.asarray(.000025)
                native_checks[key]=check(getattr(obj,name),expected,True)
        dump(out/'NATIVE_ARRAY_CHECKS.json',native_checks)
        assert native_checks and all(v['passed'] for v in native_checks.values())
        missing=[n for n in REQUIRED if n not in z]
        for n,count in (('mocap_pos',m.nmocap),('mocap_quat',m.nmocap),('eq_active',m.neq),('plugin_state',m.npluginstate),('userdata',m.nuserdata)):
            if count and n not in z: missing.append(n)
        dump(out/'STATE_SUFFICIENCY.json',dict(missing=missing,required=REQUIRED,optional=OPTIONAL,archive_inventory={k:list(v.shape) for k,v in z.items()}))
        if missing:
            finish('MISSING_STATE_STOP');return
        floors=set(meta['floor_geom_ids'])
        geomleg={int(k):LEGS.index(v) for k,v in meta['geom_map'].items()}
        for leg in LEGS: assert sum(e['leg']==leg for e in mapping)==15
        assert sorted(e['actuator'] for e in mapping)==list(range(90))
        assert all(mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,e['actuator'])==e['name'] for e in mapping)
        def forward(i,zero=False):
            d=mj.MjData(m)
            for name in REQUIRED:
                if name=='time': d.time=float(z[name][i])
                elif name=='native_qacc_warmstart': d.qacc_warmstart[:]=z[name][i]
                else: getattr(d,name)[:]=z[name][i]
            for name in OPTIONAL:
                if name in z: getattr(d,name)[:]=z[name][i]
            if zero: d.qvel[:]=0
            fields=('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied',*OPTIONAL)
            initial={n:np.asarray(getattr(d,n)).copy() for n in fields}
            initial['qacc_warmstart_input']=d.qacc_warmstart.copy()
            time=d.time
            reserve('zeroV_forward' if zero else 'dynamic_forward')
            mj.mj_forward(m,d)
            assert d.time==time and all(np.array_equal(getattr(d,n),initial[n]) for n in fields)
            c=ns['com_contact_record'](mj,m,d,floors)
            c.update(ns['constraint_record'](mj,m,d,i))
            c.update(ns['moment_record'](m,d));c.update(initial)
            for n in ('qacc','qfrc_actuator','qfrc_constraint','qfrc_bias','qfrc_passive','actuator_force','actuator_length','actuator_velocity','act_dot','qM','efc_R','efc_D','efc_aref','efc_b','efc_vel','efc_pos','efc_margin','efc_frictionloss'):
                c[n]=np.asarray(getattr(d,n)).copy()
            M=np.empty((48,48))
            mj.mj_fullM(m,M,d.qM)  # verified installed 3.9 binding; no speculative alternate calls
            c.update(M=M,Ma=M@d.qacc,qacc_warmstart_after=d.qacc_warmstart.copy())
            foot=np.zeros((6,3))
            for ci,(g1,g2) in enumerate(c['contact_geom_ids']):
                if g1 in floors or g2 in floors:
                    g=int(g2 if g1 in floors else g1)
                    if g in geomleg: foot[geomleg[g]]+=(1 if g1 in floors else -1)*c['contact_force_on_geom2_world'][ci]
            c['foot_force']=foot
            c['mass_residual']=M@d.qacc-(d.qfrc_actuator+d.qfrc_passive+d.qfrc_constraint+d.qfrc_applied-d.qfrc_bias)
            c['JTF_residual']=c['efc_J_dense'].T@c['efc_force']-c['qfrc_constraint']
            np.savez_compressed(out/f'{"ZERO_V" if zero else "DYNAMIC"}_{i:03d}.npz',**c)
            return c
        for i in POINTS:
            c=forward(i)
            checks={}
            for k in ('qacc','qfrc_actuator','qfrc_constraint','qfrc_bias','qfrc_passive','Ma','com_acceleration_world','foot_force','actuator_force','actuator_length','actuator_velocity','actuator_moment'):
                checks[k]=check(c[k],z[k][i])
            n=int(z['contact_ncon'][i]);prefix=f'sample{i:03d}_'
            for k in ('contact_force_raw6','contact_pos_world','contact_frame','contact_geom_ids','contact_dim'):
                checks[k]=check(c[k],z[k][i,:n],k in ('contact_geom_ids','contact_dim'))
            for k in ('efc_type','efc_id','efc_force','efc_J_dense','contact_efc_address'):
                checks[k]=check(c[k],rows[prefix+k],k in ('efc_type','efc_id','contact_efc_address'))
            for k in ('mass_residual','JTF_residual'):checks[k]=check(c[k],np.zeros_like(c[k]))
            gate=dict(sample=i,passed=all(v['passed'] for v in checks.values()),checks=checks)
            gates.append(gate);dump(out/'BASELINE_GATES.json',gates)
            if not gate['passed']:
                finish('DYNAMIC_BASELINE_FAILED_STOP_NO_ZERO_V');return
        # All three restored dynamic actual-state gates have passed. Exactly three zeroV forwards follow.
        static={i:forward(i,True) for i in POINTS}
        assert ledger['forward']==6
        assert np.all(m.actuator_gaintype==int(mj.mjtGain.mjGAIN_MUSCLE))
        assert np.all(m.actuator_biastype==int(mj.mjtBias.mjBIAS_MUSCLE))
        assert np.all(m.actuator_dyntype==int(mj.mjtDyn.mjDYN_MUSCLE))
        assert np.all(m.actuator_actnum==1) and np.array_equal(m.actuator_actadr,np.arange(90))
        assert not np.any(m.actuator_forcelimited) and not np.any(m.jnt_actfrclimited)
        if hasattr(m,'tendon_actfrclimited'):assert not np.any(m.tendon_actfrclimited)
        assert not np.any(m.actuator_actlimited), 'Unexpected activation clamp invalidates declared affine box'
        nodes=list(ET.parse(XML).getroot().find('actuator'))
        assert len(nodes)==90
        # evaluate uses explicit native gain/bias parameters; XML gain .5 is NOT a neural ctrl gain edit.
        for j,node in enumerate(nodes):
            assert node.tag=='general'
            for key in ('gainprm','biasprm','lengthrange'):
                xmlarr=np.asarray(list(map(float,node.get(key).split())))
                nativearr=getattr(m,'actuator_'+key)[j]
                if key!='lengthrange':xmlarr=np.pad(xmlarr,(0,10-len(xmlarr)))
                assert np.array_equal(xmlarr,nativearr), (j,key)
        dump(out/'AFFINITY_PROOF.json',dict(compiled_gain_muscle=True,compiled_bias_muscle=True,compiled_dyn_muscle=True,no_force_limits=True,
             actadr=list(map(int,m.actuator_actadr)),gainprm=m.actuator_gainprm.tolist(),biasprm=m.actuator_biasprm.tolist(),dynprm=m.actuator_dynprm.tolist(),
             ctrlrange=m.actuator_ctrlrange.tolist(),ctrllimited=m.actuator_ctrllimited.tolist(),neural_gain_half_is_ctrl_not_native_gain=True))
        activation_rows=[];residual_rows=[];contact_rows=[];results=[]
        for i,c in static.items():
            assert np.array_equal(c['qvel'],np.zeros(48)) and np.max(np.abs(c['actuator_velocity']))<=1e-12
            assert not np.any(c['qfrc_applied']) and not np.any(c['xfrc_applied'])
            assert len(c['contact_dim'])==18 and np.all(c['contact_dim']==3)
            assert c['efc_J_dense'].shape==(72,48) and np.all(c['efc_type']==6)
            assert np.all(c['contact_row_count']==4) and int(m.opt.cone)==int(mj.mjtCone.mjCONE_PYRAMIDAL)
            for ci,addr in enumerate(c['contact_efc_address']):
                assert np.array_equal(np.flatnonzero(c['efc_id']==ci),np.arange(addr,addr+4))
            gain=np.empty(90);bias=np.empty(90)
            for j,node in enumerate(nodes):_,gain[j],bias[j]=ns['evaluate'](c['actuator_length'][j],0.,0.,node)
            force=gain*c['act']+bias
            assert np.max(np.abs(force-c['actuator_force']))<=GATE_ATOL
            moment=c['actuator_moment'];J=c['efc_J_dense']
            assert np.max(np.abs(moment.T@force-c['qfrc_actuator']))<=GATE_ATOL
            B=moment.T*gain[None,:]
            base=moment.T@bias+c['qfrc_passive']-c['qfrc_bias']
            f0=np.maximum(c['efc_D']*c['efc_aref'],0.)
            np.savez_compressed(out/f'STATIC_AFFINE_{i:03d}.npz',B=B,base=base,gain=gain,bias=bias,moment=moment,J=J,M=c['M'],
                                actual_activation=c['act'],actual_ctrl=c['ctrl'],actual_force=force,f0=f0,desired_qacc=np.zeros(48),
                                forward_qacc_NOT_STATIC=c['qacc'],native_force_at_forward_NOT_f0=c['efc_force'],
                                actual_soft_static_residual=B@c['act']+base+J.T@f0,
                                native_forward_residual=B@c['act']+base+J.T@c['efc_force']-c['M']@c['qacc'])
            for upper in (.5,1.):
                for rigid in (False,True):
                    kind='rigid_upper' if rigid else 'soft'
                    label=f'{kind}_{i:03d}_upper{upper:.1f}'
                    A=np.column_stack((B,J.T)) if rigid else B
                    b=-base if rigid else -base-J.T@f0
                    info,x,residual=solve_lp(out/label,A,b,upper,rigid,linprog)
                    results.append(dict(sample=i,kind=kind,upper=upper,**info))
                    possible_control=False
                    if info['feasible_absolute_native']:
                        act=x[:90]
                        ctrl=act.copy()
                        clamped=np.where(m.actuator_ctrllimited,np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]),ctrl)
                        # MuJoCo muscle dynamics clamps ctrl to [0,1]; ctrl==act implies exactly zero numerator.
                        clamped=np.clip(clamped,0.,1.)
                        rate=(clamped-act)/np.where(clamped>act,m.actuator_dynprm[:,0]*(.5+1.5*act),m.actuator_dynprm[:,1]/(.5+1.5*act))
                        possible_control=bool(np.array_equal(ctrl,clamped) and np.array_equal(rate,np.zeros(90)))
                        np.savez_compressed(out/(label+'_POSSIBLE_CONTROL_NOT_APPLIED.npz'),ctrl=ctrl,activation=act,clamped_ctrl=clamped,offline_act_dot=rate)
                        for entry in mapping:
                            j=entry['actuator']
                            activation_rows.append(dict(sample=i,kind=kind,upper=upper,leg=entry['leg'],actuator=j,name=entry['name'],actual_activation=float(c['act'][j]),feasible_activation=float(act[j]),possible_ctrl=float(ctrl[j]),offline_act_dot=float(rate[j]),possible_control=possible_control))
                    for r in range(48):
                        residual_rows.append(dict(sample=i,kind=kind,upper=upper,dof=r,actual_forward_qacc=float(c['qacc'][r]),desired_qacc=0.,
                          actual_forward_Ma=float((c['M']@c['qacc'])[r]),actual_soft_static_residual=float((B@c['act']+base+J.T@f0)[r]),
                          solution_static_residual=None if not info['feasible_absolute_native'] else float(residual[r])))
                    forces=[('native_forward_NOT_static',c['efc_force']),('soft_qacc0_f0',f0)]
                    if rigid and info['feasible_absolute_native']:forces.append(('rigid_LP_lambda',x[90:]))
                    for force_kind,edges in forces:
                        for ci,addr in enumerate(c['contact_efc_address']):
                            e=edges[addr:addr+4];mu=c['contact_friction'][ci,:2]
                            assert np.all(mu>0)
                            local=np.array([e.sum(),mu[0]*(e[0]-e[1]),mu[1]*(e[2]-e[3])])
                            world=c['contact_frame'][ci].T@local
                            g1,g2=map(int,c['contact_geom_ids'][ci]);assert (g1 in floors)!=(g2 in floors)
                            sign=1 if g1 in floors else -1
                            if force_kind=='native_forward_NOT_static':assert np.max(np.abs(local-c['contact_force_raw6'][ci,:3]))<=GATE_ATOL
                            margin=local[0]-np.sum(np.abs(local[1:]/mu))
                            contact_rows.append(dict(sample=i,kind=kind,upper=upper,force_kind=force_kind,contact=ci,geom1=g1,geom2=g2,
                                floor_side=1 if g1 in floors else 2,force_on_fly_sign=sign,Fn=float(local[0]),Ft1=float(local[1]),Ft2=float(local[2]),
                                mu1=float(mu[0]),mu2=float(mu[1]),pyramid_margin=float(margin),min_edge=float(e.min()),
                                fly_world_x=float(sign*world[0]),fly_world_y=float(sign*world[1]),fly_world_z=float(sign*world[2])))
        dump(out/'LP_SUMMARY.json',results)
        if activation_rows:csvout(out/'ACTIVATION_PER_LEG_15.csv',activation_rows)
        csvout(out/'RESIDUAL_ALL48.csv',residual_rows)
        csvout(out/'CONTACT_PYRAMID_DECODE.csv',contact_rows)
        finish('COMPLETED_DIAGNOSTIC_ONLY')
    except Exception:
        dump(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger,no_retry=True))
        finish('FAILED_STOP_NO_RETRY')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    main(args.output or BASE/'s165-static-support-corrected'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
