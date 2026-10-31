"""S169/P1-S2 bounded near-initial native-SOFT support search; parent review/run ONLY.
No integration, neural calls, controller deployment, model simplification or retries.
Every hypothetical diagnostic forward (including baseline and finite differences)
is charged against the immutable 200-call cap. An unaccepted best is NOT a candidate.
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
S154 = BASE/'s154-activation-rate/20261024T215751775489Z'
S165 = BASE/'s165-static-support-corrected/20261024T231400000000Z'
S166 = BASE/'s166-mapped-support/20261024T231418078692647Z'
MAP = BASE/'s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json'
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP = 200
STATE = ('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied')
OPTIONAL = ('mocap_pos','mocap_quat','eq_active','plugin_state','userdata')
LEGS = ('lf','lm','lh','rf','rm','rh')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def extract(p, names):
    tree = ast.parse(p.read_text())
    return '\n'.join(ast.unparse(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)) for name in names)+'\n'


class BudgetStop(Exception):
    pass


class AcceptedStop(Exception):
    pass


def main(out):
    assert out.is_absolute() and not out.exists(), 'Fresh absolute timestamp only; no retry/resume'
    out.mkdir(parents=True)
    freeze = out/'SOURCE_FREEZE'; freeze.mkdir()
    ledger = dict(model_load=0, forward=0, baseline_forward=0, probe_forward=0, optimized_state_forward=0,
                  pose_evaluations=0, constraintUpdate=0, step=0, neural_ticks=0)
    hashes = {}; records = []; best = None; accepted = None
    def reserve(kind):
        if kind != 'model_load' and ledger['forward'] >= CAP:
            raise BudgetStop()
        ledger[kind] += 1
        if kind != 'model_load': ledger['forward'] += 1
        assert ledger['model_load'] <= 1 and ledger['forward'] <= CAP
        dump(out/'CALL_LEDGER.json', ledger)
    def finish(status):
        after = {p:sha(p) for p in hashes}
        dump(out/'INPUT_HASHES_AFTER.json', after)
        unchanged = after == hashes
        dump(out/'SUMMARY.json', dict(status=status, ledger=ledger, accepted=accepted,
             best_diagnostic=best, inputs_unchanged=unchanged, units='native; SI NOT_ESTABLISHED',
             trajectory=False, integration=False, neural_controller=False, deployment=False,
             limitation='66 constant pools are an interface upper bound, not proven neural/FIR reachable; no standing/walking claim'))
        if records:
            with (out/'EVALUATIONS.csv').open('w', newline='') as f:
                w=csv.DictWriter(f, fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        dump(out/'OUTPUT_HASHES.json', {str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='OUTPUT_HASHES.json'})
        assert unchanged, 'Input changed: preserve failure, no retry'
    try:
        specs=[(ROOT/'run_matched_state_counterfactual.py', ('snapshot',)),
               (ROOT/'run_local_pool_response.py', ('moment_record',)),
               (ROOT/'run_constraint_row_recording.py', ('constraint_record',)),
               (ROOT/'verify_muscle_reconstruction.py', ('fl','evaluate'))]
        files=[Path(__file__).resolve(), XML, MAP, S154/'INPUT_HASHES.json',
               S154/'init80ms/TRAJECTORY.npz', S154/'init80ms/MODEL_PROVENANCE.json',
               S154/'init80ms/NATIVE_MODEL_ARRAYS.npz', S154/'init80ms/COM_CONTACT_METADATA.json',
               S165/'STATIC_AFFINE_000.npz', S165/'ZERO_V_000.npz', S166/'MAPPING.npz']+[p for p,_ in specs]
        old=json.loads((S154/'INPUT_HASHES.json').read_text()); old=old.get('hashes',old)
        for p,h in old.items():
            if isinstance(h,str) and len(h)==64:
                assert sha(p)==h, 'S154 binding changed: '+p
                hashes[p]=h
        hashes.update({str(p):sha(p) for p in files})
        dump(out/'INPUT_HASHES.json',hashes)
        manifest={}
        for i,(p,h) in enumerate(sorted(hashes.items())):
            target=freeze/f'{i:04d}_{Path(p).name}';shutil.copyfile(p,target);assert sha(target)==h
            manifest[p]=dict(sha256=h,copy=str(target))
        dump(out/'SOURCE_FREEZE_MANIFEST.json',manifest)
        text='\n'.join(extract(p,names) for p,names in specs)
        (freeze/'INERT_FUNCTIONS.py').write_text(text)
        ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
        exec(compile(text,str(freeze/'INERT_FUNCTIONS.py'),'exec'),ns)
        z=load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy()
        native=load(S154/'init80ms/NATIVE_MODEL_ARRAYS.npz')
        prior=load(S165/'STATIC_AFFINE_000.npz'); priorzero=load(S165/'ZERO_V_000.npz')
        proof=json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
        meta=json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
        mapping=json.loads(MAP.read_text()); T=load(S166/'MAPPING.npz')['T']
        assert T.shape==(90,66) and np.all(np.isin(T,[0.,.5]))
        assert np.count_nonzero(T)==74 and np.sum(np.all(T==0,axis=1))==16
        assert np.all(np.count_nonzero(T,axis=1)<=1) and np.all(np.count_nonzero(T,axis=0)>0)
        assert np.all(z['act'][0]==0) and np.all(z['ctrl'][0]==0)
        tree=ET.parse(XML); nodes=list(tree.getroot().find('actuator'))
        joints=[n for n in tree.getroot().find('worldbody').iter('joint')]
        assert joints[0].get('type')=='free' and len(joints)==43 and len(q0)==49
        names=sorted({name for e in mapping for name in e['referenced_crossed_joints']})
        assert set(names)=={j.get('name') for j in joints[1:]}, 'Use exact original active leg-joint union, no guessed subset'
        indices=np.r_[2,np.arange(7,49)].astype(int)
        ranges=np.array([list(map(float,j.get('range').split())) for j in joints[1:]])
        radius=np.minimum(.15,.1*np.diff(ranges,axis=1).ravel())
        lower=np.r_[q0[2]*(1-.05), np.maximum(q0[7:]-radius,ranges[:,0])]
        upper=np.r_[q0[2]*(1+.15), np.minimum(q0[7:]+radius,ranges[:,1])]
        assert np.all(lower<upper) and np.all(q0[indices]>=lower) and np.all(q0[indices]<=upper)
        width=upper-lower; x0=(q0[indices]-lower)/width
        scale=np.maximum(np.max(abs(prior['B']@T),axis=1)+abs(prior['base'])+abs(prior['J'].T@prior['f0']),1e-12)
        footdepth=float(np.maximum(-priorzero['contact_dist'],0).max(initial=0))
        np.savez_compressed(out/'FROZEN_SEARCH.npz',qpos0=q0,indices=indices,lower=lower,upper=upper,T=T,row_scale=scale,
                            joint_ranges=ranges,original_foot_depth=footdepth)
        dump(out/'PREREG.json',dict(scope='S169/P1-S2 bounded near-initial native SOFT offline diagnostic',
             model=dict(xml=str(XML),mujoco='3.9.0',tau_act=.001,tau_deact=.0004,integrator='RK4',dt=.000025),
             start='one S154 release index0; zero pools; no multistart', variables='rootheight + exact42 active leg joints +66 pools =109; variable projection, not90 independent actuators',
             fixed='rootxy/orientation and every nonleg qpos; full head/thorax/abdomen/wings retained',
             bounds='rootz [0.95*z0,1.15*z0]; each leg q0 +/- min(.15rad,.10*native range) intersect native bounds; never widen',
             bounds_values=dict(lower=lower.tolist(),upper=upper.tolist()),pool_bounds=[0.,1.],T_gain=.5,missing16_zero=True,
             method='one bounded scipy least_squares outer pose start; finite difference jac; inner lsq_linear66 box; each pose two actual forwards (probe0 then optimizedact); hard reserve before EVERY forward',
             maximum_actual_forwards=CAP,baseline_forward_included=True,maximum_pose_evaluations=99,
             row_scaling='frozen S165 perDOF maxabs(B*T)+abs(base)+abs(J.T*f0), floor1e-12',
             objective='normalized all48 static residual + geometry penalties; tiny1e-8 normalized pose displacement secondary',
             native_soft='zeroV, desiredqacc0: equality f0=D*aref, unilateral limits/frictionless/pyramid f0=max(D*aref,0); no free contact variables; actual forward qacc never overwritten',
             supported_row_types=[0,3,4,5,6],unsupported='stop immediately, preserve raw witness; no elliptic/frictionloss assumption',
             accept=dict(max_abs_static_residual=1e-8,max_abs_forward_dynamic_closure=1e-8,
                         max_abs_normalized_forward_qacc=1e-6,qacc_scale='9801 root translation;9801/z0 all otherDOFs',
                         foot_depth_cap=footdepth,nonfoot_depth_cap=1e-12,minimum_contact_legs=3,
                         friction_margin_min=-1e-10,joint_bound_atol=1e-8,act_dot_exact_zero=True),
             no_accept='SEARCH_FAILED; save BEST_DIAGNOSTIC_NOT_CANDIDATE; no extra verification forward/reload',
             no_step=True,no_neural=True,no_deployment=True,no_restarts=True,units='native; SI NOT_ESTABLISHED'))
        # Only after immutable preregistration/freeze may native engine import/load occur.
        import mujoco as mj
        from scipy.optimize import least_squares, lsq_linear
        assert mj.__version__=='3.9.0'
        for name in ('control','act_dyn','act_gain','act_bias','passive','sensor'):
            getter=getattr(mj,'get_mjcb_'+name,None);assert getter is None or getter() is None
        reserve('model_load');m=mj.MjModel.from_xml_path(str(XML))
        before=ns['snapshot'](m);assert before==proof['before']
        m.actuator_dynprm[:,0]=.001;m.actuator_dynprm[:,1]=.0004
        m.opt.integrator=mj.mjtIntegrator.mjINT_RK4;m.opt.timestep=.000025
        after=ns['snapshot'](m);assert after==proof['after']
        native_checks={}
        for key,value in native.items():
            name=key.removeprefix('model.');obj=m.opt if key.startswith('opt.') else m
            name=key.split('.',1)[1] if '.' in key else name
            if hasattr(obj,name):
                expected=value.copy()
                if obj is m and name=='actuator_dynprm':expected[:,0]=.001;expected[:,1]=.0004
                if obj is m.opt and name=='integrator':expected=np.asarray(int(mj.mjtIntegrator.mjINT_RK4))
                if obj is m.opt and name=='timestep':expected=np.asarray(.000025)
                native_checks[key]=bool(np.array_equal(getattr(obj,name),expected))
        dump(out/'NATIVE_ARRAY_CHECKS.json',native_checks)
        assert native_checks and all(native_checks.values())
        dump(out/'MODEL_PROOF.json',dict(before=before,after=after,exact_S154=True))
        assert (m.nv,m.nu,m.na,m.nq)==(48,90,90,49)
        assert int(m.opt.cone)==int(mj.mjtCone.mjCONE_PYRAMIDAL)
        assert np.array_equal(m.jnt_qposadr[1:],np.arange(7,49)) and np.array_equal(m.jnt_range[1:],ranges)
        assert [mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,i) for i in range(43)]==[j.get('name') for j in joints]
        assert np.all(m.actuator_actearly==0) and np.array_equal(m.actuator_actadr,np.arange(90))
        assert np.all(m.actuator_gaintype==int(mj.mjtGain.mjGAIN_MUSCLE)) and np.all(m.actuator_biastype==int(mj.mjtBias.mjBIAS_MUSCLE))
        assert np.all(m.actuator_dyntype==int(mj.mjtDyn.mjDYN_MUSCLE)) and np.all(m.actuator_actnum==1)
        assert not np.any(m.actuator_forcelimited) and not np.any(m.jnt_actfrclimited) and not np.any(m.actuator_actlimited)
        if hasattr(m,'tendon_actfrclimited'):assert not np.any(m.tendon_actfrclimited)
        for j,node in enumerate(nodes):
            for key in ('gainprm','biasprm','lengthrange'):
                a=np.array(list(map(float,node.get(key).split())))
                if key!='lengthrange':a=np.pad(a,(0,10-len(a)))
                assert np.array_equal(a,getattr(m,'actuator_'+key)[j])
        dump(out/'BODY_STRUCTURE.json',dict(body_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i) for i in range(m.nbody)],
             body_parentid=m.body_parentid.tolist(),geom_bodyid=m.geom_bodyid.tolist(),geom_type=m.geom_type.tolist(),
             geom_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)],
             geom_rgba=m.geom_rgba.tolist(),original_xml_unmodified=True,rendering_not_performed=True))
        floors=set(meta['floor_geom_ids']);geomleg={int(k):LEGS.index(v) for k,v in meta['geom_map'].items()}
        def forward(q,pool,kind,original=False):
            d=mj.MjData(m);d.time=float(z['time'][0])
            for n in STATE: getattr(d,n)[:]=z[n][0]
            for n in OPTIONAL:
                if n in z:getattr(d,n)[:]=z[n][0]
                else:assert np.asarray(getattr(d,n)).size==0, 'Missing required native state '+n
            d.qacc_warmstart[:]=z['native_qacc_warmstart'][0]
            if not original:
                d.qpos[:]=q;d.qvel[:]=0;d.act[:]=T@pool;d.ctrl[:]=d.act
                d.qacc_warmstart[:]=0
            initial={n:np.asarray(getattr(d,n)).copy() for n in (*STATE,*OPTIONAL)}
            initial['qacc_warmstart_input']=d.qacc_warmstart.copy()
            reserve(kind);mj.mj_forward(m,d)
            assert all(np.array_equal(getattr(d,n),initial[n]) for n in (*STATE,*OPTIONAL))
            c=ns['constraint_record'](mj,m,d,ledger['forward']);c.update(ns['moment_record'](m,d));c.update(initial)
            for n in ('qacc','act_dot','actuator_force','actuator_length','actuator_velocity','qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_constraint','efc_D','efc_R','efc_aref','efc_b','efc_pos','efc_vel','efc_margin','efc_frictionloss','xpos','xquat','geom_xpos','geom_xmat','qM'):
                c[n]=np.asarray(getattr(d,n)).copy()
            c['contact_dist']=np.array([v.dist for v in d.contact]);c['contact_friction']=np.array([v.friction.copy() for v in d.contact]).reshape(-1,5)
            M=np.empty((48,48));mj.mj_fullM(m,M,d.qM);c['M']=M
            c['dynamic_closure']=M@c['qacc']-(c['qfrc_actuator']+c['qfrc_passive']+c['qfrc_constraint']-c['qfrc_bias'])
            np.savez_compressed(out/f'FORWARD_{ledger["forward"]:03d}_{kind}.npz',**c)
            return c
        baseline=forward(q0,np.zeros(66),'baseline_forward',True)
        baseline_checks={k:float(abs(baseline[k]-z[k][0]).max(initial=0)) for k in ('qacc','qfrc_actuator','qfrc_passive','qfrc_bias','qfrc_constraint','actuator_force','actuator_length','actuator_velocity','actuator_moment')}
        dump(out/'BASELINE_GATE.json',baseline_checks);assert max(baseline_checks.values())<=1e-9
        assert not np.any(baseline['qfrc_applied']) and not np.any(baseline['xfrc_applied'])
        qscale=np.r_[np.full(3,9801.),np.full(45,9801./q0[2])]
        def affine(c):
            types=c['efc_type'];assert np.all(np.isin(types,[0,3,4,5,6])), 'Unsupported native row type'
            assert not np.any(c['efc_frictionloss'])
            f0=c['efc_D']*c['efc_aref'];f0=np.where(types==0,f0,np.maximum(f0,0.))
            gain=np.empty(90);bias=np.empty(90)
            for j,node in enumerate(nodes):_,gain[j],bias[j]=ns['evaluate'](c['actuator_length'][j],0.,0.,node)
            moment=c['actuator_moment'];B=moment.T*gain[None,:];base=moment.T@bias+c['qfrc_passive']-c['qfrc_bias']
            assert abs(gain*c['act']+bias-c['actuator_force']).max()<=1e-9
            assert abs(moment.T@c['actuator_force']-c['qfrc_actuator']).max()<=1e-9
            return B,base,f0,gain,bias
        def geometry(c,f0):
            depth=0.;nonfoot=0.;bearing=set();margins=[];loads=np.zeros((6,3))
            for ci,(g1,g2) in enumerate(c['contact_geom']):
                g1=int(g1);g2=int(g2);leg=geomleg.get(g2 if g1 in floors else g1) if ((g1 in floors)!=(g2 in floors)) else None
                pen=max(0.,-float(c['contact_dist'][ci]));addr=int(c['contact_efc_address'][ci]);dim=int(c['contact_dim'][ci])
                if leg is None:nonfoot=max(nonfoot,pen)
                else:depth=max(depth,pen)
                if addr<0:continue
                count=int(c['contact_row_count'][ci])
                if dim==1:
                    assert count==1;local=np.r_[f0[addr],0.,0.];margin=f0[addr]
                else:
                    assert dim==3 and count==4, 'Only native frictionless or condim3 pyramids supported'
                    e=f0[addr:addr+4];mu=c['contact_friction'][ci,:2];assert np.all(mu>0)
                    local=np.r_[e.sum(),mu[0]*(e[0]-e[1]),mu[1]*(e[2]-e[3])]
                    margin=local[0]-sum(abs(local[1:]/mu))
                margins.append(float(margin))
                if leg is not None:
                    loads[leg]+=(1 if g1 in floors else -1)*c['contact_frame'][ci].T@local
                    if local[0]>0 and c['contact_dist'][ci]<=0:bearing.add(leg)
            return depth,nonfoot,len(bearing),min(margins,default=0.),loads,np.array(margins)
        def objective(x):
            nonlocal best,accepted
            # Atomic paired pose evaluation; never leave a probe without optimized-state validation.
            if ledger['forward']+2>CAP:raise BudgetStop()
            q=q0.copy();q[indices]=lower+x*width
            probe=forward(q,np.zeros(66),'probe_forward');B,base,f0,gain,bias=affine(probe)
            A=B@T;b=-base-probe['efc_J_dense'].T@f0
            inner=lsq_linear(A/scale[:,None],b/scale,bounds=(0.,1.),tol=1e-12,max_iter=300,lsq_solver='exact')
            pools=inner.x
            assert np.isfinite(pools).all() and np.all(pools>=0.) and np.all(pools<=1.)
            assert np.all(T@pools>=0.) and np.all(T@pools<=.5)
            actual=forward(q,pools,'optimized_state_forward')
            B2,base2,f02,_,_=affine(actual)
            assert np.max(abs(B-B2))<=1e-10 and np.max(abs(base-base2))<=1e-10 and np.array_equal(f0,f02)
            res=B2@(T@pools)+base2+actual['efc_J_dense'].T@f02
            depth,nonfoot,nlegs,margin,loads,margins=geometry(actual,f02)
            penalties=np.array([max(depth-footdepth,0.)/q0[2],nonfoot/q0[2],max(3-nlegs,0.),max(-margin,0.)])
            vector=np.r_[res/scale,penalties,1e-8*(x-x0)]
            scalar=float(vector@vector);maxres=float(abs(res).max());qaccnorm=float(abs(actual['qacc']/qscale).max())
            limits=bool(np.all(q[7:]>=ranges[:,0]-1e-8) and np.all(q[7:]<=ranges[:,1]+1e-8))
            clamped=np.where(m.actuator_ctrllimited,np.clip(actual['ctrl'],m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]),actual['ctrl'])
            controlok=bool(np.array_equal(clamped,actual['ctrl']) and np.array_equal(actual['ctrl'],actual['act']) and np.all(actual['act_dot']==0))
            passed=bool(maxres<=1e-8 and abs(actual['dynamic_closure']).max()<=1e-8 and qaccnorm<=1e-6 and depth<=footdepth+1e-12 and nonfoot<=1e-12 and nlegs>=3 and margin>=-1e-10 and limits and controlok)
            ledger['pose_evaluations']+=1;eid=ledger['pose_evaluations']
            witness=out/f'EVAL_{eid:03d}.npz'
            saved=dict(actual)
            saved.update(qpos=q,pools=pools,act=T@pools,ctrl=actual['ctrl'],T=T,B=B2,base=base2,J=actual['efc_J_dense'],
                 f0=f02,gain=gain,bias=bias,residual48=res,row_scale=scale,objective_vector=vector,foot_load=loads,pyramid_margins=margins,
                 actual_forward_qacc=actual['qacc'],actual_forward_efc_force=actual['efc_force'],probe_forward_qacc=probe['qacc'],probe_act=probe['act'],
                 actual_forward_id=ledger['forward'],probe_forward_id=ledger['forward']-1,inner_status=inner.status,inner_optimality=inner.optimality)
            np.savez_compressed(witness,**saved)
            if eid==1:np.savez_compressed(out/'ORIGINAL_POSE_OPTIMIZED_POOLS_DIAGNOSTIC.npz',**saved)
            geometry_ok=bool(depth<=footdepth+1e-12 and nonfoot<=1e-12 and nlegs>=3 and margin>=-1e-10 and limits and controlok)
            row=dict(evaluation=eid,witness=str(witness),objective=scalar,max_abs_native_residual=maxres,normalized_qacc=qaccnorm,
                     foot_depth=depth,nonfoot_depth=nonfoot,contact_legs=nlegs,friction_margin=margin,control_ok=controlok,geometry_ok=geometry_ok,accepted=passed,forwards=ledger['forward'])
            records.append(row);dump(out/'EVALUATIONS.json',records);dump(out/'CALL_LEDGER.json',ledger)
            rank=(not geometry_ok,maxres,scalar)
            if best is None or rank<(not best['geometry_ok'],best['max_abs_native_residual'],best['objective']):
                best=row.copy();shutil.copyfile(witness,out/'BEST_DIAGNOSTIC_NOT_CANDIDATE.npz');dump(out/'BEST_DIAGNOSTIC_NOT_CANDIDATE.json',best)
            if passed:
                accepted=row.copy();shutil.copyfile(witness,out/'ACCEPTED_CANDIDATE.npz');dump(out/'ACCEPTED_CANDIDATE.json',accepted)
                raise AcceptedStop()
            return vector
        try:
            result=least_squares(objective,x0,bounds=(np.zeros(43),np.ones(43)),jac='2-point',diff_step=1e-5,
                                 max_nfev=200,ftol=1e-12,xtol=1e-12,gtol=1e-12,x_scale='jac')
            dump(out/'OPTIMIZER_TERMINATION.json',dict(status=int(result.status),message=result.message,nfev=int(result.nfev),
                  njev=None if result.njev is None else int(result.njev),actual_forward_count=ledger['forward']))
        except BudgetStop:
            dump(out/'OPTIMIZER_TERMINATION.json',dict(reason='ACTUAL_FORWARD_HARD_CAP_GRACEFUL_STOP',actual_forward_count=ledger['forward']))
        except AcceptedStop:
            dump(out/'OPTIMIZER_TERMINATION.json',dict(reason='FIRST_FULLY_ACCEPTED_DIAGNOSTIC',actual_forward_count=ledger['forward']))
        finish('ACCEPTED_STATIC_DIAGNOSTIC_ONLY' if accepted else 'SEARCH_FAILED')
    except Exception:
        dump(out/'FAILED.json',dict(traceback=traceback.format_exc(),ledger=ledger,no_retry=True))
        finish('FAILED_STOP_NO_RETRY')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    main(args.output or BASE/'s169-near-initial-support'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
