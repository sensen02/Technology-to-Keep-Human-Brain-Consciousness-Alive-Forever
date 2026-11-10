"""V2 offline swing geometry: free airborne orientation, exact stance knots.

This changes the declared airborne constraint, not the previous failed evidence.
No physics/neural integration; no accepted GAIT export unless every dense gate passes.
"""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.sparse import diags, eye, vstack
from scipy.spatial.transform import Rotation

OLD = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T094930Z/prepare_swing.py')
spec = importlib.util.spec_from_file_location('frozen_swing_v1', OLD)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
mj = b.mj
OUT = b.OUT.parent/'20261107T095519Z'
b.OUT = OUT
LEGS = b.LEGS


class Preparation(b.Preparation):
    def __init__(self):
        super().__init__()
        self.nonfoot = [g for g in range(self.m.ngeom) if not self.p.planes[g] and self.p.feet[g]<0
                       and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])]
        source_soles = []
        for q in self.old:
            self.fk(q); source_soles.append(self.soles())
        self.source_soles = np.asarray(source_soles)
        self.source_medians = np.array([np.median(self.source_soles[self.stance[:,j],j]) for j in range(6)])
        self.common_plane = float(self.source_medians.max())

    def ik(self, phase, leg, previous=None, seed=None, filtered_prior=None, max_nfev=100):
        original, goal, orientation, stance = self.target(phase, leg)
        ix = self.ids[leg]; qa = self.p.qa[ix]; va = self.p.va[ix]
        if previous is None: previous = original[ix]
        if seed is None: seed = original[ix]
        cache = {}
        def evaluate(q):
            if 'q' in cache and np.array_equal(cache['q'],q): return cache['r'],cache['J']
            full = original.copy(); full[ix] = q; self.fk(full)
            point = self.d.xpos[self.p.bids[leg]]
            jp = np.zeros((3,self.m.nv)); mj.mj_jac(self.m,self.d,jp,None,point,self.p.bids[leg])
            pieces = [point-goal,.001*(q-original[ix]),.01*(q-previous)]
            blocks = [jp[:,va],.001*np.eye(7),.01*np.eye(7)]
            if filtered_prior is not None:
                pieces.append(.01*(q-filtered_prior)); blocks.append(.01*np.eye(7))
            residual = np.concatenate(pieces); jac = np.vstack(blocks)
            cache.update(q=q.copy(),r=residual,J=jac); return residual,jac
        fit = least_squares(lambda q:evaluate(q)[0],np.clip(seed,self.lo[ix]+1e-12,self.hi[ix]-1e-12),
            jac=lambda q:evaluate(q)[1],bounds=(self.lo[ix],self.hi[ix]),max_nfev=max_nfev,
            ftol=1e-10,xtol=1e-10,gtol=1e-9,x_scale='jac')
        error,_ = evaluate(fit.x)
        full = original.copy(); full[ix] = fit.x; self.fk(full)
        actual_orientation = self.d.xmat[self.p.bids[leg]].reshape(3,3)
        orientation_change = float(np.linalg.norm(Rotation.from_matrix(orientation.T@actual_orientation).as_rotvec()))
        clearance = float(self.soles()[leg]-self.common_plane)
        distances = [(mj.mj_geomDistance(self.m,self.d,g,self.plane,1000.,self.fromto),g) for g in self.nonfoot]
        nearest = min(distances) if distances else (1000.,-1)
        self_contacts = [dict(geom1=self.m.geom(int(c.geom1)).name,geom2=self.m.geom(int(c.geom2)).name,distance=float(c.dist))
                         for c in self.d.contact if not(self.p.planes[c.geom1] or self.p.planes[c.geom2]) and c.dist<0]
        v = float((self.local_phase(phase)[leg]-self.duty)/(1-self.duty))
        plateau = .25-1e-10<=v<=.75+1e-10
        row = dict(leg=LEGS[leg],phase_rad=float(phase),swing_v=v,solver_success=bool(fit.success),
            status=int(fit.status),nfev=int(fit.nfev),position_error=float(np.linalg.norm(error[:3])),
            orientation_error=orientation_change,air_orientation_gated=False,target=goal,
            target_orientation=orientation,achieved_orientation=actual_orientation.copy(),
            achieved=self.d.xpos[self.p.bids[leg]].copy(),solved_joints=fit.x.copy(),
            source_joints=original[ix],previous_joints=np.asarray(previous).copy(),joint_names=self.names[ix],
            joint_lower=self.lo[ix],joint_upper=self.hi[ix],active_joint_bounds=fit.active_mask,
            wholefoot_clearance_above_common_plane=clearance,plateau=plateau,
            nearest_nonfoot_ground_distance=float(nearest[0]),nearest_nonfoot_geom=self.m.geom(nearest[1]).name if nearest[1]>=0 else None,
            actual_FK_self_penetrating_contacts=self_contacts,
            passed=bool(np.linalg.norm(error[:3])<=.005 and (not plateau or clearance>=.1)))
        return fit.x.copy(),row

    def evaluate(self,qknots):
        report,evidence = super().evaluate(qknots)
        dense = evidence['phase']; stance = evidence['nominal_stance']
        u = self.local_phase(dense); v = (u-self.duty)/(1-self.duty)
        plateau = (~stance)&(v>=.25-1e-10)&(v<=.75+1e-10)
        source_soles = []
        for q in evidence['source_qpos_hinges']:
            self.fk(q); source_soles.append(self.soles())
        source_soles = np.asarray(source_soles)
        clear = evidence['wholefoot_sole_distance']-self.common_plane
        deviation = abs(evidence['wholefoot_sole_distance']-source_soles)
        oe = evidence['orientation_error_rad']; pe = evidence['position_error']
        report['checks'].pop('orientation')
        report['checks'].update(plateau_wholefoot_clearance=bool(clear[plateau].min()>=.1),
            stance_point_position=bool(pe[stance].max()<=.005),
            stance_point_sole_distance=bool(deviation[stance].max()<=.005),
            stance_orientation=bool(oe[stance].max()<=.03))
        report.update(passed=bool(all(report['checks'].values())),common_stance_plane=self.common_plane,
            plateau_min_clearance=float(clear[plateau].min()),
            per_leg_plateau_min_clearance=[float(clear[plateau[:,j],j].min()) for j in range(6)],
            stance_point_position_max=float(pe[stance].max()),stance_point_sole_distance_max=float(deviation[stance].max()),
            stance_orientation_max=float(oe[stance].max()),air_orientation_change_max=float(oe[~stance].max()),
            per_leg_air_orientation_change_max=[float(oe[~stance[:,j],j].max()) for j in range(6)])
        evidence.update(source_wholefoot_sole_distance=source_soles,plateau=plateau,
            clearance_above_common_plane=clear,stance_point_sole_distance_deviation=deviation)
        return report,evidence


def main():
    def forbidden(*a,**kw): raise AssertionError('No mj_step in offline free-swing preparation')
    mj.mj_step=forbidden
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists(): raise FileExistsError('Preserve prior candidate '+str(OUT))
    files=[Path(__file__),OLD,b.SOURCE/'GAIT.npz',b.SOURCE/'METADATA.json',b.WAVE,b.UTIL,
        Path(b.util.v2.__file__),Path(b.util.v1.__file__),b.BASE/'plant_probe.py',b.BASE/'incremental_actuator.py',
        b.ROOT/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py',b.DEFAULT_MODEL_PATH]
    hashes={str(p):b.sha(p) for p in files}
    prereg=dict(stage='S5 free-air-orientation lift-first swing geometry only',source_hashes=hashes,
        prior_failure='094930 retains9/18failedsame-phaseorientationkeyframes. This new candidate explicitly changes airborne orientation freedom; it does not amend old failure.',
        source=str(b.SOURCE/'GAIT.npz'),unchanged_geometry='duty5/6,phaseoffsets,513knots,XYsourcewarpS5((v-.2)/.6),.25peakZwithplateau.25-.75,samequietstanceZ,exactsourceSTANCEknots/boundarywithin1e-10',
        constraint_change='Airborne foot orientation FREE: omit orientation cost/gate in swing, report every resulting rotation vs same-phase source. Stance geometry/orientation remains source.',
        justification='Airborne orientation is not a ground support constraint. Actual wholefoot collision clearance, rather than originheight alone, must pass.',
        keyframes=dict(v=[.25,.5,.75],max_nfev=100,position_tolerance=.005,
            source_joint_prior=.001,previous_phase_continuity=.01,clearance_above_common_stance_plane=.1,
            seed='same-phase source, previous accepted keyframe as continuityprior; first continuityprior issource',
            stop='anyposition/clearance/nativeboundfailure stops fullcurve; no lift/threshold/rate changes'),
        fullcurve='Onlyswing boundedIKmax100; perlegprevious5msphase solutioncontinuity .01, sourceprior .001. Fixedlambda100 D2 periodic smoothing ofsource-relativecorrection, hardstanceδ0. Atmostone localfilteredprior.01 reprojectionmax30 withbothsource/previousphasepriorsretained, onlyifgeometryfails.',
        dense=dict(phases=4097,position=.005,plateau_wholefoot_clearance=.1,stance_sole_median_spread=.005,
            stance_point_position_deviation=.005,stance_point_sole_distance_deviation=.005,
            stance_orientation=.03,native_bound_tolerance=1e-9,rate_1Hz=30.,derivative_seam=1e-10),
        interpolation='sameperiodictwo-neighborPCHIP,no clipping. Exactstanceknotscanstillhaveadjacentinterpolatedstancechanges; checkedpointwise.',
        boundary_disclosure='Rawwarpzeroendvelocity differs frombackwardstancevelocity; rawtargetnotC1. SourcequietstanceZmedianmaydifferfromboundaryZ; nohiddenboundaryrepair.',
        export='KEYFRAME/FK/STATUS evidence retained onfailure; GAIT exported ONLYifALLdensechecksPASSincludingrate.',
        provenance='Engineeredteacherretiming ofexistingreference, notnewanimalmotiondata. No physicalunloading/materialslip/support/neuralclaim.',physics_steps=0,neural_steps=0)
    b.dump(OUT/'PREREGISTRATION.json',prereg);b.dump(OUT/'SOURCE_SHA.json',hashes)
    start=time.monotonic();prep=Preparation();rows=[]
    for leg in range(6):
        previous=None
        for v in (.25,.5,.75):
            phase=((prep.duty+(1-prep.duty)*v-prep.offset[leg])%1)*2*np.pi
            q,row=prep.ik(phase,leg,previous=previous);previous=q;rows.append(row)
            print(json.dumps(dict(leg=LEGS[leg],v=v,position=row['position_error'],clearance=row['wholefoot_clearance_above_common_plane'],orientation_change=row['orientation_error'],passed=row['passed'])),flush=True)
    keypass=all(r['passed'] for r in rows)
    b.dump(OUT/'KEYFRAME_SCREEN.json',dict(passed=keypass,common_stance_plane=prep.common_plane,
        source_stance_medians=prep.source_medians,rows=rows,physics_steps=0))
    np.savez_compressed(OUT/'KEYFRAME_EVIDENCE.npz',phase=[r['phase_rad'] for r in rows],legs=[r['leg'] for r in rows],
        target=[r['target'] for r in rows],achieved=[r['achieved'] for r in rows],solved_joints=[r['solved_joints'] for r in rows],
        target_orientation=[r['target_orientation'] for r in rows],achieved_orientation=[r['achieved_orientation'] for r in rows])
    b.figure(rows)
    if not keypass:
        status=dict(status='KEYFRAME_POSITION_OR_CLEARANCE_FAILED',fullcurve_prepared=False,
            failed=[dict(leg=r['leg'],v=r['swing_v'],position=r['position_error'],clearance=r['wholefoot_clearance_above_common_plane']) for r in rows if not r['passed']],
            air_orientation_change_max=max(r['orientation_error'] for r in rows),physics_steps=0,neural_steps=0,
            FK_calls=prep.calls,wall_s=time.monotonic()-start,source_hashes_unchanged={p:b.sha(p)==h for p,h in hashes.items()})
        b.dump(OUT/'STATUS.json',status);print(json.dumps(status),flush=True);return 2
    raw=prep.old.copy();rawlogs=[];previous=[None]*6
    for i in range(512):
        for leg,ix in enumerate(prep.ids):
            if not prep.stance[i,leg]:
                q,row=prep.ik(prep.phase[i],leg,previous=previous[leg]);raw[i,ix]=q;rawlogs.append(row)
            previous[leg]=raw[i,ix].copy()
    raw[-1]=raw[0];b.dump(OUT/'RAW_IK_LOG.json',rawlogs)
    n=512;D=diags([np.ones(n-1),-2*np.ones(n),np.ones(n-1)],[-1,0,1],shape=(n,n),format='lil')
    D[0,n-1]=1;D[n-1,0]=1;D=D.tocsr();filtered=prep.old.copy();smooth=[]
    for leg,ix in enumerate(prep.ids):
        free=np.flatnonzero(~prep.stance[:n,leg]);A=vstack([eye(len(free),format='csr'),10*D[:,free]])
        for j in ix:
            fit=lsq_linear(A,np.r_[raw[free,j]-prep.old[free,j],np.zeros(n)],
                bounds=(prep.lo[j]-prep.old[free,j],prep.hi[j]-prep.old[free,j]),tol=1e-9,
                lsq_solver='lsmr',lsmr_tol=1e-9,max_iter=100,lsmr_maxiter=2000)
            filtered[free,j]=prep.old[free,j]+fit.x;smooth.append(dict(joint=str(prep.names[j]),success=bool(fit.success),iterations=int(fit.nit)))
    filtered[-1]=filtered[0];b.dump(OUT/'SMOOTH_SOLVERS.json',smooth)
    first,evidence=prep.evaluate(filtered);b.dump(OUT/'FILTERED_VERIFICATION.json',first)
    geometry=('position','sole_median_spread','plateau_wholefoot_clearance','stance_point_position','stance_point_sole_distance','stance_orientation')
    need=not all(first['checks'][k] for k in geometry);final=filtered.copy();projection=[]
    if need:
        previous=[filtered[-2,ix].copy() for ix in prep.ids]
        for i in range(512):
            for leg,ix in enumerate(prep.ids):
                if not prep.stance[i,leg]:
                    q,row=prep.ik(prep.phase[i],leg,previous=previous[leg],seed=filtered[i,ix],filtered_prior=filtered[i,ix],max_nfev=30)
                    final[i,ix]=q;projection.append(row)
                previous[leg]=final[i,ix].copy()
        final[-1]=final[0];result,evidence=prep.evaluate(final)
    else:result=first
    np.savez_compressed(OUT/'FK_EVIDENCE.npz',**evidence)
    np.savez_compressed(OUT/'CORRECTIONS.npz',phase=prep.phase,source=prep.old,raw=raw-prep.old,
        filtered=filtered-prep.old,final=final-prep.old,source_stance=prep.stance)
    b.dump(OUT/'PROJECTION_LOG.json',projection)
    meta=json.loads(str(prep.source['metadata_json'].item()))
    meta.update(version='lift-first-free-air-orientation-v2',source=str(b.SOURCE/'GAIT.npz'),
        source_sha256=hashes[str(b.SOURCE/'GAIT.npz')],air_orientation_free=True,fk_feasible=result['passed'],
        checks=result['checks'],max_rate_at_1hz=result['max_rate_1hz'],teacher_only=True,physics_steps=0)
    if result['passed']:
        np.savez_compressed(OUT/'GAIT.npz',phase=prep.phase,qpos_hinges=final,qpos_addresses=prep.p.qa,
            joint_names=prep.names,metadata_json=np.array(json.dumps(meta)))
    b.dump(OUT/'METADATA.json',meta)
    result.update(status='FK_ACCEPTED' if result['passed'] else 'DENSE_GEOMETRY_CLEARANCE_OR_RATE_FAILED',
        gait_exported=bool(result['passed']),filtered_prior_projection_performed=need,FK_calls=prep.calls,
        physics_steps=0,neural_steps=0,wall_s=time.monotonic()-start,
        source_hashes_unchanged={p:b.sha(p)==h for p,h in hashes.items()})
    b.dump(OUT/'VERIFICATION.json',result);b.dump(OUT/'STATUS.json',result)
    print(json.dumps(result,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),flush=True)
    return 0 if result['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
