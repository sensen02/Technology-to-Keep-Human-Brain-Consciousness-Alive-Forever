"""Bounded offline lift-first swing screen; no physical or neural integration.

Source stance knots remain bitwise exact. Broadening lift is engineering
retiming, not a new animal-motion measurement. Keyframe failure stops full IK.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.sparse import diags, eye, vstack
from scipy.spatial.transform import Rotation

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
SOURCE = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T083316Z')
WAVE = SOURCE.parent/'20261107T081009Z/METADATA.json'
OUT = SOURCE.parent/'20261107T094930Z'
UTIL = ROOT/'experiments/s5-ann-walk/20261107T075536Z/prepare_stance_reference.py'
sys.path.insert(0, str(BASE))
import mujoco as mj
from plant_probe import Plant, sha
from incremental_actuator import DEFAULT_MODEL_PATH
spec = importlib.util.spec_from_file_location('swing_geometry_util', UTIL)
util = importlib.util.module_from_spec(spec); spec.loader.exec_module(util)
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda x:x.tolist() if hasattr(x, 'tolist') else str(x))+'\n')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def s5(v):
    x = np.clip(v, 0., 1.)
    return x*x*x*(10.-15.*x+6.*x*x)


class Preparation:
    def __init__(self):
        self.source = load(SOURCE/'GAIT.npz')
        self.wave = json.loads(WAVE.read_text())
        self.phase = self.source['phase']; self.old = self.source['qpos_hinges']
        self.offset = np.asarray(self.wave['phase_offsets']); self.duty = self.wave['parameters']['duty']
        assert self.duty == 5/6 and len(self.phase) == 513
        self.curve = util.v2.periodic_pchip(self.phase, self.old)
        self.p = Plant(); self.m = self.p.m; self.d = mj.MjData(self.m); self.calls = 0
        assert np.array_equal(self.p.qa, self.source['qpos_addresses'])
        self.names = self.source['joint_names'].astype(str)
        assert list(self.names) == [self.m.joint(int(j)).name for j in self.p.jids]
        self.ids = [np.flatnonzero(np.char.startswith(self.names, 'joint_'+leg)) for leg in LEGS]
        assert all(len(ix) == 7 for ix in self.ids)
        self.lo, self.hi = self.m.jnt_range[self.p.jids].T
        self.q0 = self.p.q0.copy(); self.q0[self.p.qa] = np.clip(self.q0[self.p.qa], self.lo, self.hi)
        self.oldpos = np.empty((513, 6, 3)); self.oldrot = np.empty((513, 6, 3, 3))
        for i, q in enumerate(self.old):
            self.fk(q); self.oldpos[i] = self.d.xpos[self.p.bids]
            self.oldrot[i] = self.d.xmat[self.p.bids].reshape(6, 3, 3)
        u = self.local_phase(self.phase)
        self.stance = (u <= self.duty+1e-10) | (u >= 1.-1e-10)
        quiet = (u > .15) & (u < self.duty-.15)
        self.stance_z = np.array([np.median(self.oldpos[quiet[:, j], j, 2]) for j in range(6)])
        self.plane = int(np.flatnonzero(self.p.planes)[0])
        self.geoms = [np.flatnonzero(self.p.feet == j) for j in range(6)]
        self.fromto = np.zeros(6)

    def local_phase(self, phase):
        return (np.asarray(phase)[..., None]/(2*np.pi)+self.offset) % 1.

    def fk(self, q):
        self.d.qpos[:] = self.q0; self.d.qpos[self.p.qa] = q
        mj.mj_forward(self.m, self.d); self.calls += 1
        assert self.d.time == 0 and np.array_equal(self.d.qpos[:7], self.p.q0[:7])
        assert not self.d.warning.number.any()

    def target(self, phase, leg):
        phase = float(phase) % (2*np.pi)
        original = self.curve(phase)
        self.fk(original)
        goal = self.d.xpos[self.p.bids[leg]].copy()
        orientation = self.d.xmat[self.p.bids[leg]].reshape(3, 3).copy()
        u = float(self.local_phase(phase)[leg])
        if u <= self.duty+1e-10 or u >= 1.-1e-10:
            return original, goal, orientation, True
        v = (u-self.duty)/(1-self.duty)
        warped = s5((v-.2)/.6)
        source_phase = ((self.duty+(1-self.duty)*warped-self.offset[leg]) % 1.)*2*np.pi
        self.fk(self.curve(source_phase))
        goal[:2] = self.d.xpos[self.p.bids[leg], :2]
        height = .25*s5(v/.25) if v < .25 else .25*s5((1-v)/.25) if v > .75 else .25
        goal[2] = self.stance_z[leg]+height
        return original, goal, orientation, False

    def ik(self, phase, leg, seed=None, prior=None, prior_weight=.001, max_nfev=100):
        original, goal, orientation, stance = self.target(phase, leg)
        ix = self.ids[leg]; qa = self.p.qa[ix]; va = self.p.va[ix]
        if seed is None: seed = original[ix]
        if prior is None: prior = original[ix]
        cache = {}
        def evaluate(q):
            if 'q' in cache and np.array_equal(cache['q'], q):
                return cache['r'], cache['J']
            full = original.copy(); full[ix] = q; self.fk(full)
            point = self.d.xpos[self.p.bids[leg]]
            rv = Rotation.from_matrix(orientation.T@self.d.xmat[self.p.bids[leg]].reshape(3, 3)).as_rotvec()
            residual = np.r_[point-goal, .2*rv, prior_weight*(q-prior)]
            jp = np.zeros((3, self.m.nv)); jr = np.zeros_like(jp)
            mj.mj_jac(self.m, self.d, jp, jr, point, self.p.bids[leg])
            jac = np.vstack((jp[:, va], .2*util.v2.log_left_inverse(rv)@orientation.T@jr[:, va], prior_weight*np.eye(7)))
            cache.update(q=q.copy(), r=residual, J=jac)
            return residual, jac
        fit = least_squares(lambda q:evaluate(q)[0], np.clip(seed, self.lo[ix]+1e-12, self.hi[ix]-1e-12),
            jac=lambda q:evaluate(q)[1], bounds=(self.lo[ix], self.hi[ix]), max_nfev=max_nfev,
            ftol=1e-10, xtol=1e-10, gtol=1e-9, x_scale='jac')
        error, _ = evaluate(fit.x)
        full = original.copy(); full[ix] = fit.x; self.fk(full)
        row = dict(leg=LEGS[leg], phase_rad=float(phase), swing_v=float((self.local_phase(phase)[leg]-self.duty)/(1-self.duty)),
            solver_success=bool(fit.success), status=int(fit.status), nfev=int(fit.nfev),
            position_error=float(np.linalg.norm(error[:3])), orientation_error=float(np.linalg.norm(error[3:6])/.2),
            residual_xyz=error[:3], target=goal, achieved=self.d.xpos[self.p.bids[leg]].copy(),
            source_joints=original[ix], solved_joints=fit.x, joint_names=self.names[ix],
            joint_lower=self.lo[ix], joint_upper=self.hi[ix], active_joint_bounds=fit.active_mask,
            passed=bool(np.linalg.norm(error[:3])<=.005 and np.linalg.norm(error[3:6])/.2<=.03))
        return fit.x.copy(), row

    def soles(self):
        return np.array([min(mj.mj_geomDistance(self.m, self.d, int(g), self.plane, 1000., self.fromto)
                             for g in geoms) for geoms in self.geoms])

    def evaluate(self, qknots):
        curve = util.v2.periodic_pchip(self.phase, qknots)
        dense = np.linspace(0, 2*np.pi, 4097); qdense = curve(dense); olddense = self.curve(dense)
        target = np.empty((4097, 6, 3)); rotation = np.empty((4097, 6, 3, 3))
        for i, phase in enumerate(dense):
            for leg in range(6):
                _, target[i, leg], rotation[i, leg], _ = self.target(phase, leg)
        actual = np.empty_like(target); actualrot = np.empty_like(rotation); sole = np.empty((4097, 6))
        for i, q in enumerate(qdense):
            self.fk(q); actual[i] = self.d.xpos[self.p.bids]
            actualrot[i] = self.d.xmat[self.p.bids].reshape(6, 3, 3); sole[i] = self.soles()
        pe = np.linalg.norm(actual-target, axis=2)
        oe = np.empty((4097, 6))
        for j in range(6):
            oe[:, j] = np.linalg.norm(Rotation.from_matrix(rotation[:, j].transpose(0, 2, 1)@actualrot[:, j]).as_rotvec(), axis=1)
        u = self.local_phase(dense); stance = (u <= self.duty+1e-10) | (u >= 1.-1e-10)
        medians = np.array([np.median(sole[stance[:, j], j]) for j in range(6)])
        rates = 2*np.pi*util.exact_phase_rate_bounds(curve)
        seam = float(abs(curve(0., 1)-curve(2*np.pi, 1)).max())
        excess = float(max(np.max(self.lo-qdense), np.max(qdense-self.hi), 0.))
        exact = all(np.array_equal(qknots[self.stance[:, j]][:, self.ids[j]], self.old[self.stance[:, j]][:, self.ids[j]]) for j in range(6))
        dense_stance_delta = [float(abs(qdense[stance[:, j]][:, self.ids[j]]-olddense[stance[:, j]][:, self.ids[j]]).max()) for j in range(6)]
        checks = dict(position=float(pe.max())<=.005, orientation=float(oe.max())<=.03,
            sole_median_spread=float(np.ptp(medians))<=.005, native_bounds=excess<=1e-9,
            derivative_seam=seam<=1e-10, rate_1hz=float(rates.max())<=30., stance_knots_bitwise=exact,
            periodic_endpoint=bool(np.array_equal(qknots[0], qknots[-1])))
        report = dict(passed=bool(all(checks.values())), checks=checks, position_max=float(pe.max()),
            orientation_max=float(oe.max()), per_leg_position_max=pe.max(0), per_leg_orientation_max=oe.max(0),
            sole_medians=medians, sole_spread=float(np.ptp(medians)), bound_excess=excess, seam=seam,
            rates_1hz=rates, max_rate_1hz=float(rates.max()), worst_rate_joint=str(self.names[rates.argmax()]),
            dense_stance_joint_delta_max=dense_stance_delta)
        evidence = dict(phase=dense, qpos_hinges=qdense, source_qpos_hinges=olddense,
            target_world=target, target_orientation_world=rotation, achieved_world=actual,
            achieved_orientation_world=actualrot, position_error=pe, orientation_error_rad=oe,
            wholefoot_sole_distance=sole, nominal_stance=stance, qpos_addresses=self.p.qa)
        return report, evidence


def figure(keyframes, evidence=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for leg, ax in zip(LEGS, axes.flat):
        rows = [r for r in keyframes if r['leg'] == leg]
        goal = np.array([r['target'] for r in rows]); actual = np.array([r['achieved'] for r in rows])
        ax.plot(goal[:, 0], goal[:, 2], 'o-', label='requested keyframe x/z')
        ax.plot(actual[:, 0], actual[:, 2], 'x--', label='bounded IK achieved')
        for r in rows:
            ax.annotate('v='+format(r['swing_v'], '.2f'), (r['target'][0], r['target'][2]), fontsize=8)
        ax.set(title=leg, xlabel='native world x (fixed-root FK)', ylabel='native world z')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Lift first / return middle / lower last: geometric teacher only; 0 physics steps')
    fig.savefig(OUT/'SWING_TARGETS.png', dpi=150); plt.close(fig)


def main():
    def forbidden(*a, **kw):
        raise AssertionError('No mj_step in offline geometry preparation')
    mj.mj_step = forbidden
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'PREREGISTRATION.json').exists():
        raise FileExistsError('Preserve prior candidate: '+str(OUT))
    files = [Path(__file__), SOURCE/'GAIT.npz', SOURCE/'METADATA.json', WAVE, UTIL,
        Path(util.v2.__file__), Path(util.v1.__file__), BASE/'plant_probe.py', BASE/'incremental_actuator.py',
        ROOT/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py', DEFAULT_MODEL_PATH]
    hashes = {str(p):sha(p) for p in files}
    prereg = dict(stage='S5 bounded lift-first swing teacher geometry ONLY', source_hashes=hashes,
        target=dict(duty=5/6, lift=.25, xy_warp='source swing path at S5(clip((v-.2)/.6,0,1))',
            height='quiet stance originZ + .25*S5(v/.25), plateau v.25-.75, .25*S5((1-v)/.25)',
            orientation='same global-phase source orientation', stance='bitwise source joint knots including boundarywithin1e-10'),
        keyframe_screen=dict(swing_v=[.25, .5, .75], max_nfev=100, initial='same-phase source7hinges',
            position_weight=1., orientation_weight=.2, source_joint_prior=.001,
            stop='Any position>.005 or orientation>.03: stop before fullcurve; no threshold/lift changes'),
        fullcurve='513phase knots/512unique, only nominal swing IK max100nfev; fixedlambda100 source-relative correction smoothing with stance delta exactly0; one filteredprior .01 reprojection max30 onlyifgeometryfails',
        validation=dict(dense_phases=4097, position=.005, orientation_rad=.03, sole_median_spread=.005,
            native_bound_tolerance=1e-9, rate_rad_s_at1Hz=30., derivative_seam=1e-10),
        interpolation='exact periodic two-neighbor PCHIP; no clipping',
        boundary_disclosure='Raw XY targets have zero swing-end velocity vs backward stance velocity; target is not C1 there. Quiet-stance Z median may differ slightly from exact boundaryZ. PCHIP output is C1, but modified neighboring swing knots can alter dense stance intervals; dense stance deviation reported.',
        provenance='Explicit engineering retiming of traced source; broad Z plateau is not raw animal motion.',
        physics_steps=0, neural_steps=0, limitation='FK does not establish actual unloading, material slip, support, physical walking or neural success.')
    dump(OUT/'PREREGISTRATION.json', prereg); dump(OUT/'SOURCE_SHA.json', hashes)
    begin = time.monotonic(); prep = Preparation(); keyframes = []
    for leg in range(6):
        for v in (.25, .5, .75):
            phase = ((prep.duty+(1-prep.duty)*v-prep.offset[leg]) % 1.)*2*np.pi
            _, row = prep.ik(phase, leg); keyframes.append(row)
            print(json.dumps(dict(leg=row['leg'], v=v, position=row['position_error'], orientation=row['orientation_error'], passed=row['passed'])), flush=True)
    passed = all(r['passed'] for r in keyframes)
    dump(OUT/'KEYFRAME_SCREEN.json', dict(passed=passed, rows=keyframes, FK_calls=prep.calls, physics_steps=0))
    np.savez_compressed(OUT/'KEYFRAME_EVIDENCE.npz', phase=[r['phase_rad'] for r in keyframes],
        legs=[r['leg'] for r in keyframes], targets=[r['target'] for r in keyframes],
        achieved=[r['achieved'] for r in keyframes], solved_joints=[r['solved_joints'] for r in keyframes])
    figure(keyframes)
    if not passed:
        status = dict(status='KEYFRAME_WORKSPACE_SCREEN_FAILED', fullcurve_prepared=False,
            failed=[dict(leg=r['leg'], v=r['swing_v'], position=r['position_error'], orientation=r['orientation_error']) for r in keyframes if not r['passed']],
            physics_steps=0, neural_steps=0, FK_calls=prep.calls, wall_s=time.monotonic()-begin,
            source_hashes_unchanged={p:sha(p)==h for p,h in hashes.items()},
            limitation='Bounded local100-evaluation IK screen failed; not proof of global unreachability. No physical run or gait asset authorized from this screen.')
        dump(OUT/'STATUS.json', status); print(json.dumps(status), flush=True); return 2
    final = prep.old.copy(); raw_logs = []
    for i in range(512):
        for leg in range(6):
            if not prep.stance[i, leg]:
                q, row = prep.ik(prep.phase[i], leg); final[i, prep.ids[leg]] = q; raw_logs.append(row)
    final[-1] = final[0]; raw = final.copy(); dump(OUT/'RAW_IK_LOG.json', raw_logs)
    n = 512
    D = diags([np.ones(n-1), -2*np.ones(n), np.ones(n-1)], [-1, 0, 1], shape=(n, n), format='lil')
    D[0, n-1] = 1.; D[n-1, 0] = 1.; D = D.tocsr()
    filtered = prep.old.copy(); smooth_logs = []
    for leg, ix in enumerate(prep.ids):
        free = np.flatnonzero(~prep.stance[:n, leg]); A = vstack([eye(len(free), format='csr'), 10*D[:, free]])
        for j in ix:
            result = lsq_linear(A, np.r_[raw[free, j]-prep.old[free, j], np.zeros(n)],
                bounds=(prep.lo[j]-prep.old[free, j], prep.hi[j]-prep.old[free, j]),
                tol=1e-9, lsq_solver='lsmr', lsmr_tol=1e-9, max_iter=100, lsmr_maxiter=2000)
            filtered[free, j] = prep.old[free, j]+result.x
            smooth_logs.append(dict(joint=str(prep.names[j]), success=bool(result.success), iterations=int(result.nit)))
    filtered[-1] = filtered[0]; dump(OUT/'SMOOTH_SOLVERS.json', smooth_logs)
    first, evidence = prep.evaluate(filtered); dump(OUT/'FILTERED_VERIFICATION.json', first)
    need = not all(first['checks'][key] for key in ('position', 'orientation', 'sole_median_spread'))
    final = filtered.copy(); projection = []
    if need:
        for i in range(512):
            for leg, ix in enumerate(prep.ids):
                if not prep.stance[i, leg]:
                    q, row = prep.ik(prep.phase[i], leg, seed=filtered[i, ix], prior=filtered[i, ix], prior_weight=.01, max_nfev=30)
                    final[i, ix] = q; projection.append(row)
        final[-1] = final[0]; result, evidence = prep.evaluate(final)
    else:
        result = first
    meta = json.loads(str(prep.source['metadata_json'].item()))
    meta.update(version='lift-first-broad-swing-v1', source=str(SOURCE/'GAIT.npz'), source_sha256=hashes[str(SOURCE/'GAIT.npz')],
        engineering_retiming=prereg['target'], fk_feasible=result['passed'], checks=result['checks'],
        max_rate_at_1hz=result['max_rate_1hz'], teacher_only=True, physics_steps=0)
    np.savez_compressed(OUT/'GAIT.npz', phase=prep.phase, qpos_hinges=final, qpos_addresses=prep.p.qa,
                        joint_names=prep.names, metadata_json=np.array(json.dumps(meta)))
    np.savez_compressed(OUT/'FK_EVIDENCE.npz', **evidence)
    np.savez_compressed(OUT/'CORRECTIONS.npz', phase=prep.phase, raw=raw-prep.old, filtered=filtered-prep.old,
                        final=final-prep.old, source=prep.old, source_stance=prep.stance)
    dump(OUT/'PROJECTION_LOG.json', projection); dump(OUT/'METADATA.json', meta)
    result.update(status='FK_ACCEPTED' if result['passed'] else 'DENSE_GEOMETRY_OR_RATE_FAILED',
        filtered_prior_projection_performed=need, FK_calls=prep.calls, physics_steps=0, neural_steps=0,
        wall_s=time.monotonic()-begin, source_hashes_unchanged={p:sha(p)==h for p,h in hashes.items()})
    dump(OUT/'VERIFICATION.json', result); dump(OUT/'STATUS.json', result)
    print(json.dumps(result, default=lambda x:x.tolist() if hasattr(x, 'tolist') else str(x)), flush=True)
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
