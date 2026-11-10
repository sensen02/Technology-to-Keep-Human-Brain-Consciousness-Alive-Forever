"""Independent exported path verification; model FK and geometry distances only.
Direct periodic-PCHIP recipe and analytic derivative extrema; no preparation-helper
or solver imports, no mj_step, no creation of candidate assets. Air orientation is
reported, not gated; source stance pose/sole/orientation and knots remain gated.
"""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T101114Z')
ORIGINAL=OUT.parent/'20261107T083316Z/GAIT.npz'
INTENDED=OUT.parent/'20261107T095519Z/FK_EVIDENCE.npz'
LEGS=('LF','LM','LH','RF','RM','RH')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def curve(p,q):return PchipInterpolator(np.r_[p[-3:-1]-2*np.pi,p,p[1:3]+2*np.pi],np.concatenate((q[-3:-1],q,q[1:3]),axis=0),axis=0,extrapolate=False)
def write(path,data):
    if path.exists():path.read_bytes()
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--no-write',action='store_true');args=ap.parse_args(argv)
    if not (OUT/'GAIT.npz').exists():
        status=OUT/'STATUS.json'
        if not status.exists():print(json.dumps(dict(status='PREPARATION_NOT_FINISHED',physics_steps=0)));return 3
        saved=json.loads(status.read_text())
        failure=dict(status='NO_ACCEPTED_GAIT_EXPORTED',accepted=False,preparation_status=saved.get('status'),
            preparation_checks=saved.get('checks'),physics_steps=0,FK_calls=0,
            source_status_hash=sha(status),verifier=str(Path(__file__)),verifier_hash=sha(__file__),
            conclusion='Failed preparation retained. No candidate was created or exported by independent verifier.')
        if not args.no_write:write(OUT/'INDEPENDENT_VERIFICATION.json',failure)
        print(json.dumps(failure));return 2
    paths=[OUT/n for n in ('GAIT.npz','FK_EVIDENCE.npz','TRAJECTORY.npz','PREREGISTRATION.json','SOURCE_SHA.json','GN_LOG.json')]+[ORIGINAL,INTENDED,Path(__file__)]
    before={str(p):sha(p) for p in paths};g,e,trajectory=[load(p) for p in paths[:3]]
    source=load(ORIGINAL);intended=load(INTENDED);protocol=json.loads((OUT/'PREREGISTRATION.json').read_text());logs=json.loads((OUT/'GN_LOG.json').read_text())
    meta=json.loads(str(g['metadata_json'].item()));oldmeta=json.loads(str(source['metadata_json'].item()))
    sys.path.insert(0,str(BASE));import mujoco as mj
    from plant_probe import Plant
    p=Plant();m=p.m;d=mj.MjData(m);q0=p.q0.copy();lo=m.jnt_range[p.jids,0];hi=m.jnt_range[p.jids,1];q0[p.qa]=np.clip(q0[p.qa],lo,hi)
    names=np.array([m.joint(int(j)).name for j in p.jids]);names_saved=g['joint_names'].astype(str)
    checks=dict(full_model=(m.nq,m.nv,m.nu,m.nmesh,m.neq)==(49,48,42,71,0),native_qpos_order=bool(np.array_equal(g['qpos_addresses'],p.qa)),joint_names=bool(np.array_equal(names,names_saved)),
        same_phase_knots=bool(np.array_equal(g['phase'],source['phase'])),same_phase_offsets=bool(np.array_equal(meta['phase_offsets'],oldmeta['phase_offsets'])),same_duty=meta['parameters']['duty']==oldmeta['parameters']['duty'])
    interpolation=curve(g['phase'],g['qpos_hinges']);original=curve(source['phase'],source['qpos_hinges']);phase=np.linspace(0,2*np.pi,4097)
    q=interpolation(phase);oldq=original(phase);offset=np.asarray(meta['phase_offsets']);duty=meta['parameters']['duty'];local=(phase[:,None]/(2*np.pi)+offset)%1
    stance=(local<=duty+1e-10)|(local>=1-1e-10);v=(local-duty)/(1-duty);plateau=(~stance)&(v>=.25-1e-10)&(v<=.75+1e-10)
    assert np.array_equal(e['phase'],phase) and np.array_equal(intended['phase'],phase)
    targets=intended['target_world'];checks['targets_match_original_declared_geometry']=bool(np.array_equal(targets,e['target_world']))
    checks['exported_dense_joints_match']=bool(np.allclose(q,e['qpos_hinges'],rtol=0,atol=1e-12))
    ids=[np.flatnonzero(np.char.startswith(names,'joint_'+leg)) for leg in LEGS];knotlocal=(g['phase'][:,None]/(2*np.pi)+offset)%1
    knotstance=(knotlocal<=duty+1e-10)|(knotlocal>=1-1e-10)
    checks['all_original_STANCE_knots_bitwise']=all(np.array_equal(g['qpos_hinges'][knotstance[:,j]][:,ids[j]],source['qpos_hinges'][knotstance[:,j]][:,ids[j]]) for j in range(6))
    floor=int(np.flatnonzero(p.planes)[0]);geom=[np.flatnonzero(p.feet==j) for j in range(6)];ft=np.zeros(6)
    def forward(angles):
        d.qpos[:]=q0;d.qpos[p.qa]=angles;mj.mj_forward(m,d)
        assert d.time==0 and np.array_equal(d.qpos[:7],p.q0[:7])
        sole=np.array([min(mj.mj_geomDistance(m,d,int(g),floor,1000.,ft) for g in geom[j]) for j in range(6)])
        return d.xpos[p.bids].copy(),d.xmat[p.bids].reshape(6,3,3).copy(),sole
    actual=np.empty((4097,6,3));sourcepos=np.empty_like(actual);rot=np.empty((4097,6,3,3));oldrot=np.empty_like(rot)
    soles=np.empty((4097,6));oldsoles=np.empty_like(soles)
    for i in range(4097):
        actual[i],rot[i],soles[i]=forward(q[i]);sourcepos[i],oldrot[i],oldsoles[i]=forward(oldq[i])
    # Common stance plane uses the original513sourceknots, matching its declared median definition.
    med=np.array([np.median(oldsoles[::8][knotstance[:,j],j]) for j in range(6)]);common=float(med.max())
    pe=np.linalg.norm(actual-targets,axis=2);oe=np.empty((4097,6))
    for j in range(6):oe[:,j]=np.linalg.norm(Rotation.from_matrix(np.swapaxes(oldrot[:,j],1,2)@rot[:,j]).as_rotvec(),axis=1)
    stancedistance=abs(soles-oldsoles);stanceposition=np.linalg.norm(actual-sourcepos,axis=2)
    excess=float(max((lo-q).max(),(q-hi).max(),0.));seam=float(abs(interpolation(0,1)-interpolation(2*np.pi,1)).max())
    rate=np.zeros(42);peak=0.;location=None
    for k in range(len(interpolation.x)-1):
        left,right=interpolation.x[k:k+2]
        if left<-1e-12 or right>2*np.pi+1e-12:continue
        a,b,c=interpolation.c[:3,k];width=right-left
        for j in range(42):
            candidates=[0.,width]
            if abs(a[j])>1e-20:
                vertex=-b[j]/(3*a[j])
                if 0<vertex<width:candidates.append(vertex)
            for x in candidates:
                speed=float(abs(3*a[j]*x*x+2*b[j]*x+c[j])*2*np.pi);rate[j]=max(rate[j],speed)
                if speed>peak:peak=speed;location=dict(joint=str(names[j]),phase_rad=float(left+x),global_cycle=float((left+x)/(2*np.pi)))
    medafter=np.array([np.median(soles[stance[:,j],j]) for j in range(6)])
    checks.update(position=float(pe.max())<=.005,plateau_true_wholefoot_clearance=float((soles-common)[plateau].min())>=.1,
        stance_sole_median_spread=float(np.ptp(medafter))<=.005,stance_point_position=float(stanceposition[stance].max())<=.005,
        stance_point_sole_deviation=float(stancedistance[stance].max())<=.005,stance_orientation=float(oe[stance].max())<=.03,
        native_bounds=excess<=1e-9,periodic_endpoint=bool(np.array_equal(g['qpos_hinges'][0],g['qpos_hinges'][-1])),derivative_seam=seam<=1e-10,exact_rate_1hz=peak<=30.,
        saved_FK_positions_match=float(abs(actual-e['achieved_world']).max())<=1e-12,saved_true_geometry_distances_match=float(abs(soles-e['wholefoot_sole_distance']).max())<=1e-12)
    accepted=[]
    for leg in logs:
        for row in leg['iterations']:
            if 'objective_after' not in row:continue
            ok=row['objective_after']<row['objective_before']-1e-14 and row['step_max']<=.05+1e-8 and row['constraint_violation']<=1e-8
            if row.get('status_val')==7:ok=ok and row.get('inexact_accepted') is True and row['primal_residual']<=1e-8 and row['proposed_trust_step_max']<=.05+1e-8
            accepted.append(bool(ok))
    checks['accepted_GN_proposals_obey_declared_numerical_rule']=bool(accepted and all(accepted))
    hashes=json.loads((OUT/'SOURCE_SHA.json').read_text());unchanged={path:Path(path).exists() and sha(path)==digest for path,digest in hashes.items()}
    adaptation=protocol.get('adaptation',{})
    for key,digestkey in [('baseline','baseline_sha256'),('wrapper','wrapper_sha256'),('effective_source','effective_sha256')]:
        if key in adaptation:unchanged[adaptation[key]]=sha(adaptation[key])==adaptation[digestkey]
    checks['registered_sources_unchanged']=all(unchanged.values());assert all(sha(path)==digest for path,digest in before.items())
    report=dict(status='INDEPENDENT_FK_ACCEPTED' if all(checks.values()) else 'INDEPENDENT_FK_FAILED',accepted=all(checks.values()),checks=checks,
        errors=dict(position_max=float(pe.max()),stance_position_max=float(stanceposition[stance].max()),stance_sole_deviation_max=float(stancedistance[stance].max()),
            stance_orientation_max=float(oe[stance].max()),air_orientation_max=float(oe[~stance].max()),sole_median_spread=float(np.ptp(medafter)),
            plateau_clearance_min=float((soles-common)[plateau].min()),native_bound_excess=excess,derivative_seam=seam,
            saved_FK_position_discrepancy=float(abs(actual-e['achieved_world']).max()),saved_sole_distance_discrepancy=float(abs(soles-e['wholefoot_sole_distance']).max())),
        rate=dict(max_rad_s_1hz=peak,peak=location,per_joint_rad_s_1hz=rate.tolist()),common_stance_plane=common,
        original_stance_medians=med.tolist(),candidate_stance_medians=medafter.tolist(),
        sources_unchanged=unchanged,input_hashes=before,counts=dict(FK_forward=8194,physics_steps=0,neural_steps=0),
        limitation='FK/rate/geometry validation only; instantaneous numerical proposals and airborne orientation freedom do not establish physical unloading, material slip, neural control or walking.')
    if not args.no_write:write(OUT/'INDEPENDENT_VERIFICATION.json',report)
    print(json.dumps(report),flush=True);return 0 if report['accepted'] else 2

if __name__=='__main__':raise SystemExit(main())
