"""Read-only independent phase/mapping/rate audit; no controller imports/mj_step."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.interpolate import PchipInterpolator
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T110150Z')
SOURCE=OUT.parent/'20261107T101114Z/GAIT.npz'
LEGS=('LF','LM','LH','RF','RM','RH');ORDER=('RH','LH','RM','LM','RF','LF')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def curve(p,q):return PchipInterpolator(np.r_[p[-3:-1]-2*np.pi,p,p[1:3]+2*np.pi],np.concatenate((q[-3:-1],q,q[1:3]),axis=0),axis=0,extrapolate=False)

def main():
    files=[SOURCE,OUT/'GAIT.npz',OUT/'FK_EVIDENCE.npz',OUT/'PREREGISTRATION.json',Path(__file__)]
    before={str(p):sha(p) for p in files};g=load(OUT/'GAIT.npz');s=load(SOURCE);e=load(OUT/'FK_EVIDENCE.npz')
    meta=json.loads(str(g['metadata_json'].item()));oldmeta=json.loads(str(s['metadata_json'].item()));protocol=json.loads((OUT/'PREREGISTRATION.json').read_text())
    old=np.asarray(oldmeta['phase_offsets']);new=np.asarray(meta['phase_offsets']);duty=5/6;expected=np.array([(duty-ORDER.index(l)/6)%1 for l in LEGS])
    f=curve(s['phase'],s['qpos_hinges']);h=curve(g['phase'],g['qpos_hinges']);names=g['joint_names'].astype(str)
    ids=[np.flatnonzero(np.char.startswith(names,'joint_'+leg)) for leg in LEGS];errors=[]
    for j in range(6):
        query=(g['phase']+2*np.pi*(new[j]-old[j]))%(2*np.pi)
        desired=f(query)[:,ids[j]];desired[-1]=desired[0]
        errors.append(float(abs(g['qpos_hinges'][:,ids[j]]-desired).max()))
    sys.path.insert(0,str(BASE));from incremental_actuator import load_full_model,DEFAULT_MODEL_PATH
    m=load_full_model();jids=m.actuator_trnid[:,0];qa=m.jnt_qposadr[jids]
    dense=np.linspace(0,2*np.pi,4097);q=h(dense);lo=m.jnt_range[jids,0];hi=m.jnt_range[jids,1]
    excess=float(max((lo-q).max(),(q-hi).max(),0));seam=float(abs(h(0,1)-h(2*np.pi,1)).max())
    maxima=np.zeros(42)
    for i in range(len(h.x)-1):
        left,right=h.x[i:i+2]
        if left<-1e-12 or right>2*np.pi+1e-12:continue
        a,b,c=h.c[:3,i];width=right-left
        value=np.maximum(abs(c),abs(3*a*width*width+2*b*width+c))
        vertex=np.divide(-b,3*a,out=np.full_like(a,np.nan),where=abs(a)>1e-20)
        inside=(vertex>0)&(vertex<width)
        value=np.where(inside,np.maximum(value,abs(3*a*vertex*vertex+2*b*vertex+c)),value)
        maxima=np.maximum(maxima,value)
    rates=maxima*2*np.pi
    midpoint=(np.arange(4096)+.5)/4096
    stance=((midpoint[:,None]+new)%1)<duty
    liftoffs=sorted((float((duty-new[j])%1),LEGS[j]) for j in range(6))
    rh=LEGS.index('RH');lh=LEGS.index('LH');lh_time=(duty-new[lh])%1
    rh_u=(lh_time+new[rh])%1
    checks=dict(offset_formula=np.allclose(new,expected,atol=1e-15,rtol=0),chronological_order=tuple(l for t,l in liftoffs)==ORDER,
        nominal_five_stance=bool(np.all(stance.sum(1)==5)),one_swing=bool(np.all((~stance).sum(1)==1)),
        source_query_sign_and_perleg_knots=max(errors)<=1e-12,all42_byname_once=sorted(np.concatenate(ids).tolist())==list(range(42)),
        native_actuator_mapping=bool(np.array_equal(g['qpos_addresses'],qa) and np.array_equal(g['joint_names'].astype(str),np.array([m.joint(int(j)).name for j in jids]))),
        same_source_phasegrid=bool(np.array_equal(g['phase'],s['phase'])),same_duty=meta['parameters']['duty']==oldmeta['parameters']['duty']==duty,
        RH_absolute_phase_unchanged=bool(new[rh]==old[rh] and np.array_equal(g['qpos_hinges'][:,ids[rh]],s['qpos_hinges'][:,ids[rh]])),
        same_source_Rstar_metadata=bool(np.array_equal(meta['target_orientation_world'],oldmeta['target_orientation_world'])),
        native_bounds=excess<=1e-9,seam=seam<=1e-10,rate_1hz=float(rates.max())<=30.,
        dense_saved_joint_match=bool(np.array_equal(dense,e['phase']) and np.array_equal(q,e['qpos_hinges'])),
        periodic_end_exact=bool(np.array_equal(g['qpos_hinges'][0],g['qpos_hinges'][-1])))
    registered={path:Path(path).exists() and sha(path)==digest for path,digest in protocol['source_hashes'].items()}
    checks['registered_sources_unchanged']=all(registered.values())
    report=dict(status='INDEPENDENT_PHASE_MAPPING_ACCEPTED' if all(checks.values()) else 'PHASE_MAPPING_CHECK_FAILED',passed=bool(all(checks.values())),
        checks={k:bool(v) for k,v in checks.items()},legs=LEGS,old_offsets=old.tolist(),new_offsets=new.tolist(),liftoffs=liftoffs,
        RH_local_phase_at_LH_liftoff=float(rh_u),old_RH_local_phase_at_LH_liftoff=float(((duty-old[lh])%1+old[rh])%1),
        perleg_knot_query_error=errors,max_rate_at_1hz=float(rates.max()),worst_rate_joint=str(names[rates.argmax()]),
        bound_excess=excess,seam_derivative=seam,registered_sources_unchanged=registered,input_hashes=before,
        model_file_sha256=sha(DEFAULT_MODEL_PATH),counts=dict(physics_steps=0,neural_steps=0,FK_forward=0),
        limitation='Phase/mapping/interpolation audit only. Model loaded withoutstepping; source-shift denseFK is parent evidence and not independentlyrerun in this quickcaseaudit. Five nominal stance legs do not prove loads, low slip, or neural success.')
    assert all(sha(p)==h for p,h in before.items())
    output=OUT/'INDEPENDENT_PHASE_CHECK.json'
    if output.exists():output.read_bytes();raise FileExistsError('preserve prior phase check')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report),flush=True)
    return 0 if report['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
