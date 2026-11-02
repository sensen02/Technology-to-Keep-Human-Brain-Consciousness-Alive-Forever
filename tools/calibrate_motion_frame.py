"""Train-locked source anatomical-frame hypothesis, offline S2/S4 diagnosis only.
No runtime control, body integration, network initialization, or training.
"""
from pathlib import Path
import argparse, csv, json, sys, traceback, math
import xml.etree.ElementTree as ET
import numpy as np
import retarget_continuous_fly_motion as old

OUT=Path('/home/sensen/Desktop/engine修复验收/s4-fixed-motion-frame-calibration/20261102T023600Z')
PREV=Path('/home/sensen/Desktop/engine修复验收/s4-middle-leg-retarget-diagnosis/20261102T020329Z')
COUNTS=dict(model_compile_attempts=0,model_compile=0,kinematics=0,com_pos=0,jacobian=0,forward=0,ik_solves=0,body_steps=0,neural_steps=0,network_inits=0,training=0)
def dump(n,x): old.dump(OUT/n,x)
def load(n): return old.load(OUT/n)
def table(n,rows):
    with (OUT/n).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def identities():
    paths=[Path(__file__),Path(old.__file__),old.MODEL,old.SOURCE/'annotated_3d_walking_kinematics.csv',old.SOURCE/'NMF2_CONTINUOUS.npz',old.SOURCE/'NMF2_FIELDS_AND_TRANSFORMS.json',old.SOURCE/'SOURCE_NMF2_RESEARCH.md',old.OUT/'RETARGETED_TARGETS.npz',old.OUT/'TRANSFORM_AND_UNIT_LEDGER.json',old.OUT/'MODEL_MARKER_CONTRACT.json',old.OUT/'LOSS_ADMISSION.npz',old.OUT/'MANIFEST_SHA256.json',PREV/'FIRST_SEGMENT_REACHABILITY_CERTIFICATE.json',PREV/'DIAGNOSTIC_CANDIDATES.npz',PREV/'MANIFEST_SHA256.json']
    return {str(p):old.sha(p) for p in paths}
def raw():
    rows=list(csv.DictReader((old.SOURCE/'annotated_3d_walking_kinematics.csv').open()))
    v=lambda name:np.array([[float(r[name+'_'+c]) for c in 'xyz'] for r in rows])
    return rows,{s:v(s) for s in ['He','Th','Abd','LA','RA']}
def anatomical(z,lm):
    tr=z['split']==0
    # Equal weight to each unit forward vector, then average all train observations.
    forward=(old.unit(lm['He']-lm['Th'])[tr]+old.unit(lm['Th']-lm['Abd'])[tr]).mean(0)/2
    x=old.unit(forward)
    lateral=z['xyz_source'][tr,:3,0].mean((0,1))-z['xyz_source'][tr,3:,0].mean((0,1))
    y=old.unit(lateral-x*np.dot(x,lateral));zz=np.cross(x,y)
    R=np.stack([x,y,zz]);assert np.max(np.abs(R@R.T-np.eye(3)))<1e-12 and abs(np.linalg.det(R)-1)<1e-12
    return R,dict(forward_pre_normalization=forward,lateral_pre_orthogonalization=lateral,axes_rows=R,determinant=np.linalg.det(R),train_source_frames=z['source_frame'][tr],status='DERIVED_HYPOTHESIS',model_convention='+x anterior, +y left, +z right-handed cross product in canonical Thorax',origin='Th fixed (0,0,200), no measured world root',definition_limit='He/Abd exact anatomical placement and source body alignment not established. LA/RA likely antennae per prior research, not used to fit. No independent measured extrinsics.')
def normalized_layout(a):
    a=a-a.mean(0);return a/np.linalg.norm(a)
def envelopes():
    tree=ET.parse(old.MODEL);b={x.get('name'):x for x in tree.iter('body')};j={x.get('name'):x for x in tree.iter('joint') if x.get('name')};out=[]
    for leg in old.LEGS:
        c=b[leg+'Coxa'];assert not any(c.get(k) for k in ['quat','euler','axisangle','xyaxes','zaxis'])
        actual=c.findall('joint');order=[x.get('name').rsplit('_',1)[1] for x in actual]
        assert order==(['roll','yaw','pitch'] if leg=='RF' else ['yaw','pitch','roll'])
        jj=[j['joint_'+leg+'Coxa_'+s] for s in ['yaw','pitch','roll']]
        for x,axis in zip(jj,np.eye(3)):
            assert np.linalg.norm(np.fromstring(x.get('axis'),sep=' ')-axis)<1e-15
            assert np.linalg.norm(np.fromstring(x.get('pos'),sep=' '))==0
        v=old.unit(np.fromstring(b[leg+('Trochanter' if leg[1]=='F' else 'Femur')].get('pos'),sep=' '))
        eps=float(np.arccos(np.clip(-v[2],-1,1)));yy=np.fromstring(jj[0].get('range'),sep=' ');pp=np.fromstring(jj[1].get('range'),sep=' ')
        assert np.all(np.abs(yy)<np.pi/2) and np.all(np.abs(pp)<np.pi/2)
        slack=2*np.sin(eps/2)+1e-10
        out.append(dict(leg=leg,epsilon_rad=eps,cap_rad=float(np.arccos(np.cos(np.abs(yy).max())*np.cos(np.abs(pp).max()))+eps+1e-10),ymin=float(max(-1,min(0,np.sin(yy[0]))-slack)),ymax=float(min(1,max(0,np.sin(yy[1]))+slack)),xmin=float(max(-1,-np.sin(pp[1])-slack)),xmax=float(min(1,-np.sin(pp[0])+slack)),coxa_bounds=[np.fromstring(x.get('range'),sep=' ') for x in jj],initial_unit_chord=v))
        out[-1]['actual_joint_order']=order
        if leg=='RF':
            # Rz roll is outermost: it preserves down-polar angle, but rotates x/y slabs.
            out[-1].update(xmin=-1.,xmax=1.,ymin=-1.,ymax=1.)
    return out
def necessary(d,env):
    allb=[]
    for li,e in enumerate(env):
        t=d[:,li,0];p=np.maximum(0,np.arccos(np.clip(-t[:,2],-1,1))-e['cap_rad'])
        for k,s in [(0,'x'),(1,'y')]:
            a=np.arcsin(np.clip(t[:,k],-1,1));p=np.maximum.reduce([p,a-np.arcsin(e[s+'max']),np.arcsin(e[s+'min'])-a])
        allb.append(np.rad2deg(p))
    return np.array(allb).T

def prereg():
    return dict(stage='S2/S4 offline frame identifiability; not model repair or walking',candidates=['old ThC-Kabsch reused exact, zero new solves','train-landmark anatomical R'],third_candidate='NONE: no independent scientific constraint justifies optimizing rotation',fit_frames=list(range(29,51)),excluded_fit_frames=list(range(51,66)),frame_rule='x=unit(mean_train((unit(He-Th)+unit(Th-Abd))/2)); y=unit(mean_train,left3 ThC - mean_train,right3 ThC minus x projection); z=cross(x,y); R rows x,y,z to canonical +x anterior,+y left,+z cross. No per-frame fit.',anchor_rule='center source train mean 6 ThC and model anchors separately, each divide Frobenius norm; residual per-leg Euclidean, compare mean; frozen normalization on holdout (train source centroid/norm)',necessary_bound='Five legs Rx(yaw)Ry(pitch)Rz(roll): down polar cap plus conservative x/y latitude slabs, offset slack 2sin(epsilon/2)+1e-10. RF actual Rz(roll)Rx(yaw)Ry(pitch): same down polar cap, no x/y slabs. Necessary only, zero is inconclusive',selection=dict(anchor_mean_increase_max=.05,middle_train_mean_violation_reduction_min_deg=10,each_leg_train_mean_violation_increase_max_deg=2,each_leg_train_max_violation_increase_max_deg=5,action='select anatomical only if every condition passes; otherwise retain old and mark R not independently identifiable'),full_record_trigger='same conjunction as selection; one chronological 37x6 nativebounds IK using original direction+1e-6 temporal objective',fallback_cases=[dict(leg='LM',frame=64,role='old high HOLDOUT diagnostic only'),dict(leg='LM',frame=37,role='old low train'),dict(leg='RM',frame=45,role='old high train'),dict(leg='RM',frame=47,role='old low-mid train')],fallback_seeds=['spring','old previous','old next','native midpoint','rng20261102 draw0','rng20261102 draw1'],fallback_max_solves=24,solver=dict(max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-9,jacobian='analytic unit-direction from mj_jac'),baseline='reuse old qpos and previous q prior under same objective and target definition; only R changes; no source point changes',root_valid=False,Claw_valid=False,native_joint_truth_target=False,runtime_enabled=False,source_units='NOT_ESTABLISHED',expected=dict(main_compile=1,fresh_fk_compile=1,body_steps=0,neural_steps=0,network_inits=0,training=0))

def run():
    from scipy.optimize import least_squares
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'PREREGISTRATION.json').exists()
    dump('PREREGISTRATION.json',prereg());dump('SOURCE_IDENTITY.json',identities());(OUT/'EXECUTION_SOURCE.py').write_bytes(Path(__file__).read_bytes())
    for directory in [old.OUT,PREV]:
        assert all(old.sha(p)==h for p,h in json.loads((directory/'MANIFEST_SHA256.json').read_text()).items())
    z=old.load(old.SOURCE/'NMF2_CONTINUOUS.npz');a=old.load(old.OUT/'RETARGETED_TARGETS.npz');rr,lm=raw();assert len(rr)==37
    assert np.array_equal(np.array([int(r['frame_idx']) for r in rr]),z['source_frame']) and np.all(lm['Th']==[0,0,200])
    R,ledger=anatomical(z,lm);ro=a['rotation_source_to_model'];env=envelopes()
    dirs,dm,refs,pm=old.targets(z,R,a['model_anchor_points'],a['model_chord_lengths']);oldlb=necessary(a['reference_directions'],env);newlb=necessary(dirs,env)
    # Confirm the conservative new certificate against baseline achieved residuals on every leg.
    assert np.all(oldlb<=np.rad2deg(a['direction_residual_rad'][:,:,0])+1e-7)
    tr=z['split']==0;sm=z['xyz_source'][tr,:,0].mean(0);sn=normalized_layout(sm);mn=normalized_layout(a['model_anchor_points'])
    ar0=np.linalg.norm(sn@ro.T-mn,axis=1);ar1=np.linalg.norm(sn@R.T-mn,axis=1)
    delta_mean=newlb[tr].mean(0)-oldlb[tr].mean(0);delta_max=newlb[tr].max(0)-oldlb[tr].max(0);improve=float((oldlb[tr][:,[1,4]]-newlb[tr][:,[1,4]]).mean())
    gates=dict(anchor=bool(ar1.mean()-ar0.mean()<=.05),middle=bool(improve>=10),six_leg_mean=bool(np.all(delta_mean<=2)),six_leg_max=bool(np.all(delta_max<=5)))
    trigger=all(gates.values());diffdeg=float(np.rad2deg(np.arccos(np.clip((np.trace(R@ro.T)-1)/2,-1,1))))
    stats={}
    # Rotation locked and selection computed before reading any held-out statistic.
    for code,name in enumerate(['train','validation','test']):
        sel=z['split']==code;norm=np.linalg.norm(sm-sm.mean(0));sf=(z['xyz_source'][sel,:,0]-sm.mean(0))/norm
        stats[name]=dict(frames=int(sel.sum()),old_necessary_mean_deg=oldlb[sel].mean(0),anatomical_necessary_mean_deg=newlb[sel].mean(0),old_necessary_max_deg=oldlb[sel].max(0),anatomical_necessary_max_deg=newlb[sel].max(0),old_anchor_frozen_normalization_mean=float(np.linalg.norm(sf@ro.T-mn,axis=-1).mean()),anatomical_anchor_frozen_normalization_mean=float(np.linalg.norm(sf@R.T-mn,axis=-1).mean()),longitudinal_anatomical_mean=old.unit((old.unit(lm['He']-lm['Th'])+old.unit(lm['Th']-lm['Abd']))[sel]) .mean(0)@R.T,antenna_left_minus_right_mean=(lm['LA']-lm['RA'])[sel].mean(0)@R.T)
    result=dict(status='DERIVED_HYPOTHESIS',rotation_difference_deg=diffdeg,anatomical_frame=ledger,old_R=ro,anatomical_R=R,old_anchor_per_leg=ar0,anatomical_anchor_per_leg=ar1,old_anchor_mean=float(ar0.mean()),anatomical_anchor_mean=float(ar1.mean()),anchor_mean_increase=float(ar1.mean()-ar0.mean()),middle_train_mean_improvement_deg=improve,each_leg_train_mean_increase_deg=delta_mean,each_leg_train_max_increase_deg=delta_max,gates=gates,full_record_trigger=trigger,selected_R='anatomical' if trigger else 'old',R_independently_identifiable=False,splits=stats,conclusion='fixed anatomical hypothesis meets predeclared gate' if trigger else 'Only global coordinates cannot resolve the inconsistency under these two justified hypotheses; retain old R. Marker/body geometry requires external evidence. This does not exhaust arbitrary SO3.')
    dump('FRAME_COMPARISON.json',result);dump('NECESSARY_ENVELOPES.json',env)
    np.savez_compressed(OUT/'FRAME_ARRAYS.npz',old_R=ro,anatomical_R=R,old_lower_deg=oldlb,anatomical_lower_deg=newlb,source_frame=z['source_frame'],split=z['split'],anatomical_reference_directions=dirs,anatomical_reference_points=refs,direction_mask=dm,point_mask=pm,source_body_landmarks=np.stack([lm[k] for k in ['He','Th','Abd','LA','RA']],axis=1),source_anchor_train_mean=sm,model_anchor_points=a['model_anchor_points'])
    print('frame decision',json.dumps(result,default=lambda x:x.tolist()),flush=True)
    COUNTS['model_compile_attempts']+=1;mj,m,assets=old.compile_model();COUNTS['model_compile']+=1;dump('ASSET_IDENTITY.json',assets)
    data,thorax,bodies,jids,adr,bounds,base,p0,lengths=old.setup(mj,m);COUNTS['kinematics']+=1
    assert np.array_equal(p0[:,0],a['model_anchor_points']) and np.array_equal(bounds,a['joint_bounds_rad'])
    cases=([dict(leg=leg,frame=int(frame)) for frame in z['source_frame'] for leg in old.LEGS] if trigger else prereg()['fallback_cases'])
    qs=[];pts=[];targs=[];records=[];starts=[];previous=base.copy()
    for ci,case in enumerate(cases):
        li=old.LEGS.index(case['leg']);t=int(np.flatnonzero(z['source_frame']==case['frame'])[0]);address=adr[li];dofs=m.jnt_dofadr[jids[li]];lo,hi=bounds[li].T;span=hi-lo
        if trigger:
            if li==0:qbase=previous.copy()
            prior=previous[address].copy()
        else:qbase=a['qpos'][t].copy();prior=a['qpos'][max(0,t-1),address].copy() if t else base[address].copy()
        target=dirs[t,li,:3];cache={}
        def evaluate(v):
            if 'v' not in cache or not np.array_equal(v,cache['v']):
                data.qpos[:]=qbase;data.qpos[address]=v;mj.mj_kinematics(m,data);COUNTS['kinematics']+=1;mj.mj_comPos(m,data);COUNTS['com_pos']+=1
                p=data.xpos[bodies[li,:4]].copy();dv=np.diff(p,axis=0);ln=np.linalg.norm(dv,axis=1);u=dv/ln[:,None];jp=[]
                for bid in bodies[li,:4]:
                    j=np.zeros((3,m.nv));mj.mj_jac(m,data,j,None,data.xpos[bid],int(bid));COUNTS['jacobian']+=1;jp.append(j[:,dofs])
                jd=np.diff(np.array(jp),axis=0);jac=np.concatenate([(np.eye(3)-np.outer(u[k],u[k]))@jd[k]/ln[k] for k in range(3)])
                cache.update(v=v.copy(),p=p,u=u,j=jac)
            return cache
        def fun(v):return np.r_[(evaluate(v)['u']-target).ravel(),np.sqrt(old.TEMPORAL)*(v-prior)/span]
        def jac(v):return np.vstack([evaluate(v)['j'],np.diag(np.sqrt(old.TEMPORAL)/span)])
        rng=np.random.default_rng(20261102);seeds=([prior] if trigger else [base[address],a['qpos'][max(t-1,0),address],a['qpos'][min(t+1,36),address],(lo+hi)/2,rng.uniform(lo,hi),rng.uniform(lo,hi)])
        fits=[]
        for si,seed in enumerate(seeds):
            r=least_squares(fun,np.clip(seed,lo+1e-12,hi-1e-12),jac=jac,bounds=(lo,hi),max_nfev=300,ftol=1e-9,xtol=1e-9,gtol=1e-9);COUNTS['ik_solves']+=1
            geom=float(np.sum((evaluate(r.x)['u']-target)**2));fits.append((geom,r.x.copy(),r));starts.append(dict(case=ci,seed=si,geometry_ss=geom,total_cost=float(r.cost),success=bool(r.success),nfev=int(r.nfev)))
        geom,v,r=min(fits,key=lambda x:x[0]);ev=evaluate(v);q=qbase.copy();q[address]=v;qs.append(q);pts.append(ev['p'].copy());targs.append(target)
        err=np.rad2deg(np.arccos(np.clip(np.sum(ev['u']*target,axis=-1),-1,1)));assert newlb[t,li]<=err[0]+1e-7
        baseline_target=a['reference_directions'][t,li,:3];baseline_ss=float(np.sum((a['achieved_directions'][t,li,:3]-baseline_target)**2))
        records.append(dict(case=ci,**case,row=t,split=int(z['split'][t]),old_geometry_ss=baseline_ss,anatomical_geometry_ss=geom,old_segment_error_deg=np.rad2deg(a['direction_residual_rad'][t,li,:3]),anatomical_segment_error_deg=err,old_necessary_lower_deg=float(oldlb[t,li]),anatomical_necessary_lower_deg=float(newlb[t,li]),success=bool(r.success),native_joint_truth_target=False))
        if trigger:
            qbase=q.copy()
            if li==5:previous=q.copy()
        print('case',ci,case['leg'],case['frame'],'old',baseline_ss,'anatomical',geom,flush=True)
    table('MULTISTART.csv',starts);dump('CASES.json',records)
    np.savez_compressed(OUT/'IK_DIAGNOSTIC.npz',qpos=np.array(qs),achieved_points=np.array(pts),target_directions=np.array(targs),leg_indices=np.array([old.LEGS.index(c['leg']) for c in cases]),source_frames=np.array([c['frame'] for c in cases]),native_joint_truth_target=np.array(False),native_joint_loss_mask=np.zeros((len(cases),7),bool),root_valid=np.zeros(len(cases),bool),Claw_valid=np.zeros(len(cases),bool))
    selected=R if trigger else ro;sd,sdm,sp,spm=old.targets(z,selected,a['model_anchor_points'],a['model_chord_lengths'])
    # Observable targets remain actual source geometry, never achieved IK substituted for observations.
    np.savez_compressed(OUT/'TRAIN_LOCKED_GEOMETRY_TARGETS.npz',rotation_source_to_model=selected,source_xyz=z['xyz_source'],source_frame=z['source_frame'],split=z['split'],time_relative_s=z['time_relative_s'],reference_directions=sd,direction_valid=sdm,reference_points=sp,point_valid=spm,native_joint_truth_target=np.array(False),native_joint_loss_mask=np.zeros((37,6,7),bool),root_valid=np.zeros(37,bool),runtime_enabled=np.array(False),model_conversion_floor_deg=(newlb if trigger else oldlb),independently_calibrated=np.array(False))
    dump('EXECUTION.json',dict(success=True,executable=sys.executable,counters=COUNTS,full_record_solves=222 if trigger else 0,diagnostic_solves=0 if trigger else 24,baseline_new_solves=0))

def verify_arrays():
    a=old.load(old.OUT/'RETARGETED_TARGETS.npz');f=load('FRAME_ARRAYS.npz');z=old.load(old.SOURCE/'NMF2_CONTINUOUS.npz');rows,lm=raw()
    # Independent scalar construction from CSV; no use of anatomical()/necessary().
    vec=lambda name,t:np.array([float(rows[t][name+'_'+c]) for c in 'xyz'])
    unit=lambda v:v/math.sqrt(sum(float(x*x) for x in v))
    fw=np.zeros(3);lat=np.zeros(3)
    for t in range(22):
        fw+=(unit(vec('He',t)-vec('Th',t))+unit(vec('Th',t)-vec('Abd',t)))/44
        lat+=sum((vec(l+'-ThC',t) for l in ['LF','LM','LH']),np.zeros(3))/66-sum((vec(l+'-ThC',t) for l in ['RF','RM','RH']),np.zeros(3))/66
    x=unit(fw);y=unit(lat-x*sum(x*lat));R=np.array([x,y,np.cross(x,y)]);rerr=float(np.max(np.abs(R-f['anatomical_R'])));assert rerr<1e-14
    env=json.loads((OUT/'NECESSARY_ENVELOPES.json').read_text());mx=0.
    for nm,dirs in [('old',a['reference_directions']),('anatomical',f['anatomical_reference_directions'])]:
        for t in range(37):
            for li,e in enumerate(env):
                d=dirs[t,li,0];lb=max(0,math.acos(max(-1,min(1,-d[2])))-e['cap_rad'])
                for k,s in [(0,'x'),(1,'y')]:
                    lat=math.asin(max(-1,min(1,float(d[k]))));lb=max(lb,lat-math.asin(e[s+'max']),math.asin(e[s+'min'])-lat)
                mx=max(mx,abs(math.degrees(lb)-f[nm+'_lower_deg'][t,li]))
    assert mx<1e-10
    target=load('TRAIN_LOCKED_GEOMETRY_TARGETS.npz');decision=json.loads((OUT/'FRAME_COMPARISON.json').read_text());tr=z['split']==0
    ar=[]
    sm=z['xyz_source'][tr,:,0].mean(0);mn=normalized_layout(a['model_anchor_points'])
    for RR in [f['old_R'],R]:ar.append(np.linalg.norm(normalized_layout(sm)@RR.T-mn,axis=1).mean())
    delta=f['anatomical_lower_deg'][tr].mean(0)-f['old_lower_deg'][tr].mean(0);dm=f['anatomical_lower_deg'][tr].max(0)-f['old_lower_deg'][tr].max(0)
    imp=(f['old_lower_deg'][tr][:,[1,4]]-f['anatomical_lower_deg'][tr][:,[1,4]]).mean();gate=ar[1]-ar[0]<=.05 and imp>=10 and np.all(delta<=2) and np.all(dm<=5)
    assert bool(gate)==decision['full_record_trigger'];expected=R if gate else f['old_R'];assert np.max(np.abs(target['rotation_source_to_model']-expected))<1e-14
    d,dm,p,pm=old.targets(z,expected,a['model_anchor_points'],a['model_chord_lengths']);assert np.allclose(target['reference_points'],p,equal_nan=True) and np.allclose(target['reference_directions'],d)
    assert not target['native_joint_loss_mask'].any() and not target['root_valid'].any() and not target['direction_valid'][:,:,3].any() and not target['point_valid'][:,:,4].any()
    assert identities()==json.loads((OUT/'SOURCE_IDENTITY.json').read_text())
    dump('INDEPENDENT_ARRAY_VERIFICATION.json',dict(passed=True,fresh_process=True,frame_maxabs=rerr,scalar_certificate_maxabs_deg=mx,selection_recomputed=True,source_hashes_exact=True,train_only=True,counters=dict(model_compile=0,ik_solves=0,body_steps=0,neural_steps=0,network_inits=0,training=0)))

def verify_fk():
    b=load('IK_DIAGNOSTIC.npz');a=old.load(old.OUT/'RETARGETED_TARGETS.npz');contract=json.loads((old.OUT/'MODEL_MARKER_CONTRACT.json').read_text());mj,m,assets=old.compile_model();d=mj.MjData(m);err=0.;rows=[]
    for i,q in enumerate(b['qpos']):
        li=int(b['leg_indices'][i]);j=contract['joints'][li*7:li*7+7];ad=np.array([x['qpos_address'] for x in j]);bounds=np.array([x['range_rad'] for x in j]);assert np.all(q[ad]>=bounds[:,0]-1e-12) and np.all(q[ad]<=bounds[:,1]+1e-12);assert np.array_equal(q[:7],[0,0,0,1,0,0,0])
        d.qpos[:]=q;mj.mj_forward(m,d);p=np.array([d.xpos[m.body(x['model_body']).id].copy() for x in contract['legs'][li]['markers'][:4]]);err=max(err,float(np.max(np.abs(p-b['achieved_points'][i]))))
        u=old.unit(np.diff(p,axis=0));ss=float(np.sum((u-b['target_directions'][i])**2));rows.append(dict(case=i,geometry_ss=ss))
    cases=json.loads((OUT/'CASES.json').read_text());assert err<1e-10 and all(abs(x['anatomical_geometry_ss']-y['geometry_ss'])<1e-12 for x,y in zip(cases,rows));assert assets==json.loads((OUT/'ASSET_IDENTITY.json').read_text())
    dump('INDEPENDENT_FK_VERIFICATION.json',dict(passed=True,fresh_process=True,fk_maxabs=err,cases=rows,counters=dict(model_compile=1,forward=len(rows),ik_solves=0,body_steps=0,neural_steps=0,network_inits=0,training=0)))

def seal():
    assert not (OUT/'MANIFEST_SHA256.json').exists();assert json.loads((OUT/'PARENT_REVIEW.json').read_text())['passed']
    assert json.loads((OUT/'INDEPENDENT_ARRAY_VERIFICATION.json').read_text())['passed'] and json.loads((OUT/'INDEPENDENT_FK_VERIFICATION.json').read_text())['passed']
    assert identities()==json.loads((OUT/'SOURCE_IDENTITY.json').read_text())
    dump('MANIFEST_SHA256.json',{str(p):old.sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','verify_arrays','verify_fk','seal']);args=parser.parse_args()
    try:globals()[args.mode]()
    except Exception:
        if OUT.exists() and not (OUT/'MANIFEST_SHA256.json').exists():dump('FAILED_'+args.mode+'.json',dict(traceback=traceback.format_exc(),counters=COUNTS))
        raise
