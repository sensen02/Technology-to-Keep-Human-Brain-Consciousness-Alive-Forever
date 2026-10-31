"""S112 saved-array-only COM/contact verification. Never imports a simulator or neural engine.

Thresholds are fixed before outcomes: instantaneous maximum component residual
<=1e-8 AND relative<=1e-6; impulse all prefix0..17 maximum component residual
<=1e-4 AND relative<=1e-2 with scale M*max norm(COM velocity). Also report the
supplemental additive instantaneous bound. Failed scientific checks are recorded, not raised. Malformed/missing evidence is recorded by the CLI as a failed report.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ('baseline', 'pool15_up')
BINS = ((0,5),(5,10),(10,20),(20,40),(40,80))
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s112-com-contact-probe/20261024T055500Z')
FORCE_ABS, FORCE_REL, IMPULSE_ABS, IMPULSE_REL = 1e-8, 1e-6, 1e-4, 1e-2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def require(ok, label):
    if not ok:
        raise ValueError(label)


class Checks:
    def __init__(self): self.rows = []
    def add(self, name, ok, **observed):
        self.rows.append(dict(name=name, passed=bool(ok), **observed))
        return bool(ok)
    def eq(self, name, a, b, atol=0):
        a,b = np.asarray(a),np.asarray(b)
        same = a.shape == b.shape
        err = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if same and a.size else (0. if same else None)
        return self.add(name, same and bool(np.allclose(a,b,rtol=0,atol=atol)), actual_shape=list(a.shape), expected_shape=list(b.shape), max_abs_error=err, atol=atol)
    @property
    def passed(self): return all(x['passed'] for x in self.rows)


def reconstruct_fir(spikes, pools, mapping, weights):
    require(spikes.ndim==2 and np.all((spikes==0)|(spikes==1)), 'nonbinary spikes')
    history=np.zeros((80,len(pools))); fs=[]; us=[]
    for spike in spikes:
        history[1:]=history[:-1].copy()
        history[0]=[np.mean(spike[list(p)])/.0005/100 for p in pools]
        f=np.stack([history[lo:hi].mean(0) for lo,hi in BINS],axis=1)
        pooled=np.clip((f*weights).sum(1),0,1); u=np.zeros(len(mapping)); mapped=mapping>=0
        u[mapped]=pooled[mapping[mapped]]; fs.append(f); us.append(u)
    return np.asarray(fs),np.asarray(us),history


def resolve_key(a, *names):
    for name in names:
        if name in a: return a[name]
    raise KeyError('required raw field absent: '+ ' / '.join(names))


def reconstruct(a, meta, checks, label):
    """Only raw per-body COM/world kinematics and contact records enter algebra."""
    mass=resolve_key(a,'body_mass')
    pos=resolve_key(a,'body_xipos','xipos','body_com_position')
    vel6=resolve_key(a,'body_linear_velocity_world','body_velocity_world','body_velocity','body_cvel_world')
    acc6=resolve_key(a,'body_linear_acceleration_world','body_acceleration_world','body_acceleration','body_cacc_world')
    # Match per-state BLAS summation ordering for archive identity checks;
    # the external-force closure below remains an independently reconstructed check.
    jacv=np.stack([j@v for j,v in zip(a['body_com_jacp'],a['qvel'])])
    jaca=np.stack([j@ac+jd@v for j,jd,ac,v in zip(a['body_com_jacp'],a['body_com_jacdot'],a['qacc'],a['qvel'])])
    checks.eq(label+' independent Jp*qvel BODY COM velocity',vel6,jacv,2e-12)
    checks.eq(label+' independent Jp*qacc+Jdot*qvel BODY COM acceleration',acc6,jaca,2e-12)
    vel=vel6[...,3:] if vel6.shape[-1]==6 else vel6
    acc=acc6[...,3:] if acc6.shape[-1]==6 else acc6
    if mass.ndim==2:
        checks.eq(label+' constant body masses',mass,np.broadcast_to(mass[0],mass.shape)); mass=mass[0]
    require(pos.shape==vel.shape==acc.shape and pos.shape[0]==31 and pos.shape[-1]==3,'raw body kinematic dimensions')
    require(mass.shape==(pos.shape[1],) and mass[0]==0 and np.all(mass>=0) and mass.sum()>0,'body mass schema')
    M=float(mass.sum()); com=np.einsum('b,tbi->ti',mass,pos)/M
    v=np.stack([mass@value/M for value in vel]); acccom=np.stack([mass@value/M for value in acc])
    ncon=resolve_key(a,'contact_ncon','ncon'); frame=resolve_key(a,'contact_frame')
    raw=resolve_key(a,'contact_raw6','contact_force_raw6','contact_wrench_local')
    point=resolve_key(a,'contact_pos_world','contact_pos','contact_point','contact_position')
    bodies=resolve_key(a,'contact_body_ids','contact_body','contact_bodies','contact_bodyid')
    geoms=resolve_key(a,'contact_geom_ids','contact_geom','contact_geoms','contact_geomid')
    mask=resolve_key(a,'contact_external_mask','external_contact_mask')
    saved=np.concatenate((a['contact_force_on_geom2_world'],a['contact_intrinsic_torque_on_geom2_world']),axis=2)
    require(raw.ndim==3 and raw.shape[0]==31 and raw.shape[2]==6,'contact raw6 dimensions')
    cap=raw.shape[1]; require(bodies.shape==geoms.shape==(31,cap,2),'padded contact identity dimensions')
    require(point.shape==(31,cap,3) and mask.shape==(31,cap) and saved.shape==raw.shape,'padded contact dimensions')
    frame=frame.reshape(31,cap,3,3)
    valid=np.arange(cap)[None,:]<ncon[:,None]
    checks.add(label+' ncon integer/in range',np.issubdtype(ncon.dtype,np.integer) and np.all((ncon>=0)&(ncon<=cap)))
    external=valid & ((bodies[:,:,0]==0) ^ (bodies[:,:,1]==0))
    checks.eq(label+' external mask world body0 only/self excluded',mask,external)
    sign=np.where(bodies[:,:,0]==0,1.,-1.)
    # Raw six-vector is FORCE then TORQUE ON BODY2, contact frame rows are axes.
    worldf=np.einsum('tcji,tcj->tci',frame,raw[:,:,:3])
    worldm=np.einsum('tcji,tcj->tci',frame,raw[:,:,3:])
    onbody2=np.concatenate((worldf,worldm),axis=2)
    checks.eq(label+' saved world wrench ON BODY2',saved[valid],onbody2[valid],2e-12)
    if np.any(valid):
        gram=np.einsum('tcik,tcjk->tcij',frame,frame)
        checks.eq(label+' contact frame orthonormal',gram[valid],np.broadcast_to(np.eye(3),gram[valid].shape),2e-12)
        checks.add(label+' contact frame right handed',np.allclose(np.linalg.det(frame[valid]),1,rtol=0,atol=2e-12))
    signedf=worldf*sign[:,:,None]*external[:,:,None]
    signedm=worldm*sign[:,:,None]*external[:,:,None]
    contactF=signedf.sum(1)
    contactT=(np.cross(point-com[:,None,:],signedf)+signedm).sum(1)
    checks.eq(label+' contact valid mask',a['contact_valid_mask'],valid)
    checks.eq(label+' external signs BODY2 positive when BODY1 world',a['contact_external_sign'],sign*external)
    checks.add(label+' required sign label body2', 'geom2' in meta['contact_sign'] and 'geom1' in meta['contact_sign'])
    geom_body=np.asarray(meta['geom_bodyid'])
    checks.eq(label+' geom/body identity',bodies[valid],geom_body[geoms[valid]])
    checks.eq(label+' contact torque ON BODY2 about COM',a['contact_torque_on_geom2_about_com'][valid],(worldm+np.cross(point-com[:,None,:],worldf))[valid],2e-12)
    gravity=np.asarray(meta['gravity_world'],float)
    if gravity.ndim==1: gravity=np.broadcast_to(gravity,(31,3))
    require(gravity.shape==(31,3),'gravity dimensions')
    gravityF=M*gravity
    gravityT=np.cross(pos-com[:,None,:],mass[None,:,None]*gravity[:,None,:]).sum(1)
    checks.eq(label+' gravity torque about COM zero',gravityT,np.zeros((31,3)),1e-12)
    xfrc=resolve_key(a,'xfrc_applied'); qfrc=resolve_key(a,'qfrc_applied')
    checks.add(label+' applied external forces zero',not xfrc.any() and not qfrc.any())
    # xfrc is force:torque at each body COM. Still included, not silently dropped.
    xF=xfrc[:,1:,:3].sum(1)
    xT=(np.cross(pos[:,1:]-com[:,None,:],xfrc[:,1:,:3])+xfrc[:,1:,3:]).sum(1)
    netF=contactF+gravityF+xF; netT=contactT+gravityT+xT
    ma=M*acccom; residual=ma-netF; scale=np.maximum(np.abs(ma),np.abs(netF))
    threshold=FORCE_ABS+FORCE_REL*scale
    forcepass=np.abs(residual)<=threshold
    maxerr=float(np.abs(residual).max()); globalscale=max(float(np.abs(ma).max()),float(np.abs(netF).max()),1e-12)
    checks.add(label+' instantaneous COM force balance',maxerr<=FORCE_ABS and maxerr/globalscale<=FORCE_REL, max_abs_residual=maxerr, relative_residual=maxerr/globalscale, scale=globalscale, absolute_tolerance=FORCE_ABS,relative_tolerance=FORCE_REL,criterion='BOTH absolute and relative bounds, all31 samples', supplemental_additive_bound_also_passed=bool(forcepass.all()), max_additive_threshold_ratio=float((np.abs(residual)/threshold).max()), failed_additive_sample_components=np.argwhere(~forcepass).tolist())
    t=a['time']; impulse=np.vstack((np.zeros(3),np.cumsum(.5*(netF[1:]+netF[:-1])*np.diff(t)[:,None],axis=0)))
    mdv=M*(v-v[0]); impulse_err=float(np.abs((impulse-mdv)[:18]).max()); impulse_scale=max(float(M*np.linalg.norm(v[:18],axis=1).max()),1e-12)
    checks.add(label+' impulse inclusive indices0..17 0..1.7ms',impulse_err<=IMPULSE_ABS and impulse_err/impulse_scale<=IMPULSE_REL, impulse=impulse[:18], mass_delta_velocity=mdv[:18], residual=(impulse-mdv)[:18], max_abs_residual=impulse_err, relative_residual=impulse_err/impulse_scale, scale=impulse_scale,criterion='BOTH absolute and relative bounds, every inclusive prefix0..17; scale M*max norm(v_COM)', absolute_tolerance=IMPULSE_ABS, relative_tolerance=IMPULSE_REL, sample_indices=list(range(18)))
    fd=np.gradient(v,t,axis=0,edge_order=2); fderr=fd[1:-1]-acccom[1:-1]
    # Crosscheck is diagnostic only: no post-hoc acceptance threshold invented.
    fdcheck=dict(window_ms=[.1,2.9],method='numpy.gradient(COM_velocity,time,edge_order=2), interior indices1..29',max_abs_error=float(np.abs(fderr).max()),rms_error=float(np.sqrt(np.mean(fderr*fderr))),acceptance_threshold=None,acceptance_role='diagnostic only; discrete differentiation/integration truncation not instant API balance')
    checks.add(label+' FD crosscheck finite',np.isfinite(fderr).all(),**fdcheck)
    for key,expected in (('com_position_world',com),('com_velocity_world',v),('com_acceleration_world',acccom),('com_mass_times_acceleration_world',ma),('com_momentum_world',M*v),('contact_force_world',contactF),('contact_torque_about_com',contactT),('gravity_force_world',gravityF),('gravity_torque_about_com',gravityT),('external_force_world',netF),('external_torque_about_com',netT),('external_impulse_world',impulse),('com_momentum_change_world',mdv)):
        checks.eq(label+' saved '+key,a[key],expected,2e-12)
    for name,lhs,rhs,ab,re in (('preregistered positionFD velocity',np.gradient(com,t,axis=0,edge_order=2),v,.1,.01),('preregistered momentumFD force',np.gradient(M*v,t,axis=0,edge_order=2),netF,1.,.05)):
        er=float(np.abs((lhs-rhs)[1:18]).max()); sc=max(float(np.abs(lhs[1:18]).max()),float(np.abs(rhs[1:18]).max()),1e-12)
        checks.add(label+' '+name,er<=ab and er/sc<=re,max_abs_residual=er,relative_residual=er/sc,absolute_tolerance=ab,relative_tolerance=re,scope='runner diagnostic gate indices1..17; primary force/impulse separate')
    return dict(mass=M,com=com,velocity=v,acceleration=acccom,fd_acceleration=fd,contact_force=contactF,gravity_force=gravityF,net_force=netF,contact_torque=contactT,net_torque=netT,force_residual=residual,external_mask=external,contact_point=point,contact_normal=frame[:,:,0,:],raw=raw,signed_contact_force=signedf,body_mass=mass,body_position=pos,body_velocity=vel,body_acceleration=acc,fd_crosscheck=fdcheck)


def serial(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=serial,allow_nan=False)+'\n')


def verify(out):
    require('mujoco' not in sys.modules,'physics imported')
    checks=Checks(); pre=js(out/'PREREG.json'); hashes=pre['source_hashes']; artifacts={}
    checks.add('runner complete',js(out/'COMPLETE.json')['status']=='COMPLETE' and not (out/'FAILED.json').exists())
    checks.add('required runner committed',str(ROOT/'run_com_contact_probe.py') in hashes)
    for path,digest in hashes.items(): checks.add('source SHA '+path,sha(path)==digest)
    checks.add('INPUT_HASHES agreement',js(out/'INPUT_HASHES.json')==hashes)
    for path,digest in js(out/'MODEL_DEPENDENCIES.json').items(): checks.add('dependency SHA '+path,hashes.get(path)==digest and sha(path)==digest)
    ledger=js(out/'CALL_LEDGER.json')
    checks.add('exact call ledger',ledger==dict(model_load=2,forward=62,step=60,neural_ticks=172,active=None,completed=list(ARMS)),observed=ledger)
    checks.add('preregistered timing',pre['initialization_ticks']==80 and pre['physical_duration_ms']==3 and pre['dt_ms']==.1 and pre['neural_dt_ms']==.5)
    sourcew=npz(pre['same_weights'])['weights']; direction=npz(out/'DIRECTION.npz')
    checks.eq('source direction W',direction['weights'],sourcew)
    manifests=[Path(p) for p in hashes if Path(p).name=='MANIFEST.json']
    require(len(manifests)==1,'one mapping manifest required'); manifest=js(manifests[0])
    rows=sorted(manifest['rows'],key=lambda r:r['actuator_index'])
    require([r['actuator_index'] for r in rows]==list(range(90)),'manifest actuator rows')
    pools=sorted({tuple(sorted(r['local_indices'])) for r in rows if r['local_indices']})
    lookup={p:i for i,p in enumerate(pools)}; mapping=np.asarray([lookup.get(tuple(sorted(r['local_indices'])),-1) for r in rows])
    checks.add('pool identity',len(pools)==66 and mapping[4]==15 and manifest['n_neurons']==17099)
    results=js(out/'RESULTS.json'); checks.add('exact arm order',[r['label'] for r in results]==list(ARMS))
    plotted=[]; reference=None
    for result in results:
        label=result['label']; d=out/label
        required=('TRAJECTORY.npz','INITIALIZATION.npz','NEURAL_COLD.npz','CONNECTIVITY.npz','NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','NEURAL_FINAL.npz','INITIAL_OBSERVATION.json','OBSERVATION_00.json','SCHEDULE.json','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','COM_CLOSURE.json','MODEL_PROVENANCE.json')
        committed={Path(path).resolve() for path in result['hashes']}
        checks.add(label+' unique canonical artifact paths',len(committed)==len(result['hashes']))
        for name in required: checks.add(label+' artifact committed '+name,(d/name).resolve() in committed)
        for path,digest in result['hashes'].items():
            observed=sha(path); checks.add('artifact SHA '+path,observed==digest); artifacts[path]=observed
        checks.add(label+' direction hash',result['direction_sha256']==sha(out/'DIRECTION.npz'))
        for filename,key in (('TRAJECTORY.npz','trajectory_sha256'),('INITIALIZATION.npz','initialization_sha256')): checks.add(label+' '+key,sha(d/filename)==result[key])
        a=npz(d/'TRAJECTORY.npz'); init=npz(d/'INITIALIZATION.npz'); cold=npz(d/'NEURAL_COLD.npz'); csr=npz(d/'CONNECTIVITY.npz'); meta=js(d/'COM_CONTACT_METADATA.json')
        for k,value in a.items(): checks.add(label+' finite '+k,np.isfinite(value).all())
        w=sourcew.copy()
        if label=='pool15_up': w[15,3]*=1.2
        checks.eq(label+' exact330 weights',a['weights'],w); checks.eq(label+' init weights',init['weights'],w); checks.eq(label+' direction weights',direction[label],w)
        checks.add(label+' bounds/zero mask',np.all((w>=0)&(w<=4)) and np.array_equal(w==0,sourcew==0))
        checks.eq(label+' actuator mapping',a['actuator_pool'],mapping)
        checks.eq(label+' physical clock',a['time'],np.arange(31)*.0001,1e-12)
        for key,shape in dict(qpos=(31,49),qvel=(31,48),qacc=(31,48),spikes=(6,17099),features=(6,66,5),applied=(6,90)).items(): checks.add(label+' shape '+key,a[key].shape==shape)
        for key in ('qvel','act','ctrl'):
            checks.add(label+' zero initial '+key,not a[key][0].any() and not init['body_'+key].any())
        checks.add(label+' no warnings',not a['warnings'].any())
        checks.eq(label+' initial qpos',a['qpos'][0],init['body_qpos'])
        fs,us,history=reconstruct_fir(np.concatenate((init['spikes'],a['spikes'])),pools,mapping,w)
        checks.eq(label+' warm FIR features',init['features'],fs[:80],1e-14); checks.eq(label+' warm commands',init['decoded'],us[:80],1e-14)
        checks.eq(label+' runtime FIR features',a['features'],fs[80:],1e-14); checks.eq(label+' runtime commands',a['applied'],us[80:],1e-14)
        checks.eq(label+' release features',init['release_features'],fs[79],1e-14); checks.eq(label+' release command',init['release_decoded'],us[79],1e-14)
        checks.eq(label+' warm history',init['history'],reconstruct_fir(init['spikes'],pools,mapping,w)[2],1e-14)
        checks.eq(label+' final FIR',npz(d/'NEURAL_FINAL.npz')['FIR_history'],history,1e-14)
        checks.eq(label+' control five steps hold',a['ctrl'][1:],np.repeat(a['applied'],5,axis=0))
        checks.eq(label+' actual sensory held current',a['current'],np.repeat(init['current'][None],6,axis=0))
        schedule=js(d/'SCHEDULE.json'); checks.add(label+' schedule six rows',len(schedule)==6)
        for k,row in enumerate(schedule):
            checks.add(label+' schedule ids '+str(k),row['tick']==k and row['consume_id']==80+k)
            for key,value in dict(source_start_s=(80+k)*.0005,source_end_s=(81+k)*.0005,physical_start_s=k*.0005,physical_end_s=(k+1)*.0005,source_offset_s=.04).items(): checks.add(label+' schedule '+str(k)+key,abs(row[key]-value)<=1e-12)
        for filename,steps,ms in (('NEURAL_COLD_META.json',0,0),('NEURAL_RELEASE_META.json',80,40),('NEURAL_FINAL_META.json',86,43)):
            scalars=js(d/filename)['neural_scalars']; checks.add(label+' neural clock '+filename,scalars['steps']==steps and abs(scalars['t_ms']-ms)<=1e-10)
        prov=js(d/'MODEL_PROVENANCE.json'); checks.add(label+' integrator only',prov['changed_numeric_fields']==['opt.integrator'] and [k for k in prov['before'] if prov['before'][k]!=prov['after'][k]]==['opt.integrator'] and prov['after']['opt.integrator']==1)
        if reference is None: reference=(init,cold,csr,prov,d)
        ri,rc,rs,rp,rd=reference
        for name,current,ref in (('cold',cold,rc),('CSR',csr,rs)):
            checks.add(label+' matched '+name+' keys',set(current)==set(ref))
            for key in ref: checks.eq(label+' matched '+name+' '+key,current[key],ref[key])
        for key in set(init)-{'weights','decoded','release_decoded'}: checks.eq(label+' matched initialization '+key,init[key],ri[key])
        for name in ('NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','INITIAL_OBSERVATION.json'):
            checks.add(label+' matched '+name,js(d/name)==js(rd/name))
        checks.add(label+' matched model provenance',prov==rp)
        rec=reconstruct(a,meta,checks,label)
        checks.eq(label+' signed actuator4 DOF8 torque identity',a['actuator_force'][:,4]*a['actuator_moment'][:,4,8],(a['actuator_force'][:,:,None]*a['actuator_moment'])[:,4,8])
        checks.eq(label+' actuator generalized projection',np.einsum('tu,tud->td',a['actuator_force'],a['actuator_moment']),a['qfrc_actuator'],2e-12)
        plotted.append((label,a,rec,result))
    require('mujoco' not in sys.modules,'physics module imported')
    return checks,plotted,dict(source_hashes=hashes,observed_artifact_hashes=artifacts,ledger=ledger)


def plot(out,plotted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,2,figsize=(16,19),sharex=True)
    for label,a,r,result in plotted:
        t=a['time']*1000; weight=result['weight_native']; ls='-' if label=='baseline' else '--'
        axes[0,0].plot(t,a['foot_force'][:,:,2].sum(1)/weight,ls,label=label+' allfoot Fz/weight')
        axes[0,0].plot(t,r['contact_force'][:,2]/weight,ls,alpha=.6,label=label+' all external contact Fz/weight')
        for j,axis in enumerate('xyz'):
            axes[0,1].plot(t,r['net_force'][:,j],ls,label=label+' F'+axis)
            axes[1,0].plot(t,r['net_torque'][:,j],ls,label=label+' M'+axis)
            axes[1,1].plot(t,r['velocity'][:,j],ls,label=label+' v'+axis)
            axes[2,0].plot(t,r['acceleration'][:,j],ls,label=label+' a'+axis)
            axes[2,0].plot(t[1:-1],r['fd_acceleration'][1:-1,j],':',alpha=.5)
            axes[2,1].plot(t,r['force_residual'][:,j],ls,label=label+' residual'+axis)
        axes[3,0].plot(t,a['actuator_force'][:,4]*a['actuator_moment'][:,4,8],ls,label=label+' act4 torque DOF8')
        axes[3,1].plot(t,a['contact_ncon'],ls,label=label+' all contacts')
        axes[3,1].plot(t,r['external_mask'].sum(1),ls,label=label+' external contacts')
        ext=r['external_mask']; norm=r['contact_normal']; point=r['contact_point']
        for i in range(len(t)):
            use=ext[i]
            axes[4,0].scatter(np.repeat(t[i],use.sum()),point[i,use,2],s=3,alpha=.45,label=label if i==0 else None)
            axes[4,1].scatter(np.repeat(t[i],use.sum()),norm[i,use,2],s=3,alpha=.45,label=label if i==0 else None)
    titles=('all-foot and all external-contact load','Net external F = contacts + gravity + applied','Net external moment about whole-body COM','Whole-body COM velocity world','Primary COM acceleration; dots = FD velocity','M*aCOM - external F','Actual actuator4 torque DOF8','Contact cardinality (self excluded from external)','External contact point z world','External contact frame normal z world')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10); ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=2); ax.set_xlim(0,3); ax.axvline(1.7,color='grey',ls=':',alpha=.6)
    axes[0,0].axhline(.8,color='black',ls=':')
    for ax in axes[-1]: ax.set_xlabel('Physical time (ms); native units NOT_ESTABLISHED')
    fig.suptitle('S112 independent saved-array COM/contact diagnostic: baseline vs pool15_up\nNo physics/neural execution; no standing/walking or checkpoint-selection claim')
    fig.tight_layout(); path=out/'COM_COMPARISON.png'; fig.savefig(path,dpi=150); plt.close(fig); return path


def pair_details(plotted):
    left,a,ra,_=plotted[0]; right,b,rb,_=plotted[1]
    curves={}
    for name in ('com','velocity','acceleration','contact_force','net_force','contact_torque','net_torque'):
        curves[name]=dict(reference=ra[name],intervention=rb[name],delta=rb[name]-ra[name])
    for name,x,y in (('actuator4_force',a['actuator_force'][:,4],b['actuator_force'][:,4]),('actuator4_torque_DOF8',a['actuator_force'][:,4]*a['actuator_moment'][:,4,8],b['actuator_force'][:,4]*b['actuator_moment'][:,4,8]),('allfoot_Fz',a['foot_force'][:,:,2].sum(1),b['foot_force'][:,:,2].sum(1))): curves[name]=dict(reference=x,intervention=y,delta=y-x)
    geometry={}
    for label,arr,rec in ((left,a,ra),(right,b,rb)):
        geometry[label]={key:arr[key] for key in ('contact_ncon','contact_valid_mask','contact_geom_ids','contact_body_ids','contact_pos_world','contact_frame','contact_dist','contact_force_raw6','contact_force_on_geom2_world','contact_intrinsic_torque_on_geom2_world','contact_external_mask','contact_external_sign','contacts','foot_force')}
        geometry[label].update(contact_normals_world=rec['contact_normal'],self_contact_counts=(arr['contact_valid_mask'] & ~rec['external_mask']).sum(1),external_contact_counts=rec['external_mask'].sum(1),foot_contacts_penetrating=arr['contacts'].sum(1),contact_geometry_pairing='Padded row IDs are live contact order, NOT stable material-point identity across arms or time; inspect geom IDs with point/frame/dist.')
    return dict(reference=left,intervention=right,time_ms=a['time']*1000,curves=curves,contact_geometry=geometry,raw_body_arrays_saved_in_trajectories=True,convention='World axes; raw contact force/intrinsic torque on BODY2(geom2). For world BODY1 use +; world BODY2 use -. Self contacts excluded.',interpretation='Paired trajectory differences, not unique causal attribution or added intervention',physical_units='NOT_ESTABLISHED',behavior_claim='NONE',selection=False)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=DEFAULT)
    out=p.parse_args().output.resolve(); report=dict(passed=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_physics_calls=0,new_neural_ticks=0,physical_units='NOT_ESTABLISHED',behavior_claim='NONE')
    try:
        checks,plotted,identities=verify(out)
        details=pair_details(plotted); pairpath=out/'COM_PAIR_DETAILS.json'; write_json(pairpath,details)
        report.update(identities,passed=checks.passed,checks=checks.rows,pair_details=str(pairpath),comparison_png=str(plot(out,plotted)),per_arm_derived={label:{k:v for k,v in r.items() if k not in ('body_position','body_velocity','body_acceleration')} for label,a,r,result in plotted},caveats=['Kinematic COM primary acceleration independently reconstructed from saved jacBodyCom and jacDot, not root freejoint qacc. Raw objectAcceleration after rnePostConstraint is auxiliary only.','No independently rederived simulator Jacobians or physical solver; hashes bind saved evidence, not proof against all unrecorded calls.','FD velocity vs acceleration crosscheck interior .1..2.9ms diagnostic has no invented acceptance tolerance.','Computational initialization is not physical support; 3ms diagnostic is not standing/walking success.'])
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    write_json(out/'COM_INDEPENDENT_CHECK.json',report)
    print(json.dumps(dict(passed=report['passed'],report=str(out/'COM_INDEPENDENT_CHECK.json'),error=report.get('error')),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__': sys.exit(main())
