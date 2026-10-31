"""S113/P1/S2-S4 offline timestep evidence; NumPy algebra, no simulator/neural run.

Fixed acceptance: instantaneous abs1e-8 AND rel1e-6; inclusive 0..1.7ms
impulse abs1e-4 AND rel.01 (M*max COM speed); momentum FD abs1 AND rel.05.
Original np.gradient(edge_order=2), centered indices1..1.7ms, no deletion.
Matplotlib is used only for the required figure. Native units NOT_ESTABLISHED.
"""
from pathlib import Path
import argparse
import csv
import itertools
import json
import sys
import traceback
import xml.etree.ElementTree as ET
import numpy as np
from verify_com_contact_probe import Checks, require, sha, js, npz, serial, write_json, reconstruct_fir

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/sensen/Desktop/engine修复验收/s112-com-contact-probe/20261024T055500Z')
DEFAULT = Path('/home/sensen/Desktop/engine修复验收/s113-com-timestep/20261024T060500Z')
ARMS = ('dt50us', 'dt25us')
GATES = ('height', 'angle', 'legs', 'load', 'nonfoot', 'limit')


def canonical_checks(checks, mapping, prefix, observed):
    resolved = [str(Path(p).resolve()) for p in mapping]
    checks.add(prefix+' unique canonical paths', len(resolved)==len(set(resolved)))
    for path, digest in mapping.items():
        p = Path(path)
        checks.add(prefix+' existing absolute source '+path, p.is_absolute() and p.resolve().is_file())
        actual = sha(p); observed[path] = actual
        checks.add(prefix+' SHA '+path, actual==digest, expected_sha256=digest, observed_sha256=actual)


def measure(lhs, rhs, ids, absolute, relative):
    delta = lhs-rhs
    error = float(np.abs(delta[ids]).max())
    scale = max(float(np.abs(lhs[ids]).max()), float(np.abs(rhs[ids]).max()), 1e-12)
    local, component = np.unravel_index(np.argmax(np.abs(delta[ids])), delta[ids].shape)
    index = int(ids[local])
    return dict(passed=bool(error<=absolute and error/scale<=relative), max_abs_residual=error,
                relative_residual=error/scale, global_denominator=scale,
                absolute_tolerance=absolute, relative_tolerance=relative,
                criterion='absolute AND relative; maximum across all 3 components on the SAME window',
                sample_indices=ids.tolist(), worst_index=index, worst_component='xyz'[component],
                worst_component_index=int(component), lhs_at_worst=float(lhs[index,component]),
                rhs_at_worst=float(rhs[index,component]), signed_residual_at_worst=float(delta[index,component]))


def reconstruct(a, meta, checks, label):
    """Variable-shape raw reconstruction; no saved aggregate enters primary balance."""
    t = a['time']; n = len(t)
    mass = a['body_mass']
    if mass.ndim==2:
        checks.eq(label+' masses constant',mass,np.broadcast_to(mass[0],mass.shape)); mass=mass[0]
    pos = a['body_xipos']; J = a['body_com_jacp']; Jd = a['body_com_jacdot']
    require(pos.shape==(n,len(mass),3) and mass[0]==0 and np.all(mass>=0) and mass.sum()>0,'body mass/position dimensions')
    vel = np.stack([j@v for j,v in zip(J,a['qvel'])])
    acc = np.stack([j@ac+jd@v for j,jd,ac,v in zip(J,Jd,a['qacc'],a['qvel'])])
    checks.eq(label+' body Jp qvel',a['body_linear_velocity_world'],vel,2e-12)
    checks.eq(label+' body Jp qacc plus Jdot qvel',a['body_linear_acceleration_world'],acc,2e-12)
    M = float(mass.sum()); com = np.stack([mass@x/M for x in pos])
    v = np.stack([mass@x/M for x in vel]); ma = np.stack([mass@x for x in acc])
    raw = a['contact_force_raw6']; cap = raw.shape[1]
    require(raw.shape==(n,cap,6),'raw contact dimensions')
    frames = a['contact_frame'].reshape(n,cap,3,3); bodies=a['contact_body_ids']; geoms=a['contact_geom_ids']
    valid = np.arange(cap)[None,:]<a['contact_ncon'][:,None]
    checks.add(label+' ncon integer bounds',np.issubdtype(a['contact_ncon'].dtype,np.integer) and np.all((a['contact_ncon']>=0)&(a['contact_ncon']<=cap)))
    ext = valid & ((bodies[:,:,0]==0) ^ (bodies[:,:,1]==0))
    signs = np.where(bodies[:,:,0]==0,1.,-1.)*ext
    f = np.einsum('tcji,tcj->tci',frames,raw[:,:,:3])
    spin = np.einsum('tcji,tcj->tci',frames,raw[:,:,3:])
    contactF = (f*signs[:,:,None]).sum(1)
    moment = spin+np.cross(a['contact_pos_world']-com[:,None,:],f)
    contactT = (moment*signs[:,:,None]).sum(1)
    checks.eq(label+' valid mask',a['contact_valid_mask'],valid)
    checks.eq(label+' external body0 XOR; self excluded',a['contact_external_mask'],ext)
    checks.eq(label+' external body2/body1 signs',a['contact_external_sign'],signs)
    checks.eq(label+' geom/body identities',bodies[valid],np.asarray(meta['geom_bodyid'])[geoms[valid]])
    checks.add(label+' explicit contact sign', 'geom2' in meta['contact_sign'] and 'geom1' in meta['contact_sign'])
    checks.eq(label+' raw6 frame transpose force on geom2',a['contact_force_on_geom2_world'][valid],f[valid],2e-12)
    checks.eq(label+' raw6 frame transpose intrinsic torque on geom2',a['contact_intrinsic_torque_on_geom2_world'][valid],spin[valid],2e-12)
    checks.eq(label+' raw torque about COM',a['contact_torque_on_geom2_about_com'][valid],moment[valid],2e-12)
    if valid.any():
        gram = frames@np.swapaxes(frames,-1,-2)
        checks.eq(label+' orthonormal contact frames',gram[valid],np.broadcast_to(np.eye(3),gram[valid].shape),2e-12)
        checks.eq(label+' right handed contact frames',np.linalg.det(frames[valid]),np.ones(valid.sum()),2e-12)
    gravity = np.broadcast_to(np.asarray(meta['gravity_world']), (n,3))
    gravF=M*gravity
    gravT=np.cross(pos-com[:,None,:],mass[None,:,None]*gravity[:,None,:]).sum(1)
    checks.eq(label+' gravity COM torque zero',gravT,np.zeros((n,3)),1e-12)
    xf=a['xfrc_applied']; qf=a['qfrc_applied']
    checks.add(label+' zero external applied force',not xf.any() and not qf.any())
    netF=contactF+gravF+xf[:,1:,:3].sum(1)
    netT=contactT+gravT+(np.cross(pos[:,1:]-com[:,None,:],xf[:,1:,:3])+xf[:,1:,3:]).sum(1)
    p=M*v
    impulse=np.vstack((np.zeros(3),np.cumsum(.5*(netF[1:]+netF[:-1])*np.diff(t)[:,None],axis=0)))
    for k,value in dict(com_position_world=com,com_velocity_world=v,com_acceleration_world=ma/M,
                        com_mass_times_acceleration_world=ma,com_momentum_world=p,
                        contact_force_world=contactF,contact_torque_about_com=contactT,
                        gravity_force_world=gravF,gravity_torque_about_com=gravT,
                        external_force_world=netF,external_torque_about_com=netT,
                        external_impulse_world=impulse,com_momentum_change_world=p-p[0]).items():
        checks.eq(label+' aggregate '+k,a[k],value,2e-12)
    return dict(mass=M,com=com,velocity=v,acceleration=ma/M,ma=ma,momentum=p,contact_force=contactF,
                external_force=netF,external_torque=netT,force_residual=ma-netF,
                external_contact_counts=ext.sum(1),self_contact_counts=(valid&~ext).sum(1),
                external_mask=ext,contact_point=a['contact_pos_world'],contact_dist=a['contact_dist'])


def balance(a,r,stride,checks,label,mode):
    ix=np.arange(0,len(a['time']),stride); t=a['time'][ix]
    p=r['momentum'][ix]; f=r['external_force'][ix]; ma=r['ma'][ix]
    ids=np.flatnonzero((t<=.0017+1e-12)&(np.arange(len(t))>0)&(np.arange(len(t))<len(t)-1))
    inclusive=np.flatnonzero(t<=.0017+1e-12)
    h=float(np.mean(np.diff(t))); require(np.allclose(np.diff(t),h,rtol=0,atol=1e-12),'uniform clock')
    fd=np.gradient(p,t,axis=0,edge_order=2)
    fdv=np.gradient(r['com'][ix],t,axis=0,edge_order=2)
    instantaneous=measure(ma,f,np.arange(len(t)),1e-8,1e-6)
    impulse=np.vstack((np.zeros(3),np.cumsum(.5*(f[1:]+f[:-1])*np.diff(t)[:,None],axis=0)))
    im=measure(impulse,p-p[0],inclusive,1e-4,.01)
    im['global_denominator']=max(float(r['mass']*np.linalg.norm(r['velocity'][ix][inclusive],axis=1).max()),1e-12)
    im['relative_residual']=im['max_abs_residual']/im['global_denominator']
    im['passed']=bool(im['max_abs_residual']<=1e-4 and im['relative_residual']<=.01)
    im['denominator_definition']='M * max norm(v_COM) on inclusive 0..1.7ms; floor1e-12'
    momentumfd=measure(fd,f,ids,1.,.05)
    velocityfd=measure(fdv,r['velocity'][ix],ids,.1,.01)
    w=momentumfd['worst_index']; c=momentumfd['worst_component_index']
    momentumfd.update(worst_time_s=float(t[w]),worst_time_ms=float(t[w]*1000),
                      original_native_sample_index=int(ix[w]),stencil_indices=[w-1,w,w+1],
                      native_stencil_indices=ix[w-1:w+2].tolist(),stencil_times_s=t[w-1:w+2].tolist(),
                      stencil_coefficients_per_s=[-1/(2*h),0.,1/(2*h)],
                      stencil_momentum_all_xyz=p[w-1:w+2],dpforce_all_xyz=fd[w],Fext_all_xyz=f[w],
                      dpforce_component=float(fd[w,c]),Fext_component=float(f[w,c]),
                      method='np.gradient(momentum,time,axis=0,edge_order=2) evaluated BEFORE window slicing',
                      boundary_policy='indices1..end1.7ms inclusive; centered stencil retains next sample >1.7ms; none deleted')
    coefficients=np.array([-1/(2*h),0.,1/(2*h)])
    checks.eq(label+' '+mode+' worst centered stencil',coefficients@p[w-1:w+2],fd[w],2e-10)
    for name,value in (('instantaneous',instantaneous),('impulse',im),('momentum_FD',momentumfd),('position_FD',velocityfd)):
        checks.add(label+' '+mode+' '+name,value['passed'],**{k:v for k,v in value.items() if k!='passed'})
    return dict(mode=mode,stride=stride,effective_dt_s=h,time_s=t,native_indices=ix,
                instantaneous=instantaneous,impulse=im,momentum_FD=momentumfd,position_FD=velocityfd,
                fd_force=fd,fd_residual=fd-f,instantaneous_residual=ma-f)


def support(mask,t):
    value=float(np.sum(.5*(mask[:-1].astype(float)+mask[1:])*np.diff(t))*1000)
    padded=np.r_[False,mask,False].astype(int)
    starts=np.flatnonzero(np.diff(padded)==1); ends=np.flatnonzero(np.diff(padded)==-1)-1
    longest=max((float(t[b]-t[a])*1000 for a,b in zip(starts,ends)),default=0.)
    return value,longest


def physical(a,r,checks,label,weight,xml):
    q=a['qpos']; t=a['time']; quat=q[:,3:7]; norms=np.linalg.norm(quat,axis=1)
    height=q[:,2]/q[0,2]; angle=2*np.arccos(np.clip(np.abs(quat@quat[0])/(norms*norms[0]),0,1))
    load=a['foot_force'][:,:,2].sum(1)/weight; legs=np.count_nonzero(a['contacts']>0,axis=1)
    # Independently derive all hinge limits from explicit model XML; never trust saved maximum alone.
    tree=ET.parse(xml).getroot(); require(not tree.findall('.//include'),'explicit XML required')
    require(tree.find('compiler') is not None and tree.find('compiler').get('angle','degree')=='radian','XML angle radians')
    adr=0; excess=[]
    for node in tree.find('worldbody').iter():
        if node.tag not in ('joint','freejoint'): continue
        typ='free' if node.tag=='freejoint' else node.get('type','hinge')
        if typ=='free': adr+=7; continue
        require(typ=='hinge' and node.get('range') is not None,'hinge range required')
        lo,hi=map(float,node.get('range').split()); x=q[:,adr]; adr+=1
        excess.append(np.maximum(np.maximum(lo-x,x-hi),0))
    limit=np.max(excess,axis=0); checks.eq(label+' independent XML limits',a['limit_excess'],limit,1e-14)
    gates=np.column_stack((height>=.85,angle<=.35,legs>=3,load>=.8,a['nonfoot_depth']<=1e-12,limit<=1e-6))
    total,longest=support(gates.all(1),t)
    first={name:float(t[np.flatnonzero(~gates[:,j])[0]]*1000) if (~gates[:,j]).any() else None for j,name in enumerate(GATES)}
    if 'six_gates' in a: checks.eq(label+' saved six gates',a['six_gates'],gates)
    if 'all_six_gates' in a: checks.eq(label+' saved all six gates',a['all_six_gates'],gates.all(1))
    if 'load_ratio' in a: checks.eq(label+' saved load ratio',a['load_ratio'],load,2e-12)
    curves=dict(height_ratio=height,angle_rad=angle,load_ratio=load,contact_legs=legs,
                nonfoot_depth=a['nonfoot_depth'],limit_excess=limit,COM_z=r['com'][:,2],
                COM_vz=r['velocity'][:,2],COM_az=r['acceleration'][:,2],Fext_z=r['external_force'][:,2],
                contact_Fz=r['contact_force'][:,2],root_vz=a['qvel'][:,2],
                external_contact_count=r['external_contact_counts'],self_contact_count=r['self_contact_counts'])
    def extrema(x):
        lo=int(np.argmin(x)); hi=int(np.argmax(x))
        return dict(min=float(x[lo]),max=float(x[hi]),min_at_ms=float(t[lo]*1000),max_at_ms=float(t[hi]*1000))
    ext=r['external_mask']; distances=r['contact_dist']; points=r['contact_point']
    return dict(first_failures_ms=first,all_gates_support_ms=total,longest_all_gates_true_endpoint_span_ms=longest,
                gate_thresholds=dict(height=.85,angle=.35,legs=3,load=.8,nonfoot=1e-12,limit=1e-6),
                extrema={name:extrema(x) for name,x in curves.items()},
                external_contact_distance_min=float(distances[ext].min()) if ext.any() else None,
                external_contact_distance_max=float(distances[ext].max()) if ext.any() else None,
                external_contact_point_z_min=float(points[:,:,2][ext].min()) if ext.any() else None,
                external_contact_point_z_max=float(points[:,:,2][ext].max()) if ext.any() else None,
                gates=gates,load_ratio=load)


def same_arrays(checks,label,current,reference):
    checks.add(label+' keys',set(current)==set(reference))
    for key in set(current)&set(reference): checks.eq(label+' '+key,current[key],reference[key])


def verify(out):
    require(not any(x in sys.modules for x in ('mujoco','torch')),'simulator/neural module loaded')
    checks=Checks(); artifacts={}; pre=js(out/'PREREG.json'); hashes=pre['source_hashes']
    canonical_checks(checks,hashes,'source',artifacts)
    checks.add('runner source committed',str(ROOT/'run_com_timestep_probe.py') in hashes)
    # The runner freezes its own input set before execution; verifier identity is separately saved in this report.
    artifacts[str(Path(__file__).resolve())]=sha(__file__)
    artifacts[str(ROOT/'verify_com_contact_probe.py')]=sha(ROOT/'verify_com_contact_probe.py')
    checks.add('INPUT_HASHES exact agreement',js(out/'INPUT_HASHES.json')==hashes)
    dependencies=js(out/'MODEL_DEPENDENCIES.json'); canonical_checks(checks,dependencies,'model dependencies',artifacts)
    checks.add('dependency commitments included',all(hashes.get(p)==d for p,d in dependencies.items()))
    checks.add('runner complete',js(out/'COMPLETE.json')['status']=='COMPLETE' and not (out/'FAILED.json').exists())
    checks.add('fixed timing',pre['initialization_ticks']==80 and pre['physical_duration_ms']==3 and pre['neural_dt_ms']==.5)
    ledger=js(out/'CALL_LEDGER.json')
    expected_calls=dict(model_load=2,forward=182,step=180,neural_ticks=172,active=None,completed=list(ARMS))
    expected_per_arm=dict(dt50us=dict(model_load=1,forward=61,step=60,neural_ticks=86),dt25us=dict(model_load=1,forward=121,step=120,neural_ticks=86))
    checks.add('exact fresh two-arm call ledger',ledger==dict(expected_calls,per_arm=expected_per_arm),observed=ledger)
    oldpre=js(OLD/'PREREG.json'); oldresults=js(OLD/'RESULTS.json'); oldresult=next(r for r in oldresults if r['label']=='baseline')
    # Bind old reference to its immutable source/result manifests without requiring now-modified unrelated historical source files.
    canonical_checks(checks,oldresult['hashes'],'S112 baseline artifacts',artifacts)
    for name in ('TRAJECTORY.npz','COM_CONTACT_METADATA.json'):
        path=str(OLD/'baseline'/name); checks.add('old source committed '+name,hashes.get(path)==sha(path))
    w=npz(pre['same_weights'])['weights']; direction=npz(out/'DIRECTION.npz')
    checks.eq('same source W as S112',w,npz(oldpre['same_weights'])['weights'])
    checks.eq('DIRECTION source W exact',direction['weights'],w)
    manifests=[Path(p) for p in hashes if Path(p).name=='MANIFEST.json']; require(len(manifests)==1,'one mapping manifest')
    manifest=js(manifests[0]); rows=sorted(manifest['rows'],key=lambda x:x['actuator_index'])
    pools=sorted({tuple(sorted(row['local_indices'])) for row in rows if row['local_indices']}); lookup={p:i for i,p in enumerate(pools)}
    mapping=np.array([lookup.get(tuple(sorted(row['local_indices'])),-1) for row in rows])
    checks.add('mapping dimensions',len(pools)==66 and len(mapping)==90 and mapping[4]==15)
    xmls=[Path(p) for p in hashes if Path(p).suffix=='.xml']; require(len(xmls)==1,'one physical XML')
    results=js(out/'RESULTS.json'); checks.add('arm order',[r['label'] for r in results]==list(ARMS))
    allresults=[('dt100us',OLD/'baseline',oldresult,1e-4,1)]+[(r['label'],out/r['label'],r,5e-5 if r['label']=='dt50us' else 2.5e-5,2 if r['label']=='dt50us' else 4) for r in results]
    plotted=[]; baseline=None
    for label,d,result,dt,stride in allresults:
        if label!='dt100us': canonical_checks(checks,result['hashes'],label+' artifacts',artifacts)
        committed={str(Path(p).resolve()) for p in result['hashes']}
        for name in ('TRAJECTORY.npz','INITIALIZATION.npz','NEURAL_COLD.npz','CONNECTIVITY.npz','NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','NEURAL_FINAL.npz','INITIAL_OBSERVATION.json','OBSERVATION_00.json','SCHEDULE.json','MOMENT_METADATA.json','COM_CONTACT_METADATA.json','COM_CLOSURE.json','MODEL_PROVENANCE.json'):
            checks.add(label+' committed '+name,str((d/name).resolve()) in committed)
        a=npz(d/'TRAJECTORY.npz'); init=npz(d/'INITIALIZATION.npz'); meta=js(d/'COM_CONTACT_METADATA.json')
        for key,value in a.items(): checks.add(label+' finite '+key,np.isfinite(value).all())
        n=int(round(.003/dt))+1; hold=int(round(.0005/dt))
        checks.eq(label+' physical clock',a['time'],np.arange(n)*dt,1e-12)
        for key,shape in dict(qpos=(n,49),qvel=(n,48),qacc=(n,48),spikes=(6,17099),features=(6,66,5),applied=(6,90)).items(): checks.add(label+' shape '+key,a[key].shape==shape)
        checks.eq(label+' exact weights',a['weights'],w); checks.eq(label+' initialization W',init['weights'],w)
        if label!='dt100us': checks.eq(label+' direction W',direction[label],w)
        checks.eq(label+' actuator pool',a['actuator_pool'],mapping)
        checks.eq(label+' initial body qpos',a['qpos'][0],init['body_qpos'])
        for key in ('qvel','act','ctrl'): checks.add(label+' zero initial '+key,not a[key][0].any() and not init['body_'+key].any())
        checks.add(label+' no warnings',not a['warnings'].any())
        checks.eq(label+' hold '+str(hold)+' steps',a['ctrl'][1:],np.repeat(a['applied'],hold,axis=0))
        checks.eq(label+' fixed sensory current',a['current'],np.repeat(init['current'][None],6,axis=0))
        fs,us,history=reconstruct_fir(np.concatenate((init['spikes'],a['spikes'])),pools,mapping,w)
        for name,x,y in (('warm features',init['features'],fs[:80]),('warm decoded',init['decoded'],us[:80]),('runtime features',a['features'],fs[80:]),('runtime commands',a['applied'],us[80:]),('release features',init['release_features'],fs[79]),('release decoded',init['release_decoded'],us[79]),('final FIR',npz(d/'NEURAL_FINAL.npz')['FIR_history'],history)):
            checks.eq(label+' FIR '+name,x,y,1e-14)
        schedule=js(d/'SCHEDULE.json'); checks.add(label+' six schedule rows',len(schedule)==6)
        for k,row in enumerate(schedule):
            checks.add(label+' schedule IDs '+str(k),row['tick']==k and row['consume_id']==80+k)
            for key,value in dict(source_start_s=(80+k)*.0005,source_end_s=(81+k)*.0005,physical_start_s=k*.0005,physical_end_s=(k+1)*.0005,source_offset_s=.04).items(): checks.add(label+' schedule '+str(k)+' '+key,abs(row[key]-value)<=1e-12)
        for name,steps,ms in (('NEURAL_COLD_META.json',0,0),('NEURAL_RELEASE_META.json',80,40),('NEURAL_FINAL_META.json',86,43)):
            scalars=js(d/name)['neural_scalars']; checks.add(label+' neural clock '+name,scalars['steps']==steps and abs(scalars['t_ms']-ms)<=1e-10)
        prov=js(d/'MODEL_PROVENANCE.json'); changed=[k for k in prov['before'] if prov['before'][k]!=prov['after'][k]]
        checks.add(label+' numeric integrator/timestep ONLY',set(changed)<= {'opt.integrator','opt.timestep'} and set(prov['changed_numeric_fields'])==set(changed) and prov['after']['opt.integrator']==1 and abs(prov['after']['opt.timestep']-dt)<=1e-15,observed_fields=changed)
        if baseline is None: baseline=(d,a,init,prov)
        bd,ba,bi,bp=baseline
        if label!='dt100us':
            same_arrays(checks,label+' matched full initialization',init,bi)
            for name in ('NEURAL_COLD.npz','CONNECTIVITY.npz','NEURAL_FINAL.npz'): same_arrays(checks,label+' matched '+name,npz(d/name),npz(bd/name))
            for name in ('NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','INITIAL_OBSERVATION.json','OBSERVATION_00.json'):
                checks.add(label+' matched '+name,js(d/name)==js(bd/name))
            for key in ('current','spikes','features','applied'): checks.eq(label+' matched all6 runtime '+key,a[key],ba[key])
            checks.add(label+' matched premodel snapshot',prov['before']==bp['before'])
            checks.add(label+' identical model except timestep',all(prov['after'][k]==bp['after'][k] for k in bp['after'] if k!='opt.timestep'))
        rec=reconstruct(a,meta,checks,label)
        native=balance(a,rec,1,checks,label,'native'); coarse=balance(a,rec,stride,checks,label,'downsample100us')
        savedclosure=js(d/'COM_CLOSURE.json')
        for mode,computed in (('native',native),('downsample100us',coarse)):
            recorded=savedclosure if label=='dt100us' else savedclosure[mode]
            for key,savedkey in (('instantaneous','instantaneous_force'),('impulse','impulse'),('momentum_FD','pointwise_force'),('position_FD','com_velocity_fd')):
                for scalar in ('max_abs_residual','relative_residual'):
                    checks.eq(label+' '+mode+' runner closure '+key+' '+scalar,computed[key][scalar],recorded[savedkey][scalar],2e-12)
                checks.add(label+' '+mode+' runner closure '+key+' status',computed[key]['passed']==recorded[savedkey]['passed'])
        weight=float(result['weight_native']); metrics=physical(a,rec,checks,label,weight,xmls[0])
        checks.eq(label+' native weight',weight,rec['mass']*abs(float(meta['gravity_world'][2])),1e-12)
        jload=float(np.trapezoid((np.maximum(.8-metrics['load_ratio'],0)/.8)**2,x=a['time'])/.003)
        jlimit=float(np.trapezoid((np.maximum(a['limit_excess']-1e-6,0)/.001)**2,x=a['time'])/.003)
        metrics.update(Jload=jload,Jlimit=jlimit)
        for name,value in (('Jload',jload),('Jlimit',jlimit)):
            if name in result: checks.eq(label+' '+name,value,result[name],2e-12)
        checks.eq(label+' signed actuator projection',np.einsum('tu,tud->td',a['actuator_force'],a['actuator_moment']),a['qfrc_actuator'],2e-12)
        plotted.append(dict(label=label,path=str(d),dt_s=dt,stride100=stride,arrays=a,derived=rec,native=native,downsample100us=coarse,physical=metrics))
    shared=[]
    for left,right in itertools.combinations(plotted,2):
        a=left['arrays']; b=right['arrays']; li=np.arange(0,len(a['time']),left['stride100']); ri=np.arange(0,len(b['time']),right['stride100'])
        checks.eq(left['label']+' '+right['label']+' shared times',a['time'][li],b['time'][ri],1e-12)
        curves={}
        for name,x,y in [('COM',left['derived']['com'],right['derived']['com']),('COM_velocity',left['derived']['velocity'],right['derived']['velocity']),('Fext',left['derived']['external_force'],right['derived']['external_force']),('contact_force',left['derived']['contact_force'],right['derived']['contact_force']),('force_balance_residual',left['derived']['force_residual'],right['derived']['force_residual'])]+[(key,a[key],b[key]) for key in ('actuator_force','act','ctrl','foot_force')]:
            x=x[li]; y=y[ri]; delta=y-x
            curves[name]=dict(reference=x,comparison=y,delta=delta,max_abs_delta=float(np.abs(delta).max()),rms_delta=float(np.sqrt(np.mean(delta*delta))))
        shared.append(dict(reference=left['label'],comparison=right['label'],time_ms=a['time'][li]*1000,curves=curves))
    native_errors=[p['native']['momentum_FD']['max_abs_residual'] for p in plotted]
    coarse_errors=[p['downsample100us']['momentum_FD']['max_abs_residual'] for p in plotted]
    consistent_native=all(b<a for a,b in zip(native_errors,native_errors[1:]))
    # A native decline alone can simply reflect the narrower derivative stencil.
    coarse_spread=(max(coarse_errors)-min(coarse_errors))/max(max(coarse_errors),1e-12)
    trajectories_close=all(item['curves']['COM']['max_abs_delta']<1e-3 for item in shared)
    decision=dict(native_FD_max_residuals=native_errors,downsample100us_FD_max_residuals=coarse_errors,
                  native_strictly_decreasing=consistent_native,downsample_relative_spread=coarse_spread,
                  interpretation=('Consistent native-residual decrease with nearly unchanged common-grid residual supports FD sampling/stencil truncation; common-grid force trajectories must also be inspected.' if consistent_native and coarse_spread<.1 else 'No unambiguous consistent convergence pattern established; do not infer integration convergence from a pass/fail flip.'),
                  convergence_evidence_only=bool(consistent_native and coarse_spread<.1),
                  standing_or_walking_claim=False,automatic_training=False,threshold_changes=False,selection=False,
                  caution='Three short diagnostic trajectories do not establish asymptotic order, biological force units, or persistent support.')
    return checks,plotted,dict(source_hashes=hashes,observed_hashes=artifacts,ledger=ledger,shared_time_comparisons=shared,decision=decision)


def plot(out,plotted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(16,14))
    for item in plotted:
        label=item['label']; a=item['arrays']; t=a['time']*1000
        for mode,ax in (('native',axes[0,0]),('downsample100us',axes[0,1])):
            b=item[mode]; ids=b['momentum_FD']['sample_indices']; times=b['time_s']*1000
            for j,component in enumerate('xyz'): ax.plot(times[ids],b['fd_residual'][ids,j],label=label+' '+component)
        axes[1,0].plot(t,item['physical']['load_ratio'],label=label)
        axes[1,1].plot(t,np.max(np.abs(item['derived']['force_residual']),axis=1),label=label)
        axes[2,0].plot(t,item['derived']['velocity'][:,2],label=label+' COM vz')
        axes[2,1].plot(t,item['derived']['external_force'][:,2],label=label+' Fext z')
    inset=axes[0,0].inset_axes([.57,.57,.4,.38])
    for mode in ('native','downsample100us'):
        inset.plot([p['dt_s']*1e6 for p in plotted],[p[mode]['momentum_FD']['max_abs_residual'] for p in plotted],'o-',label=mode)
    inset.set_xlabel('dt (us)',fontsize=7); inset.set_ylabel('max |FD-Fext|',fontsize=7); inset.tick_params(labelsize=6); inset.legend(fontsize=5)
    titles=('Native momentum FD residual (all components; window1..1.7ms)','Downsampled100us momentum FD residual (identical window)','Load trajectory: allfoot Fz / native weight','Primary raw COM force balance max component residual','COM vertical velocity trajectory','External net vertical force trajectory')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10); ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=2); ax.set_xlabel('Physical time (ms)'); ax.axvline(1.7,color='grey',ls=':')
    axes[1,0].axhline(.8,color='black',ls=':'); axes[1,1].set_yscale('symlog',linthresh=1e-15)
    fig.suptitle('S113 baseline-only timestep diagnostic, unchanged W\nOffline raw-array reconstruction; no sample deletion, threshold change, training, or behavior claim; units NOT_ESTABLISHED')
    fig.tight_layout(); path=out/'TIMESTEP_COMPARISON.png'; fig.savefig(path,dpi=150); plt.close(fig); return path


def csvs(out,checks,plotted):
    scalar=out/'TIMESTEP_SCALAR_CHECKS.csv'
    fields=['arm','mode','dt_us','check','passed','max_abs_residual','relative_residual','global_denominator','absolute_tolerance','relative_tolerance','worst_time_ms','worst_component']
    with scalar.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for p in plotted:
            for mode in ('native','downsample100us'):
                for key in ('instantaneous','impulse','momentum_FD','position_FD'):
                    b=p[mode][key]; row={k:b[k] for k in fields if k in b}
                    row.update(arm=p['label'],mode=mode,dt_us=p['dt_s']*1e6,check=key)
                    w.writerow(row)
    allchecks=out/'TIMESTEP_ALL_CHECKS.csv'
    with allchecks.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['name','passed','observed_json'])
        for row in checks.rows: w.writerow([row['name'],row['passed'],json.dumps({k:v for k,v in row.items() if k not in ('name','passed')},default=serial,allow_nan=False)])
    return [str(scalar),str(allchecks)]


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=DEFAULT)
    out=parser.parse_args().output.resolve()
    report=dict(passed=False,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),new_physics_calls=0,new_neural_ticks=0,physical_units='NOT_ESTABLISHED',behavior_claim='NONE')
    try:
        checks,plotted,details=verify(out)
        report.update(details,passed=checks.passed,checks=checks.rows,
                      per_arm={p['label']:{k:v for k,v in p.items() if k not in ('arrays','derived')} for p in plotted},
                      comparison_png=str(plot(out,plotted)),scalar_csvs=csvs(out,checks,plotted))
        require(not any(x in sys.modules for x in ('mujoco','torch')),'physics/neural imported')
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    write_json(out/'TIMESTEP_INDEPENDENT_CHECK.json',report)
    print(json.dumps(dict(passed=report['passed'],error=report.get('error'),report=str(out/'TIMESTEP_INDEPENDENT_CHECK.json')),indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__': sys.exit(main())
