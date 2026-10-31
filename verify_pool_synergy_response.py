"""S111 offline four-arm synergy verification; no physics/neural execution.

Only pool identity validation is reused from the translator. FIR, CSR projection,
limits, pairwise differences and fixed-reference torque algebra are recomputed.
"""
from pathlib import Path
import argparse
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
from verify_neural_initialized_startup import (
    ROOT, MANIFEST, INITIAL, BINS, DT, require, equal, sha, load_json,
    load_npz, fir_recompute, physical_metrics,
)
from verify_local_pool_response import check_dynamics, sparse_dense

RUNNER = ROOT / 'run_pool_synergy_response.py'
ARMS = ('baseline', 'pool3_up', 'pool15_up', 'both_up')
PAIRS = (('baseline', 'pool15_up'), ('pool3_up', 'both_up'))
PRE, TICKS, SAMPLES = 80, 20, 101
DT_BODY = .0001


def expected_weights(w):
    require(w.shape == (66, 5) and np.isfinite(w).all(), 'source W shape/finite')
    result = {}
    for label in ARMS:
        v = w.copy()
        if label in ('pool3_up', 'both_up'): v[3, [1, 2, 3]] *= 1.2
        if label in ('pool15_up', 'both_up'): v[15, 3] *= 1.2
        require(np.all((v >= 0) & (v <= 4)), label + ' W bounds')
        require(np.array_equal(v == 0, w == 0), label + ' zero mask')
        result[label] = v
    return result


def joint_order(path):
    """Parse this explicit, radians XML without importing the physics backend."""
    root = ET.parse(path).getroot()
    require(not root.findall('.//include'), 'included XML needs explicit independent resolution')
    compiler = root.find('compiler')
    require(compiler is not None and compiler.get('angle', 'degree') == 'radian', 'XML radians required')
    nodes = [n for n in root.find('worldbody').iter() if n.tag in ('joint', 'freejoint')]
    qadr = vadr = 0; joints = []; names = []; qa = []; va = []
    for n in nodes:
        typ = 'free' if n.tag == 'freejoint' else n.get('type', 'hinge')
        names.append(n.get('name')); qa.append(qadr); va.append(vadr)
        if typ == 'free': qadr += 7; vadr += 6; continue
        require(typ == 'hinge' and n.get('range') is not None, 'explicit hinge ranges required')
        bounds = list(map(float, n.get('range').split()))
        require(len(bounds) == 2 and bounds[0] <= bounds[1], 'joint range')
        require(n.get('limited', 'auto') != 'false', 'unexpected unlimited hinge')
        joints.append(dict(name=n.get('name'), qpos_index=qadr, dof_index=vadr,
                           lower=bounds[0], upper=bounds[1]))
        qadr += 1; vadr += 1
    require((qadr, vadr, len(joints)) == (49, 48, 42), 'XML body dimensions')
    for qi, word, vi in ((7, 'yaw', 6), (8, 'pitch', 7), (9, 'roll', 8)):
        j = next(j for j in joints if j['qpos_index'] == qi)
        require(word in j['name'].lower() and j['dof_index'] == vi, 'yaw/pitch/roll address mismatch')
    return joints, names, qa, va


def extrema(x, time, mask=None):
    x = np.asarray(x); ids = np.arange(len(x)) if mask is None else np.flatnonzero(mask)
    if not len(ids): return dict(samples=0, min=None, max=None, min_at_ms=None, max_at_ms=None)
    lo = int(ids[np.argmin(x[ids])]); hi = int(ids[np.argmax(x[ids])])
    return dict(samples=len(ids), min=float(x[lo]), max=float(x[hi]),
                min_at_ms=float(time[lo]*1000), max_at_ms=float(time[hi]*1000))


def limits(a, joints):
    q = a['qpos']; t = a['time']; excess = []; rows = []
    for j in joints:
        x = q[:, j['qpos_index']]
        ex = np.maximum(np.maximum(j['lower']-x, x-j['upper']), 0)
        excess.append(ex); fail = np.flatnonzero(ex > 1e-6)
        rows.append(dict(**j, angle=extrema(x,t), excess=extrema(ex,t),
                         first_fail_ms=float(t[fail[0]]*1000) if len(fail) else None))
    equal(a['limit_excess'], np.max(excess,axis=0), 'independent all42 joint-limit maximum', 1e-14)
    return rows


def pairwise(plotted):
    by = {label:(a,m) for label,a,m in plotted}; pairs = []
    for left,right in PAIRS:
        a,ma = by[left]; b,mb = by[right]; t = a['time']
        # Saved sample zero precedes the command assignment; applied[0] starts at 0ms.
        require(a['applied'][0,4] != b['applied'][0,4], right+' actuator4 command must differ at interval start 0ms')
        require(a['ctrl'][1,4] != b['ctrl'][1,4], right+' actuator4 held ctrl first endpoint')
        require(a['actuator_force'][1,4] != b['actuator_force'][1,4], right+' actuator4 force must differ at .1ms')
        firstfails = [v for m in (ma,mb) for v in m['first_failures_ms'].values() if v is not None]
        cutoff = min(firstfails) if firstfails else 10.0
        windows = {'full_0_10ms':np.ones(SAMPLES,bool), 'early_0_5ms':t<=.005+1e-12,
                   'strict_pre_first_either_gate_failure':t*1000 < cutoff-1e-10}
        curves = {}; algebra = {}
        def add(name, x, y):
            delta = y-x
            curves[name] = dict(reference= x.tolist(), intervention=y.tolist(), delta=delta.tolist(),
                extrema={w:dict(reference=extrema(x,t,mask),intervention=extrema(y,t,mask),delta=extrema(delta,t,mask)) for w,mask in windows.items()},
                endpoints={str(ms)+'ms':dict(reference=float(x[i]),intervention=float(y[i]),delta=float(delta[i])) for ms,i in ((.1,1),(1.6,16))})
        add('actuator4_scalar_force', a['actuator_force'][:,4], b['actuator_force'][:,4])
        add('actuator4_ctrl_saved', a['ctrl'][:,4], b['ctrl'][:,4])
        for qi,word in ((7,'yaw'),(8,'pitch')):
            add(f'qpos{qi}_{word}',a['qpos'][:,qi],b['qpos'][:,qi])
            add(f'qpos{qi}_{word}_displacement_from_initial',a['qpos'][:,qi]-a['qpos'][0,qi],b['qpos'][:,qi]-b['qpos'][0,qi])
        for dof,word in ((6,'yaw'),(7,'pitch'),(8,'roll')):
            f0=a['actuator_force']; f1=b['actuator_force']
            m0=a['actuator_moment'][:,:,dof]; m1=b['actuator_moment'][:,:,dof]
            tau0=f0*m0; tau1=f1*m1
            force_term=(f1-f0)*m0; moment_term=f1*(m1-m0)
            actual=tau1-tau0; reconstructed=force_term+moment_term
            equal(reconstructed,actual,f'{left}->{right} fixed-reference DOF{dof}',2e-12)
            equal(actual.sum(1),b['qfrc_actuator'][:,dof]-a['qfrc_actuator'][:,dof],f'{left}->{right} actual aggregate delta DOF{dof}',2e-12)
            add(f'actuator4_moment_DOF{dof}_{word}', m0[:,4],m1[:,4])
            add(f'actuator4_torque_DOF{dof}_{word}',tau0[:,4],tau1[:,4])
            add(f'aggregate_torque_DOF{dof}_{word}',a['qfrc_actuator'][:,dof],b['qfrc_actuator'][:,dof])
            algebra[str(dof)] = dict(label='FIXED_REFERENCE_ALGEBRA_NOT_EXTRA_INTERVENTION',
                formula='(F1-F0)*M0 + F1*(M1-M0) = F1*M1-F0*M0',
                all_actuator_force_term=force_term.tolist(),all_actuator_moment_term=moment_term.tolist(),
                actual_all_actuator_delta=actual.tolist(),max_abs_error=float(np.max(np.abs(reconstructed-actual))),
                actuator4_force_term=force_term[:,4].tolist(),actuator4_moment_term=moment_term[:,4].tolist(),
                aggregate_force_term=force_term.sum(1).tolist(),aggregate_moment_term=moment_term.sum(1).tolist())
            if dof==8:
                for u in range(90): add(f'actuator{u}_torque_DOF8',tau0[:,u],tau1[:,u])
        reverse = np.array(curves['actuator4_torque_DOF8_roll']['delta'])
        agg = np.array(curves['aggregate_torque_DOF8_roll']['delta'])
        trade = {k:dict(reference=ma[k],intervention=mb[k],delta=mb[k]-ma[k]) for k in (
            'Jload','Jlimit','max_angle_rad','max_limit_excess_rad','min_height_ratio','min_load_ratio','all_gates_support_ms',
            'max_downward_velocity_native','max_nonfoot_depth_native','min_contact_legs')}
        pairs.append(dict(reference=left,intervention=right,time_ms=(t*1000).tolist(),
            command_start_0ms=dict(reference=float(a['applied'][0,4]),intervention=float(b['applied'][0,4]),
                                  delta=float(b['applied'][0,4]-a['applied'][0,4])),
            first_common_gate_failure_ms=cutoff,curves=curves,torque_decomposition=algebra,tradeoff=trade,
            decision=dict(actuator4_negative_DOF8_increment_exists=bool(np.any(reverse[1:] < -1e-12)),
                actuator4_negative_DOF8_increment_pre_failure_exists=bool(np.any(reverse[windows['strict_pre_first_either_gate_failure']] < -1e-12)),
                aggregate_negative_DOF8_increment_exists=bool(np.any(agg[1:] < -1e-12)),
                interpretation='Observed signed trajectory increments, not isolated fixed-state causality or pure roll compensation.',
                checkpoint_selection=False,deployment=False)))
    return dict(passed=True,pairs=pairs,new_physics_calls=0,new_neural_ticks=0,
        moment_sign='d(actuator_length)/d(q DOF); actual signed force times moment, no anatomical inference',
        ctrl_0ms_convention='applied[0] is command at physical interval START 0ms; saved ctrl[0] is zero before assignment; ctrl[1] is first .1ms endpoint',
        negative_increment_threshold_native=1e-12,threshold_role='floating point screening, NOT biological calibration',
        physical_units='NOT_ESTABLISHED',behavior_claim='NONE',selection=False)


def verify(out):
    out=Path(out).resolve(); require('mujoco' not in sys.modules,'physics imported')
    require(load_json(out/'COMPLETE.json')['status']=='COMPLETE','incomplete')
    require(not (out/'FAILED.json').exists(),'FAILED.json present')
    pre=load_json(out/'PREREG.json'); hashes=pre['source_hashes']
    for p in (RUNNER,MANIFEST,INITIAL): require(str(p) in hashes,'required source hash '+str(p))
    for p,digest in hashes.items(): require(sha(p)==digest,'source hash '+p)
    require(load_json(out/'INPUT_HASHES.json')==hashes,'INPUT_HASHES agreement')
    for p,digest in load_json(out/'MODEL_DEPENDENCIES.json').items():
        require(hashes.get(p)==digest and sha(p)==digest,'model dependency '+p)
    require(pre['initialization_ticks']==80 and pre['physical_duration_ms']==10 and pre['dt_ms']==.1 and pre['neural_dt_ms']==.5,'prereg timing')
    ledger=load_json(out/'CALL_LEDGER.json')
    require(ledger==dict(model_load=4,forward=404,step=400,neural_ticks=400,active=None,completed=list(ARMS)),'exact ledger')
    sourcew=Path(pre['same_weights']); require(str(sourcew) in hashes,'source weights hash')
    w0=load_npz(sourcew)['weights']; expected=expected_weights(w0); direction=load_npz(out/'DIRECTION.npz')
    require('weights' in direction,'DIRECTION source weights missing'); equal(direction['weights'],w0,'DIRECTION source W')
    for label in ARMS:
        require(label in direction,'missing DIRECTION '+label); equal(direction[label],expected[label],label+' exact all330 weights')
    xmls=[Path(p) for p in hashes if Path(p).suffix=='.xml']
    require(len(xmls)==1,'one committed physical XML required'); joints,names,qa,va=joint_order(xmls[0])
    from engine.embodied.motor_translation import FiniteMotorTranslator
    mapping=FiniteMotorTranslator(load_json(MANIFEST),w0)
    require(tuple(mapping.BINS)==BINS and mapping.DT==DT and mapping.actuator_pool[4]==15,'mapping/kernel/actuator4 pool')
    results=load_json(out/'RESULTS.json'); require([r['label'] for r in results]==list(ARMS),'arm order')
    checks=[]; plotted=[]; reference=None; reference_provenance=None; observed_hashes={}
    for r in results:
        label=r['label']; d=out/label
        required=('TRAJECTORY.npz','INITIALIZATION.npz','NEURAL_COLD.npz','CONNECTIVITY.npz',
            'NEURAL_COLD_META.json','NEURAL_RELEASE_META.json','NEURAL_FINAL_META.json','NEURAL_FINAL.npz',
            'INITIAL_OBSERVATION.json','OBSERVATION_00.json','OBSERVATION_10.json','SCHEDULE.json',
            'MOMENT_METADATA.json','MODEL_PROVENANCE.json')
        committed={Path(p).resolve():digest for p,digest in r['hashes'].items()}
        require(len(committed)==len(r['hashes']),label+' duplicate resolved commitments')
        for name in required:
            p=d/name; require(p.resolve() in committed,label+' missing artifact commitment '+name)
        for p,digest in r['hashes'].items():
            require(sha(p)==digest,label+' artifact '+p); observed_hashes[p]=digest
        require(r['trajectory_sha256']==sha(d/'TRAJECTORY.npz') and r['initialization_sha256']==sha(d/'INITIALIZATION.npz'),'primary hashes')
        require(r['direction_sha256']==sha(out/'DIRECTION.npz'),'direction hash')
        a=load_npz(d/'TRAJECTORY.npz'); init=load_npz(d/'INITIALIZATION.npz')
        cold=load_npz(d/'NEURAL_COLD.npz'); csr=load_npz(d/'CONNECTIVITY.npz')
        coldmeta=load_json(d/'NEURAL_COLD_META.json'); release=load_json(d/'NEURAL_RELEASE_META.json')
        obs=load_json(d/'INITIAL_OBSERVATION.json'); meta=load_json(d/'MOMENT_METADATA.json')
        finalmeta=load_json(d/'NEURAL_FINAL_META.json')
        for stage,state,steps,ms in (('cold',coldmeta,0,0.),('release',release,80,40.),('final',finalmeta,100,50.)):
            scalars=state['neural_scalars']
            require(scalars['steps']==steps and abs(scalars['t_ms']-ms)<=1e-10,label+' '+stage+' neural source clock')
            require(scalars['n']==17099 and scalars['dt']==.5,label+' '+stage+' neural dimensions')
        provenance=load_json(d/'MODEL_PROVENANCE.json')
        require(provenance['xml_sha256']==hashes[str(xmls[0])],label+' model XML provenance')
        require(provenance['changed_numeric_fields']==['opt.integrator'],label+' model numeric changes')
        before=provenance['before']; after=provenance['after']
        require(set(before)==set(after) and [k for k in before if before[k]!=after[k]]==['opt.integrator'],label+' only integrator changed')
        require(after['opt.integrator']==1,label+' RK4 integrator')
        if reference_provenance is None: reference_provenance=provenance
        require(provenance==reference_provenance,label+' all-arm full numeric model provenance equality')
        require(meta['joint_names']==names and meta['jnt_qposadr']==qa and meta['jnt_dofadr']==va,label+' metadata addresses vs XML')
        for j in joints:
            ji=names.index(j['name'])
            require(bool(meta['jnt_limited'][ji]),label+' metadata limited joint '+j['name'])
            equal(meta['jnt_range'][ji],[j['lower'],j['upper']],label+' metadata/XML range '+j['name'],1e-14)
        require(sum(bool(x) for x in meta['jnt_limited'])==42,label+' metadata42 limited joints')
        require(meta['header'] in hashes and meta['header_sha256']==hashes[meta['header']],label+' moment header commitment')
        equal(init['weights'],expected[label],label+' init W'); equal(a['weights'],expected[label],label+' trajectory W')
        require(bool(cold),'empty cold state'); require(set(csr)=={'We_data','We_indices','We_indptr','We_shape','Wi_data','Wi_indices','Wi_indptr','Wi_shape'},'CSR fields')
        if reference is None: reference=(init,cold,csr,coldmeta,release,obs)
        bi,bc,bcsr,bcm,br,bo=reference
        require(set(cold)==set(bc) and set(csr)==set(bcsr),label+' network field sets')
        for k in cold:
            equal(cold[k],bc[k],label+' cold '+k)
            require(k in init,label+' missing release ndarray '+k); equal(init[k],bi[k],label+' release ndarray '+k)
        for k in csr: equal(csr[k],bcsr[k],label+' connectivity '+k)
        require(coldmeta==bcm and release==br and obs==bo,label+' matched full scalar metadata/initial observation')
        for k in ('current','spikes','features','history','release_features','body_qpos','body_qvel','body_act','body_ctrl'):
            equal(init[k],bi[k],label+' matched '+k)
        equal(init['body_qpos'],load_npz(INITIAL)['new_qpos'],label+' source initial qpos')
        for k,shape in dict(time=(101,),qpos=(101,49),qvel=(101,48),act=(101,90),ctrl=(101,90),contacts=(101,6),foot_force=(101,6,3),
            spikes=(20,17099),features=(20,66,5),applied=(20,90),current=(20,17099),actuator_velocity=(101,90)).items():
            require(a[k].shape==shape,label+' '+k+' shape')
        equal(a['time'],np.arange(SAMPLES)*DT_BODY,label+' clock',1e-12)
        equal(a['qpos'][0],init['body_qpos'],label+' initial pose')
        for k in ('qvel','act','ctrl'):
            require(not init['body_'+k].any() and not a[k][0].any(),label+' zero initial '+k)
        for k in ('qfrc_applied','xfrc_applied','warnings'): require(not a[k].any(),label+' nonzero '+k)
        for k in ('act','ctrl','applied'): require(np.all((a[k]>=0)&(a[k]<=1)),label+' '+k+' bounds')
        require(init['spikes'].shape==(80,17099),'warm spikes shape')
        wf,wu,history=fir_recompute(init['spikes'],mapping.pools,mapping.actuator_pool,expected[label])
        equal(init['features'],wf,label+' warm FIR',1e-14); equal(init['decoded'],wu,label+' warm decode',1e-14)
        equal(init['history'],history,label+' warm FIFO',1e-14)
        rf=np.stack([history[lo:hi].mean(0) for lo,hi in BINS],axis=1)
        equal(init['release_features'],rf,label+' release features',1e-14)
        ru=np.zeros(90); mapped=mapping.actuator_pool>=0
        ru[mapped]=np.clip((rf*expected[label]).sum(1),0,1)[mapping.actuator_pool[mapped]]
        equal(init['release_decoded'],ru,label+' release command',1e-14)
        wf,wu,history=fir_recompute(np.concatenate((init['spikes'],a['spikes'])),mapping.pools,mapping.actuator_pool,expected[label])
        equal(a['features'],wf[80:],label+' continuous FIR',1e-14); equal(a['applied'],wu[80:],label+' continuous commands',1e-14)
        equal(load_npz(d/'NEURAL_FINAL.npz')['FIR_history'],history,label+' final FIR history',1e-14)
        equal(a['ctrl'][1:],np.repeat(a['applied'],5,axis=0),label+' five step hold')
        equal(a['current'][:10],np.repeat(init['current'][None],10,axis=0),label+' first sensory hold')
        equal(a['current'][10:],np.repeat(a['current'][10:11],10,axis=0),label+' second sensory hold')
        schedule=load_json(d/'SCHEDULE.json'); require(len(schedule)==20,'schedule length')
        for k,row in enumerate(schedule):
            require(row['tick']==k and row['consume_id']==80+k,'schedule IDs')
            for key,value in dict(source_start_s=(80+k)*DT,source_end_s=(81+k)*DT,physical_start_s=k*DT,physical_end_s=(k+1)*DT,source_offset_s=.04).items():
                require(abs(row[key]-value)<=1e-12,label+' schedule '+key)
        for key,value in dict(source_offset_s=.04,neural_final_s=.05,FIR_final_s=.05,physical_final_s=.01).items():
            require(abs(r[key]-value)<=1e-12,label+' result clock '+key)
        dyn=check_dynamics(a,label)
        # Validate scalar CSR topology before trusting decode, rejecting integer truncation/duplicates.
        for k in ('moment_rownnz','moment_rowadr','moment_colind'):
            require(np.issubdtype(a[k].dtype,np.integer),label+' noninteger '+k)
        nnz=a['moment_rownnz']; adr=a['moment_rowadr']; cols=a['moment_colind']
        for s in range(SAMPLES):
            for u in range(90):
                n=int(nnz[s,u]); lo=int(adr[s,u]); require(len(np.unique(cols[s,lo:lo+n]))==n,label+' duplicate sparse columns')
        velocity=np.einsum('tud,td->tu',sparse_dense(a),a['qvel'])
        equal(a['actuator_velocity'],velocity,label+' per-actuator moment@qvel',2e-12)
        dyn['velocity_identity_max_abs']=float(np.max(np.abs(velocity-a['actuator_velocity'])))
        jointrows=limits(a,joints)
        weight=float(r['weight_native']); require(np.isfinite(weight) and weight>0,'native weight')
        if checks: require(weight==checks[0]['weight_native'],'matched native weight')
        m,_=physical_metrics(a,weight); load=a['foot_force'][:,:,2].sum(1)/weight
        jl=float(np.trapezoid((np.maximum(.8-load,0)/.8)**2,dx=DT_BODY)/.01)
        jj=float(np.trapezoid((np.maximum(a['limit_excess']-1e-6,0)/.001)**2,dx=DT_BODY)/.01)
        require(abs(jl-r['Jload'])<=1e-12 and abs(jj-r['Jlimit'])<=1e-12,label+' independent penalties')
        m.update(label=label,passed=True,weight_native=weight,Jload=jl,Jlimit=jj,all42_joint_limits=jointrows,
                 yaw_qpos7=extrema(a['qpos'][:,7],a['time']),pitch_qpos8=extrema(a['qpos'][:,8],a['time']),**dyn)
        checks.append(m); plotted.append((label,a,m))
    require('mujoco' not in sys.modules,'verifier imported physics')
    pairs=pairwise(plotted)
    report=dict(passed=True,verifier=str(Path(__file__).resolve()),verifier_sha256=sha(__file__),
        new_physics_calls=0,new_neural_ticks=0,ledger=ledger,source_hashes=hashes,observed_artifact_hashes=observed_hashes,
        checks=checks,pairwise_response=str(out/'PAIRWISE_RESPONSE.json'),physical_units='NOT_ESTABLISHED',behavior_claim='NONE',
        decision='DIRECTIONAL_INCREMENT_AND_SAFETY_LOAD_TRADEOFF_ONLY_NO_CHECKPOINT_SELECTION',
        caveats=['No network/physical rerun. Saved Ma is trusted; mass matrix multiplication is not independently reconstructed.',
            'Foot forces/contact counts/nonfoot depth are saved inputs, not reconstructed from raw contacts.',
            'Fixed-reference algebra is an exact trajectory identity, not an additional intervention or causal partition.',
            'Initialization is computational only; 10ms response is not standing, preheating or walking success.',
            'Hashes/ledger do not independently prove absence of every unrecorded runtime call.'])
    return report,pairs,plotted


def plot(out,plotted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,1,figsize=(12,15),sharex=True)
    for label,a,m in plotted:
        t=a['time']*1000; q=a['qpos']; quat=q[:,3:7]; norms=np.linalg.norm(quat,axis=1)
        angle=2*np.arccos(np.clip(np.abs(quat@quat[0])/(norms*norms[0]),0,1))
        curves=(a['foot_force'][:,:,2].sum(1)/m['weight_native'],a['limit_excess'],angle,q[:,2]/q[0,2],a['actuator_force'][:,4]*a['actuator_moment'][:,4,8])
        for ax,curve in zip(axes,curves): ax.plot(t,curve,label=label)
    for ax,value in zip(axes[:4],(.8,1e-6,.35,.85)): ax.axhline(value,color='black',ls=':')
    for ax,name in zip(axes,('Foot Fz / native weight','All42 max limit excess (rad)','Root angle from initial (rad)','Root height ratio','Actuator4 actual torque DOF8 (native)')):
        ax.set_ylabel(name); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=4)
    axes[-1].set_xlabel('Physical time after release (ms); computational initialization not physical support')
    fig.suptitle('Four-arm pool synergy response: saved-array verification only\nDirectional increments and safety/load tradeoff; no checkpoint selection')
    fig.tight_layout(); path=Path(out)/'POOL_SYNERGY_COMPARISON.png'; fig.savefig(path,dpi=150); plt.close(fig)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output.resolve(); report,pairs,plotted=verify(out)
    report['comparison_png']=str(plot(out,plotted))
    for name,obj in (('PAIRWISE_RESPONSE.json',pairs),('INDEPENDENT_CHECK.json',report)):
        (out/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,independent_check=str(out/'INDEPENDENT_CHECK.json'),pairwise_response=str(out/'PAIRWISE_RESPONSE.json'),comparison_png=report['comparison_png']),indent=2))


if __name__=='__main__': main()
