"""Build canonical neural continuation and fixed unlabelled sameleg SVD basis."""
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import numpy as np
from runtime import HERE,CORE,CORE_SHA256,LEG_ORDER,new_circuit,snapshot,restore_circuit,Actor,load

PROJECT=Path('/run/media/sensen/Data2/cell_wound_prototype')
PRIOR=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop')
TRACE=PROJECT/'experiments/banc-local-learning/20261107T192931Z/dynamics/SOURCE_TRACE.npz'
OLD_TRAIN=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz')
ENCODER=PROJECT/'experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py'
INTERFACE=PROJECT/'experiments/flybody-teacher/20261107T180844Z/mapping/STUDENT_INTERFACE.json'
LEDGER={'completed_neural_steps':0,'physical_steps':0,'optimizer_updates':0,'basis_SVD_factorizations':0}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def advance(c,s,d,duration=5.):
    before=c.net.steps
    try:return c.advance(s,d,duration)
    finally:LEDGER['completed_neural_steps']+=c.net.steps-before

def equal(a,b):return set(a)==set(b) and all(np.array_equal(a[k],b[k]) for k in a)


def main():
    if (HERE/'FOUNDATION_RESULTS.json').exists():raise RuntimeError('refuse completed evidence overwrite')
    sources=[CORE,ENCODER,TRACE,OLD_TRAIN,INTERFACE,PRIOR/'HANDOVER.npz',PRIOR/'POLICY.npz',PRIOR/'MODEL.mjb',PRIOR/'INITIAL.npz',PRIOR/'PROTOCOL.json',HERE/'PREREGISTRATION.json']
    hashes={str(p):sha(p) for p in sources};assert hashes[str(CORE)]==CORE_SHA256
    spec=importlib.util.spec_from_file_location('foundation_frozen_encoder',ENCODER);em=importlib.util.module_from_spec(spec);spec.loader.exec_module(em);encoder=em.SensoryEncoder()
    c=new_circuit();assert np.array_equal(c.root_ids,encoder.root_ids)
    idx=encoder.local_indices;assert len(idx)==556 and np.isin(idx,c.allowed_sensory_local).all()
    # Read only current and start-neural fields, never POLICY predicted actions.
    with np.load(PRIOR/'POLICY.npz') as z:
        currents=z['sensory_current_nA'][:201].copy();policy_rates=z['mn_rates_hz'][:201].copy();policy_voltage=z['mn_voltages_mV'][:201].copy();policy_time=z['time_s'][:201].copy();warm=z['teacher_warmup'][:201]
    assert currents.shape==(201,556) and np.all(currents>=0) and np.isfinite(currents).all()
    assert warm[:200].all() and not warm[200] and abs(policy_time[200]-1)<1e-9
    zero=np.zeros(20000)
    prefix_error=0.
    for k in range(200):
        prefix_error=max(prefix_error,float(np.max(np.abs(c.filtered_rates_hz[c.mn_local]-policy_rates[k]))),float(np.max(np.abs(c.net.v[c.mn_local]-policy_voltage[k]))))
        s=np.zeros(20000);s[idx]=currents[k];advance(c,s,zero)
    assert prefix_error==0 and c.net.steps==2000 and c.net.t_ms==1000
    hand=load(PRIOR/'HANDOVER.npz')
    mapping={'v':'neural_v','ge':'neural_ge','gi':'neural_gi','g_adapt':'neural_g_adapt_uS','_ring':'neural_delay_ring','_ring_i':'neural_ring_i'}
    matches={k:bool(np.array_equal(getattr(c.net,k),hand[v])) for k,v in mapping.items()}
    matches['MN_rates']=bool(np.array_equal(c.filtered_rates_hz[c.mn_local],hand['mn_rates_hz']))
    matches['POLICY200_MN_rates']=bool(np.array_equal(c.filtered_rates_hz[c.mn_local],policy_rates[200]))
    matches['POLICY200_MN_voltage']=bool(np.array_equal(c.net.v[c.mn_local],policy_voltage[200]))
    assert all(matches.values())
    start=snapshot(c);np.savez_compressed(HERE/'NEURAL_START.npz',**start)
    next_current=np.zeros(20000);next_current[idx]=currents[200]
    np.savez_compressed(HERE/'CONTINUATION_INPUT.npz',sensory_current_nA=next_current,descending_current_nA=zero,source_policy_row=200)
    advance(c,next_current,zero);continued=snapshot(c)
    np.savez_compressed(HERE/'CONTINUATION_EXPECTED.npz',**continued)
    restored=new_circuit();restore_circuit(restored);assert equal(snapshot(restored),start)
    advance(restored,next_current,zero);assert equal(snapshot(restored),continued)
    restore_circuit(restored);advance(restored,next_current,zero);assert equal(snapshot(restored),continued)
    # Source basis uses rates and declared masks ONLY; no action label fields loaded.
    with np.load(TRACE) as z:rates=z['mn_rates_hz'];roots=z['root_ids'];mn=z['mn_local'];mnleg=z['mn_leg'];times=z['time_s']
    with np.load(OLD_TRAIN) as z:train=z['train_mask'].copy();oldtimes=z['time_s'].copy()
    assert np.array_equal(times,oldtimes) and train.sum()==1160
    # Preserve the exact frozen old mask: its boundary includes index40 at 0.19999999999998003s due to source floating time; do not recompute it.
    assert np.array_equal(roots,c.root_ids) and np.array_equal(mn,c.mn_local) and np.array_equal(mnleg,c.mn_leg)
    interface=json.loads(INTERFACE.read_text());joint_names=np.array(interface['output']['ordered_joint_names']);joint_leg=np.array([name.split('_')[1][:2] for name in joint_names])
    assert np.array_equal(joint_names,encoder.joint_names) and np.array_equal(joint_leg,np.repeat(np.array(LEG_ORDER),7))
    basis=np.zeros((323,24));rawbasis=basis.copy();scales=np.empty(24);rms=np.empty(24);singular=np.empty((6,4));legstats={}
    for i,leg in enumerate(LEG_ORDER):
        ni=np.flatnonzero(mnleg==leg.lower());x=rates[train][:,ni]/100.
        _,s,vh=np.linalg.svd(x,full_matrices=False);LEDGER['basis_SVD_factorizations']+=1
        vectors=vh[:4].T.copy()
        for col in range(4):
            pivot=int(np.argmax(np.abs(vectors[:,col])))
            if vectors[pivot,col]<0:vectors[:,col]*=-1
        projected=x@vectors;localrms=np.sqrt(np.mean(projected**2,axis=0));scale=np.maximum(localrms,.05)
        rawbasis[np.ix_(ni,np.arange(4*i,4*i+4))]=vectors
        basis[np.ix_(ni,np.arange(4*i,4*i+4))]=vectors/scale
        rms[4*i:4*i+4]=localrms;scales[4*i:4*i+4]=scale;singular[i]=s[:4]
        orth=float(np.max(np.abs(vectors.T@vectors-np.eye(4))));assert orth<1e-12
        legstats[leg]={'MN_count':len(ni),'singular_values_top4':s[:4].tolist(),'training_latent_RMS':localrms.tolist(),'denominator':scale.tolist(),'gain':(1/scale).tolist(),'uncentered_energy_fraction_top4':float(np.sum(s[:4]**2)/np.sum(s**2)),'orthogonality_max_error':orth}
    assert np.all(1/scales<=20) and np.all(np.isfinite(basis))
    np.savez_compressed(HERE/'BASE.npz',basis=basis,unwhitened_basis=rawbasis,training_latent_rms=rms,normalization_denominator=scales,normalization_gain=1/scales,singular_values_top4=singular,root_ids=roots,mn_local=mn,mn_root_ids=roots[mn],mn_leg=mnleg,joint_names=joint_names,joint_leg=joint_leg,leg_order=np.array(LEG_ORDER),train_mask=train,train_source_time_s=times[train],source_trace_sha256=np.array(hashes[str(TRACE)]),neural_module_sha256=np.array(CORE_SHA256),theta_bounds=np.array([-.25,.25]),weight_bounds=np.array([-1.,1.]),rate_limit_rad_s=np.array(30.))
    dump('BASIS.json',{'status':'FIXED_UNCENTERED_UNLABELLED_ENGINEERING_BASIS','shape':[323,24],'leg_order':list(LEG_ORDER),'training_rows':int(train.sum()),'source_rates':'start-of-bin MNHz/100','normalization':'divide projected latent by max(trainingRMS,.05);gain<=20;no centering','labels_read':False,'leg_statistics':legstats,'BASE_sha256':sha(HERE/'BASE.npz'),'coefficient_shape':[6,4,7],'theta_bounds':[-.25,.25],'weight_projection':[-1,1],'action':'30tanh(MNHz/100 @ W);zero bias/nohistory','joint_names':joint_names.tolist(),'source_hashes':hashes})
    # Actor tests use explicit deterministic synthetic MN features, not physical behavior.
    a0=Actor();assert np.array_equal(a0.action(np.ones(323)*123),np.zeros(42))
    rng=np.random.default_rng(195435);actor_error=0.;clipped_max=0
    for n in range(16):
        theta=rng.uniform(-.25,.25,(6,4,7));actor=Actor(theta);features=rng.uniform(0,500,323)
        assert np.array_equal(actor.action(np.zeros(323)),np.zeros(42))
        manual=np.zeros((323,42));rawmax=0
        for i,leg in enumerate(LEG_ORDER):
            ni=np.flatnonzero(mnleg==leg.lower());ji=np.flatnonzero(joint_leg==leg)
            for row in ni:
                for j,col in enumerate(ji):
                    val=sum(float(basis[row,4*i+k])*float(theta[i,k,j]) for k in range(4));rawmax=max(rawmax,abs(val));manual[row,col]=min(1.,max(-1.,val))
        actor_error=max(actor_error,float(np.max(np.abs(actor.weights-manual))),float(np.max(np.abs(actor.action(features)-30*np.tanh(features/100@manual)))))
        assert actor_error<1e-10 and np.all(np.abs(actor.weights)<=1) and np.all(np.abs(actor.action(features))<=30)
        cross=mnleg[:,None]!=np.char.lower(joint_leg)[None,:];assert not np.any(actor.weights[cross])
        clipped_max=max(clipped_max,int(np.count_nonzero(np.abs(actor.weights)==1)))
    # Transactional corruption checks all occur without simulation steps.
    restore_circuit(restored);before=snapshot(restored);fixture=HERE/'_invalid_snapshot.npz';rejected=[]
    def corrupt(label,key,change):
        bad={k:v.copy() for k,v in start.items()};bad[key]=change(bad[key]);np.savez_compressed(fixture,**bad)
        try:restore_circuit(restored,fixture)
        except ValueError:rejected.append(label)
        else:raise AssertionError('corrupt state accepted:'+label)
        assert equal(snapshot(restored),before)
    corrupt('shape_v','v',lambda x:x[:-1]);corrupt('nan_v','v',lambda x:np.full_like(x,np.nan));corrupt('negative_ge','ge',lambda x:-np.ones_like(x));corrupt('negative_gi','gi',lambda x:-np.ones_like(x));corrupt('negative_gA','g_adapt',lambda x:-np.ones_like(x));corrupt('negative_rate','filtered_rates_hz',lambda x:-np.ones_like(x));corrupt('negative_count','spike_count',lambda x:-np.ones_like(x));corrupt('bad_ring_shape','_ring',lambda x:x[:-1]);corrupt('bad_ring_i','_ring_i',lambda x:x+1);corrupt('bad_time','t_ms',lambda x:x+1);corrupt('bad_root','root_ids',lambda x:x[::-1]);corrupt('bad_mnorder','mn_local',lambda x:x[::-1]);corrupt('bad_contract','parameter_contract_json',lambda x:np.array('invalid'))
    fixture.unlink()
    guard_count=0
    for invalid in [np.zeros(167),np.full(168,.251),np.full(168,np.nan),np.full(168,np.inf)]:
        try:Actor(invalid)
        except ValueError:guard_count+=1
        else:raise AssertionError('invalid actor accepted')
    for invalid in [np.zeros(322),np.full(323,-1.),np.full(323,np.nan),np.full(323,np.inf)]:
        try:a0.action(invalid)
        except ValueError:guard_count+=1
        else:raise AssertionError('invalid MN input accepted')
    for mode in ('negative','nan','MNrole','wrongshape'):
        bad=np.zeros(20000)
        if mode=='negative':bad[idx[0]]=-.01
        if mode=='nan':bad[idx[0]]=np.nan
        if mode=='MNrole':bad[c.mn_local[0]]=.01
        if mode=='wrongshape':bad=bad[:-1]
        try:advance(restored,bad,zero)
        except ValueError:guard_count+=1
        else:raise AssertionError('invalid current accepted')
        assert equal(snapshot(restored),before)
    protocol=json.loads((PRIOR/'PROTOCOL.json').read_text())
    canonical={'status':'VERIFIED_CANONICAL_PHYSICS1S_NEURAL1000MS','physics_state_source':str(PRIOR/'HANDOVER.npz'),'physics_state_key':'physics_state','actuator_target_key':'actuator_target','handover_sha256':hashes[str(PRIOR/'HANDOVER.npz')],'model_path':str(PRIOR/'MODEL.mjb'),'model_sha256':hashes[str(PRIOR/'MODEL.mjb')],'initial_path':str(PRIOR/'INITIAL.npz'),'initial_sha256':hashes[str(PRIOR/'INITIAL.npz')],'model_original_source':protocol['body_source'],'model_original_source_sha256':protocol['body_source_sha256'],'neural_start_path':str(HERE/'NEURAL_START.npz'),'neural_start_sha256':sha(HERE/'NEURAL_START.npz'),'neural_core':str(CORE),'neural_core_sha256':CORE_SHA256,'encoder_path':str(ENCODER),'encoder_sha256':hashes[str(ENCODER)],'physical_time_s':1.,'neural_time_ms':1000.,'neural_steps_at_start':2000,'source_POLICY_rows':[0,199],'descending_current':'zero','physics_steps_this_task':0,'readout_history':'not restored/used; new Actor is memoryless MN-only, no history/teacher/target input','restore_contract':'restore_circuit(c,path) restores complete source-derived neural state transactionally; defaultrelativepath'}
    dump('CANONICAL.json',canonical)
    assert all(sha(Path(p))==h for p,h in hashes.items()) and LEDGER['completed_neural_steps']==2030
    report={'status':'PASS_PRIMARY_FOUNDATION_CHECKS','ledger':LEDGER,'prefix_start_neural_max_error':prefix_error,'all_available_handover_exact':matches,'complete_restore_exact':True,'next5ms_continuation_exact':True,'repeat_restore_continuation_exact':True,'invalid_restore_cases_rejected_transactionally':rejected,'actor_and_current_guard_rejections':guard_count,'manual_actor_max_error':actor_error,'largest_clipped_weight_count_tested':clipped_max,'all_source_hashes_unchanged':True,'basis_leg_statistics':legstats,'artifacts':{name:sha(HERE/name) for name in ['NEURAL_START.npz','BASE.npz','CANONICAL.json','BASIS.json','runtime.py']},'meaning':'source-exact neural continuation and fixed unlabelled engineering readout foundation; no optimization/physical behavior validation'}
    dump('FOUNDATION_RESULTS.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump('FAILED_BUILD.json',{'error':repr(exc),'traceback':traceback.format_exc(),'ledger':LEDGER});raise
