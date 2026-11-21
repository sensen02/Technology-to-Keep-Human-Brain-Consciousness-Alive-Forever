"""Actual paired brain <-> fixed BANC <-> independent complete bodies."""
import argparse
import copy
from types import SimpleNamespace
from scipy.sparse import csr_matrix
from arena_trial import TRIAL, MixedBody, measure as terrain_measure
from neural_runtime import *
from repair_binding import FrontTouchNeck, CanonicalEvents

# Keep the body's archived `shared`/`context` modules separate from the
# translation experiment's modules. The frozen eight-parameter forward below
# is checked bitwise against its saved independent evaluation before use.
STOP=BASE/'20261117T202200Z-body-stop-repair'
sys.path.insert(0,str(STOP))
from policy_repair import read as load_actor
from shared import Body,feature,observation,advance,phase_state,setup
from proprioception import extra_live
import mujoco
from adapters import CalibratedReceiver
sys.path.insert(0,str(BASE/'20261117T071007Z-ground-dn-response'))
from ground_response import Ground

MODES=('native_full','reference_boundary','translated','repaired')
GROUND={'forward','forward_speed','walking_steer','walking_steer_context','walking_outer_stride',
        'halt_bluebell','halt_web_a','halt_web_b','backward_moonwalker'}
BODY_WEIGHT=BASE/'20261117T211320Z-body-command-calibration/FIT/epoch040.pt'
ENCODER_WEIGHT=FIT/'FIT/update250.pt'


class FrozenEncoder:
    def __init__(self):
        saved=torch.load(ENCODER_WEIGHT,weights_only=True,map_location='cpu')
        self.mix=saved['state']['mix_logits'];self.gain=saved['state']['log_gain']

    def __call__(self,x):
        level=(x*self.mix.softmax(-1)).sum(-1)
        gain=self.gain.clamp(-2,2).exp()
        return 1000.*(-torch.expm1(-gain*level/1000.))


def qualify_encoder(enc):
    with np.load(FIT/'DATA/held_left/PAIRED.npz',allow_pickle=False) as a:x=a['features_hz'].copy()
    with np.load(FIT/'EVAL/held_left/PREDICTIONS.npz',allow_pickle=False) as a:y=a['prediction'].copy()
    actual=enc(torch.from_numpy(x).double()).numpy()
    assert np.array_equal(actual,y),'frozen online encoder differs from independently saved model'
    return dict(frames=len(x),bitexact_saved_prediction=True,weights_sha256=sha(ENCODER_WEIGHT))


@torch.inference_mode()
def run(seed,out):
    torch.set_num_threads(1);began=time.monotonic();out.mkdir(parents=True)
    chosen=read(HERE/'PREFLIGHT/RESULT.json')['chosen']
    if chosen is None:raise ValueError('no qualified upstream motor context; no DN-answer fallback')
    verify_upstream();bound=read(FIT/'BINDING.json')
    for path,h in bound['source_sha256'].items():assert sha(path)==h
    enc=FrozenEncoder();encoder_parity=qualify_encoder(enc)
    catalog=read(BIDIR/'CATALOG.json')['ports'];admitted=read(BIDIR/'EON/RESULT.json')['admitted']
    ground_roots=[r for r in admitted if catalog[r]['group'] in GROUND]
    assert len(ground_roots)==17
    index,_=eon.get_hash_tables(eon.path_comp);weights=load_connectome(index)
    an_roots=[r['banc_root'] for r in bound['records']];source_idx=[r['flywire_index'] for r in bound['records']]
    dn_idx=[index[int(admitted[r])] for r in ground_roots]
    brain=ContextBrain(FrozenCPUBrain(4,weights),source_idx,(1,2,3))
    neck_ports={r:p for r,p in catalog.items() if p['group'] in GROUND or r in an_roots}
    neck=FrontTouchNeck({'ports':neck_ports},an_roots,
                        matched_lanes=(1,3),feedback_lanes=(3,),batch=4)
    an_cols=[neck.port_index[r] for r in an_roots];dn_cols=[neck.port_index[r] for r in ground_roots]
    receptors=[neck.validate_inputs([r for r,_ in catalog[root]['incoming_receptors']],'body') for root in an_roots]
    receivers=[CalibratedReceiver(neck) for _ in MODES]
    anchors=read(BIDIR/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json')['probes']
    for receiver in receivers:
        for key,(_,_,sign) in receiver.probes.items():
            roots=anchors['/'.join(key)];receiver.probes[key]=roots,[neck.index[r] for r in roots],sign
    calibration=read(BASE/'20261117T071007Z-ground-dn-response/REFINE/CALIBRATION.json')
    decoders=[SimpleNamespace(receiver=r,calibration=calibration) for r in receivers]
    # The native_full comparison owns its ANs inside the original brain. Only
    # receptor cells are simulated here; no second copy of those AN somata.
    aff_roots=sorted({r for root in an_roots for r,_ in catalog[root]['incoming_receptors']})
    aff_index={r:i for i,r in enumerate(aff_roots)}
    aff=eon.AlphaLIF(1,len(aff_roots),eon.DT,eon.MODEL_PARAMS);aff_state=aff.state_init()
    rows=[];cols=[];vals=[]
    for c,root in enumerate(an_roots):
        for r,n in catalog[root]['incoming_receptors']:rows.append(c);cols.append(aff_index[r]);vals.append(n)
    aff_to_an=csr_matrix((np.asarray(vals,dtype=np.float32),(rows,cols)),shape=(2,len(aff_roots)))
    aff_inputs=[[aff_index[r] for r,_ in catalog[root]['incoming_receptors']] for root in an_roots]
    bodies=[MixedBody(seed) for _ in MODES]
    for body in bodies:setup(body,'terminal')
    actor=load_actor(BODY_WEIGHT)
    cpgs=[copy.deepcopy(b.motor.cpg_network) for b in bodies];freqs=[c.intrinsic_freqs.copy() for c in cpgs]
    for i,b in enumerate(bodies):
        mujoco.mj_saveModel(b.m,str(out/(MODES[i]+'.mjb')),None)
    model_hashes=[sha(out/(m+'.mjb')) for m in MODES]
    assert len(set(model_hashes))==1,'paired physical models differ'
    assert all(np.array_equal(bodies[0].d.qpos,b.d.qpos) for b in bodies)
    write(out/'ARENA.json',dict(initial=bodies[0].initial_receipt(), tiles=bodies[0].tiles))
    n=20000
    physical={k:np.zeros((n,4,*shape),dtype=np.float64) for k,shape in dict(
        qpos=(bodies[0].m.nq,),qvel=(bodies[0].m.nv,),position=(3,),yaw=(),upright=(),
        nonfoot=(),control=(42,),pads=(6,),normal_load=(6,),command=(2,),
        terrain_normal=(7,6),feet_xyz=(6,3)).items()}
    neural={k:np.zeros((n,*shape),dtype=dtype) for k,(shape,dtype) in dict(
        dn_emitted=((4,17),bool),dn_arrived=((4,17),bool),an_emitted=((4,2),bool),an_arrived=((4,2),bool),
        brain_an_events=((4,2),bool),rates=((4,2),float),population=((4,),np.int32),
        full_an_spikes=((2,),bool),central_hits=((len(context_groups(index)[chosen]),),bool)).items()}
    initial=[b.d.xpos[b.thorax].copy() for b in bodies]
    command=np.zeros((4,2));loads=np.zeros((4,2));held_loads=np.zeros_like(loads)
    drive=np.zeros((4,len(neck.roots)),dtype=np.float32)
    aff_drive=np.zeros((1,len(aff_roots)),dtype=np.float32);extra=np.zeros((4,len(index)),dtype=np.float32)
    edrive=np.zeros((4,len(neck_ports)),dtype=bool);cut=np.zeros(len(neck_ports),dtype=bool)
    filter_state=np.zeros((4,2,3));alpha=-np.expm1(-DT/np.array([.005,.020,.050]))
    canonical=CanonicalEvents((4,2));legacy_phase=np.zeros(2)
    context_indices=np.array(context_groups(index)[chosen],dtype=int)
    rng=np.random.default_rng(np.random.SeedSequence([seed,18041]));stimulus=rng.random((n,len(context_indices)))<.01
    controls=[None]*4;pads=[None]*4
    extra_files={FIT/'BINDING.json',ENCODER_WEIGHT,BODY_WEIGHT,FIXED/'PARAMETERS.json',FIXED/'READOUT.json',
         BIDIR/'CATALOG.json',BIDIR/'EON/RESULT.json',BIDIR/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json',
         BASE/'20261117T071007Z-ground-dn-response/REFINE/CALIBRATION.json',HERE/'PREFLIGHT/RESULT.json',eon.path_comp,eon.path_con}
    # Include the actual archived physical/model asset receipts used by the body.
    assets=read(BASE/'20261117T184709Z-body-terrain-repair/PREPARE_INPUTS.json')['source_sha256'];verify(assets)
    extra_files.update(TRIAL.glob('*.py'));extra_files.add(TRIAL/'PROTOCOL.zh-CN.md')
    pins=sources(extra_files);pins.update(assets)
    write(out/'INPUTS.json',dict(seed=seed,modes=MODES,context=chosen,context_rate_hz=100.,
        context_on_tick=2200,context_off_tick=15000,source_sha256=pins,body_model_hashes=model_hashes,
        encoder_parity=encoder_parity,source_indices=source_idx,dn_roots=ground_roots,neck_ports=list(neck_ports),
        coupling='real time; each physical body feeds its own brain; DN output enters neck next tick (+.1ms)',
        boundary_feedback_excluded_lanes=[1,2,3],excluded_feedback_edges=int(weights[source_idx,:].nnz),
        matched_lanes=[1,3],matched_feedback_lanes=[3],arena='FlyGym 2.1.0 MixedTerrainWorld',
        native_full_brain_unchanged=True,teacher_body_encoder='new engineering BANC receptor->AN current; not archived native body interface'))
    warning=None
    for tick in range(n):
        if tick%5==0:held_loads=loads.copy()
        use=(held_loads>.04)&(tick%23==0)
        drive.fill(0.);aff_drive.fill(0.)
        for lane in range(4):
            for c in range(2):
                if use[lane,c]:drive[lane,receptors[c]]=68.75
        for c in range(2):
            if use[0,c]:aff_drive[0,aff_inputs[c]]=68.75
        extra.fill(0.);extra[0,source_idx]=np.asarray(aff_to_an@aff_state[2].numpy().T).ravel()
        aff_state=aff(torch.zeros_like(aff_state[2]),torch.from_numpy(aff_drive),*aff_state)
        edrive.fill(False);edrive[:,dn_cols]=brain.spikes[:,dn_idx].numpy().astype(bool)
        emitted,arrivals,_,voltage=neck.step(drive,cut,eon_output=edrive)
        filter_state+=alpha*(arrivals[:,an_cols,None]/DT-filter_state)
        events=canonical.step(arrivals[:,an_cols])
        rates=canonical.rate.copy();rates[0]=0.
        rates[2]=enc(torch.from_numpy(filter_state[2]).double()).numpy()
        events[0]=False
        legacy_phase+=rates[2]*DT;events[2]=legacy_phase>=1.;legacy_phase-=events[2]
        active=context_indices[stimulus[tick]].tolist() if 2200<=tick<15000 else []
        brain.step([active]*4,events,extra)
        neural['dn_emitted'][tick]=brain.spikes[:,dn_idx].numpy().astype(bool)
        neural['dn_arrived'][tick]=arrivals[:,dn_cols]
        neural['an_emitted'][tick]=emitted[:,an_cols];neural['an_arrived'][tick]=arrivals[:,an_cols]
        neural['brain_an_events'][tick]=events;neural['rates'][tick]=rates
        neural['population'][tick]=brain.spikes.sum(1).numpy().astype(np.int32)
        neural['full_an_spikes'][tick]=brain.spikes[0,source_idx].numpy().astype(bool)
        neural['central_hits'][tick]=stimulus[tick] if 2200<=tick<15000 else False
        if tick%5==0:
            for lane in range(4):
                receivers[lane].update(voltage[lane:lane+1],DT*5)
                command[lane]=Ground.decode(decoders[lane])
        for lane,(b,cpg,freq) in enumerate(zip(bodies,cpgs,freqs,strict=True)):
            advance(cpg,freq,command[lane]);phase_now=phase_state(cpg)
            if tick%10==0:
                obs=observation(b)
                x=np.r_[feature(b.d.qpos.copy(),b.d.qvel.copy(),obs.stumbling_contact_forces,
                               command[lane],phase_now,b.weight),extra_live(b)].astype(np.float32)
                controls[lane],pads[lane],_=actor.act(x,phase_now)
            b.apply(controls[lane],pads[lane]);load,nf,ground=b.detail();loads[lane]=load[[0,3]]
            pos=b.d.xpos[b.thorax];rot=b.d.xmat[b.thorax].reshape(3,3)
            values=dict(qpos=b.d.qpos,qvel=b.d.qvel,position=pos,yaw=np.arctan2(rot[1,0],rot[0,0]),
                 upright=rot[2,2],nonfoot=nf,control=controls[lane],pads=pads[lane],normal_load=load,command=command[lane],
                 terrain_normal=ground,feet_xyz=b.d.xpos[b.feet])
            for key,value in values.items():physical[key][tick,lane]=value
            if b.d.warning.number.any() or not np.isfinite(b.d.qpos).all():
                warning=dict(lane=lane,tick=tick,model=b.d.warning.number.tolist())
        if warning is not None:
            print('physical warning',warning,flush=True);break
        if (tick+1)%5000==0:print(seed,'live ticks',tick+1,'DN',neural['dn_emitted'][:tick+1].sum((0,2)).tolist(),round(time.monotonic()-began,1),flush=True)
        if time.monotonic()-began>600:raise TimeoutError('paired live behavior budget')
    n=tick+1
    neural={key:array[:n] for key,array in neural.items()}
    result=[]
    for lane,mode in enumerate(MODES):
        folder=out/mode;folder.mkdir();arrays={k:v[:n,lane].copy() for k,v in physical.items()};arrays['initial_position']=initial[lane]
        np.savez_compressed(folder/'TRACE.npz',**arrays)
        metrics=terrain_measure(arrays,bodies[lane]);metrics.update(mode=mode,seed=seed,steps=n,teacher_motor_calls=0,
            dn_events_by_root={r:int(neural['dn_emitted'][:,lane,j].sum()) for j,r in enumerate(ground_roots)},
            dn_arrivals_by_root={r:int(neural['dn_arrived'][:,lane,j].sum()) for j,r in enumerate(ground_roots)},
            trace_sha256=sha(folder/'TRACE.npz'),body_model=str(out/(mode+'.mjb')))
        write(folder/'RESULT.json',metrics);result.append(metrics)
    np.savez_compressed(out/'NEURAL.npz',**neural)
    verify(pins)
    write(out/'RESULT.json',dict(seed=seed,steps=n,body_episodes=4,native_body_steps=n*4,
        brain_batch_ticks=n,brain_lane_ticks=n*4,neck_batch_ticks=n,neck_lane_ticks=n*4,
        physics_warning=warning,matched_AN_soma_ticks=n*2,native_receptor_lif_ticks=n,optimizer_updates=0,
        neural_sha256=sha(out/'NEURAL.npz'),source_sha256=pins,wall_seconds=time.monotonic()-began,
        metrics=result,dn_roots=ground_roots))
    print(seed,'behavior batch complete',[(r['mode'],round(r['forward_dx_mm'],3),round(r['yaw_degrees'],2)) for r in result],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);p.add_argument('output',type=Path);a=p.parse_args();run(a.seed,a.output)
