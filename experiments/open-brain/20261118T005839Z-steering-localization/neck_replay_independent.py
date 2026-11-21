"""Actual source-qualified DN event replacements through an exact readout subgraph."""
from types import SimpleNamespace
from scope_diag import *

NAMES=('native','reference','translated','reference_dn_student_sense','student_dn_reference_sense',
       'swap_all_steer','swap_DNa02','swap_DNa01','swap_DNg13','swap_propulsion')


def run(seed):
    start=time.monotonic();folder=PREVIOUS/'FORMAL'/f'seed{seed}';prior=read(folder/'RESULT.json')
    verify(prior['source_sha256'])
    with np.load(folder/'NEURAL.npz',allow_pickle=False) as a:dn=a['dn_emitted'].copy()
    catalog=read(BIDIR/'CATALOG.json')['ports'];order=read(folder/'INPUTS.json')['neck_ports']
    anchors=read(BIDIR/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json')['probes']
    full={r:catalog[r] for r in order};probes={r for key,rs in anchors.items() if key.split('/')[0] in GROUND for r in rs}
    retained=set(order)|probes|{r for p in full.values() for r,_ in p['incoming_receptors']}
    full_roots=set(order)|{r for p in full.values() for r,_ in p['incoming_receptors']}|{r for p in full.values() for r,_ in p['outgoing']}
    projected={root:dict(port,outgoing=[edge for edge in port['outgoing'] if edge[0] in retained]) for root,port in full.items()}
    neck=FixedCircuit({'ports':projected},batch=len(NAMES))
    receivers=[CalibratedReceiver(neck) for _ in NAMES]
    for receiver in receivers:
        for key,(_,_,sign) in receiver.probes.items():
            roots=anchors['/'.join(key)];receiver.probes[key]=roots,[neck.index[r] for r in roots],sign
    cal=read(BASE/'20261117T071007Z-ground-dn-response/REFINE/CALIBRATION.json')
    decoders=[SimpleNamespace(receiver=r,calibration=cal) for r in receivers]
    roots=prior['dn_roots'];dn_cols=[neck.port_index[r] for r in roots]
    source=np.zeros((len(dn),len(NAMES),len(roots)),dtype=bool)
    source[:,0]=dn[:,0];source[:,1]=dn[:,1];source[:,2:]=dn[:,2,None,:]
    source[:,3]=dn[:,1]
    groups=[catalog[r]['group'] for r in roots]
    replacements={5:{'walking_steer','walking_steer_context','walking_outer_stride'},
         6:{'walking_steer'},7:{'walking_steer_context'},8:{'walking_outer_stride'},9:{'forward','forward_speed'}}
    for lane,wanted in replacements.items():
        cols=np.flatnonzero([g in wanted for g in groups]);source[:,lane,cols]=dn[:,1,cols]
        other=np.flatnonzero([g not in wanted for g in groups]);assert np.array_equal(source[:,lane,other],dn[:,2,other])
    sense=[]
    for mode in ('native_full','reference_boundary','translated'):
        with np.load(folder/mode/'TRACE.npz',allow_pickle=False) as a:sense.append(a['normal_load'][:,[0,3]].copy())
    sense_lanes=[0,1,2,2,1,2,2,2,2,2]
    touch_roots=[r for r,p in full.items() if p['group']=='front_touch']
    receptors=[neck.validate_inputs([r for r,_ in full[root]['incoming_receptors']],'body') for root in touch_roots]
    drive=np.zeros((len(NAMES),len(neck.roots)),dtype=np.float32)
    ext=np.zeros((len(NAMES),len(neck.ports)),dtype=bool);cut=np.zeros(len(neck.ports),dtype=bool)
    held=np.zeros((len(NAMES),2));command=np.zeros((len(dn),len(NAMES),2))
    keys=sorted(receivers[0].filtered)
    levels=np.zeros((len(dn),len(NAMES),len(keys)));raw_levels=np.zeros_like(levels)
    pins=sources({HERE/'NECK_REPLAY_ADDENDUM.zh-CN.md',folder/'NEURAL.npz',folder/'INPUTS.json',folder/'RESULT.json',
       *(folder/m/'TRACE.npz' for m in ('native_full','reference_boundary','translated')),
       FIXED/'PARAMETERS.json',FIXED/'READOUT.json',BIDIR/'CATALOG.json',
       BIDIR/'PHYSICAL_AN_PAIR/ground_forward/RESULT.json',BASE/'20261117T071007Z-ground-dn-response/REFINE/CALIBRATION.json'})
    last=np.zeros((len(NAMES),2))
    for tick in range(len(dn)):
        if tick and tick%5==0:held=np.array([sense[s][tick-1] for s in sense_lanes])
        drive.fill(0.)
        if tick%23==0:
            for lane in range(len(NAMES)):
                for c in range(2):
                    if held[lane,c]>.04:drive[lane,receptors[c]]=68.75
        ext.fill(False)
        if tick:ext[:,dn_cols]=source[tick-1]
        _,_,_,v=neck.step(drive,cut,eon_output=ext)
        if tick%5==0:
            for lane,r in enumerate(receivers):
                r.update(v[lane:lane+1],DT*5);last[lane]=Ground.decode(decoders[lane])
        command[tick]=last
        for lane,r in enumerate(receivers):
            levels[tick,lane]=[r.filtered[k] for k in keys];raw_levels[tick,lane]=[r._raw[k] for k in keys]
        if time.monotonic()-start>240:raise TimeoutError('neck replay budget')
    with np.load(HERE/'COMMANDS'/f'seed{seed}'/'BANK.npz',allow_pickle=False) as a:
        matching={name:bool(np.array_equal(a[name],command[:,i])) for i,name in enumerate(NAMES[:3])}
    out=HERE/'NECK_REPLAY'/f'seed{seed}';out.mkdir(parents=True)
    np.savez_compressed(out/'TRACE.npz',command=command,levels=levels,raw_levels=raw_levels,source_events=source)
    verify(pins)
    write(out/'RESULT.json',dict(seed=seed,names=NAMES,canonical_command_exact=matching,
       full_graph_nodes=len(full_roots),projected_nodes=len(neck.roots),projection='all event sources and readout targets retained with original edges',
       level_keys=['/'.join(k) for k in keys],source_sha256=pins,trace_sha256=sha(out/'TRACE.npz'),
       actual_neck_batch_ticks=len(dn),actual_neck_lane_ticks=len(dn)*len(NAMES),physical_steps=0,
       interventions={NAMES[k]:sorted(v) for k,v in replacements.items()},sense_context=sense_lanes,
       wall_seconds=time.monotonic()-start,optimizer_updates=0))
    print(seed,'neck projection',len(full_roots),'->',len(neck.roots),'canonical exact',matching,round(time.monotonic()-start,2),flush=True)
    if not all(matching.values()):raise ValueError('projection failed exact readout parity; no counterfactual use')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('seed',type=int);a=p.parse_args();torch.set_num_threads(1);run(a.seed)
