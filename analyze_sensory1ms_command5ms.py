"""Saved-only sensory1ms/command5ms comparison; run only after producer READY.
Imports no runner, model, neural simulator or decoder. Historical archive immutable.
"""
from pathlib import Path
import json,sys
import numpy as np
import analyze_command1ms_single as base
ROOT=Path('/home/sensen/Desktop/engine修复验收/s4-sensory1ms-command5ms/20261027T225000Z')
DESKTOP=Path('/home/sensen/Desktop'); STEM='S4_SENSORY1MS_COMMAND5MS'
LEGS=base.LEGS

def normalize(iv,ticks):
    """Memory-only repeat5ms holds into1ms views for reused structural analyzer.
    No new filtering, interpolated events, or file modification. Endpoint decoded
    values for four interior views are previous completed-block output, so their
    *first* positive readout remains its actual5ms completion time.
    """
    iv=dict(iv);n=len(iv['commands']);assert n==50
    start=np.arange(250)*.001;end=start+.001
    decoded=iv['decoded_next_controls'];pending=np.vstack([np.zeros((1,90)),decoded[:-1]])
    expanded=np.repeat(pending,5,axis=0);expanded[4::5]=decoded
    out=dict(iv,motor_start_s=start,motor_end_s=end,motor_decoded_next_controls=expanded,
        commands=np.repeat(iv['commands'],5,axis=0),spikes=ticks['spikes'].reshape(250,2,12000).sum(1),
        motor_applied_controls=np.repeat(iv['commands'],5,axis=0),motor_source_end_tick=np.r_[-1,np.arange(2,500,2)])
    # Reused analyzer expects per2tick decoded endpoints; preserve actual logical
    # rates separately, inject only completed-block readout in normalized view.
    synthetic=dict(ticks);logical=ticks['decoded_controls'].copy()
    synthetic['decoded_controls']=np.repeat(expanded,2,axis=0)
    synthetic['decoded_controls'][::2]=logical[::2]
    return out,synthetic

def first_channel_difference(new,old):
    hits=[]
    for leg in LEGS:
        a=new['per_leg'][leg];b=old['per_leg'][leg]
        for channel in a:
            av=a[channel].get('value');bv=b[channel].get('value')
            if av!=bv:hits.append(dict(leg=leg,channel=channel,new_value=av,old_value=bv))
    return hits

def main():
    p=ROOT/'normal';source=[ROOT/'MAPPING.json',p/'BODY.npz',p/'INTERVALS.npz',p/'TICKS.npz',p/'CONTACTS.json',p/'SENSORY.npz',p/'SENSORY_REFRESH.json',base.OLD/'BODY.npz',base.OLD/'INTERVALS.npz',base.OLD/'RECORDS.json',base.OLD/'CONTACTS.json',base.REACTION,base.CLASSIFICATION,Path(__file__),Path(base.__file__)]
    assert all(f.is_file() for f in source),'Producer artifacts not ready'
    hashes={str(f):base.sha(f) for f in source};archive={str(f):base.verify_archive(f) for f in source[7:13]}
    rows=base.read_json(ROOT/'MAPPING.json')['rows'];meta=base.read_json(base.CLASSIFICATION);ground=base.load_npz(base.REACTION)
    oldbody=base.load_npz(base.OLD/'BODY.npz');oldiv=base.load_npz(base.OLD/'INTERVALS.npz');oldrecords=base.read_json(base.OLD/'RECORDS.json');oldcontacts=base.read_json(base.OLD/'CONTACTS.json')
    body=base.load_npz(p/'BODY.npz');iv=base.load_npz(p/'INTERVALS.npz');ticks=base.load_npz(p/'TICKS.npz');contacts=base.read_json(p/'CONTACTS.json');sensory=base.load_npz(p/'SENSORY.npz');refresh=base.read_json(p/'SENSORY_REFRESH.json')
    arms={};worlds={}
    arms['old_sensory5ms'],worlds['old_sensory5ms']=base.summarize_arm('old_sensory5ms',oldbody,oldiv,rows,None,oldrecords,oldcontacts,meta,ground)
    niv,nt=normalize(iv,ticks)
    arms['new_sensory1ms'],worlds['new_sensory1ms']=base.summarize_arm('new_sensory1ms',body,niv,rows,nt,None,contacts,meta,ground)
    new=arms['new_sensory1ms'];tt=ticks['tick_complete_s'];logical=ticks['decoded_controls'];end=iv['interval_end_s'];start=iv['interval_start_s'];decode=iv['decoded_next_controls']
    # Restore true per-tick logical states and real5ms saturation/readout evidence.
    np.testing.assert_array_equal(ticks['spikes'].reshape(50,10,12000).sum(1),iv['spikes'])
    np.testing.assert_array_equal(logical[9::10],decode)
    np.testing.assert_array_equal(iv['commands'][1:],decode[:-1])
    new['first_saved_decode_s']=base.first(logical.any(1),tt)
    new['first_effective_motor_readout_s']=base.first(decode.any(1),end)
    new['saturation_applied']=base.saturation_stats(iv['commands'],end-start)
    new['saturation_decoded_at_window_end']=base.saturation_stats(decode,end-start)
    for leg,r in new['perleg'].items():
        ai=r['actuator_indices'];d=base.first((logical[:,ai]>0).any(1),tt);r['first_saved_decode_s']=d;r['first_effective_motor_readout_s']=base.first((decode[:,ai]>0).any(1),end)
        r['event_to_first_saved_decode_s']=None if d is None or r['first_motor_event_s'] is None else d-r['first_motor_event_s']
        early=tt<=.015+1e-12;r['first15ms']['decoded_times_s']=tt[early];r['first15ms']['decoded_controls']=logical[early][:,ai]
        early=start<=.015+1e-12;r['first15ms']['write_boundary_times_s']=start[early];r['first15ms']['written_controls']=iv['commands'][early][:,ai]
    for label,b,v in [('old_sensory5ms',oldbody,oldiv),('new_sensory1ms',body,iv)]:
        commands=v['commands'];dur=v['interval_end_s']-v['interval_start_s'];below=commands<.0001
        arms[label]['native_control_floor_evidence']=dict(native_minimum_control=.0001,raw_command_below_floor_actuator_windows=int(below.sum()),raw_command_below_floor_actuator_seconds=float((below*dur[:,None]).sum()),body_activation_samples_below_floor=int((b['activation']<.0001-1e-12).sum()),body_activation_samples_at_floor=int((abs(b['activation']-.0001)<=1e-12).sum()),note='Raw d.ctrl retains values below native actuator ctrlrange; these counts are raw-command floor eligibility, NOT measured internal clamp calls. Activation samples include native initial transient.')
    initial={k:np.array_equal(oldbody[k][0],body[k][0]) for k in ['qpos','qvel','activation','command']};assert all(initial.values())
    st=sensory['physical_time'];curr=sensory['current_nA'];np.testing.assert_allclose(st,np.arange(250)*.001,atol=1e-12,rtol=0);assert len(refresh)==250
    np.testing.assert_array_equal(curr[0],oldiv['sensory_current_nA'][0])
    held=np.repeat(oldiv['sensory_current_nA'],5,axis=0);diff=curr!=held;first_idx=np.flatnonzero(diff.any(1));first_idx=None if not len(first_idx) else int(first_idx[0])
    chfirst=None
    for i,record in enumerate(refresh):
        difference=first_channel_difference(record['channel_record'],oldrecords['records'][i//5]['channel_record'])
        if difference:chfirst=dict(physical_time_s=float(st[i]),old_held_source_time_s=float((i//5)*.005),changes=difference);break
    common=curr[::5]!=oldiv['sensory_current_nA'];sample_sources=[]
    for i,r in enumerate(refresh):
        assert abs(r['physical_time']-r['source_body_time'])<1e-12
        bid=int(round(r['source_body_time']/base.BODY_DT));np.testing.assert_array_equal(sensory['qpos'][i],body['qpos'][bid]);np.testing.assert_array_equal(sensory['qvel'][i],body['qvel'][bid]);sample_sources.append(bid)
    feedback=dict(initial_state_exact=initial,initial_current_exact=True,first_new_refresh_s=float(st[0]),refresh_count=len(st),first_current_difference_from_old_held_s=None if first_idx is None else float(st[first_idx]),first_different_current_local_indices=[] if first_idx is None else np.flatnonzero(diff[first_idx]).tolist(),first_channel_value_difference_from_old_held=chfirst,first_difference_at_common5ms_refresh_s=base.first(common.any(1),np.arange(50)*.005),first5ms_network_count_difference_start_s=base.first((iv['spikes']!=oldiv['spikes']).any(1),np.arange(50)*.005),actual_current_min=float(curr.min()),actual_current_max=float(curr.max()),input_clamp='NOT_IMPLEMENTED / N_A; encoder normalizes and scales without input clipping; do not report fictitious zero clamp saturation',completed_body_refresh_sources_verified=True,source_body_ticks=sample_sources,observations_and_channel_records_source=str(p/'SENSORY_REFRESH.json'),caveat='Sensory refresh is intervention; evolving body/current/network feedback differences do not isolate pure latency or establish biological adequacy.')
    report=dict(scope='S4 saved-only sensory1ms command5ms versus historical sensory5ms command5ms',execution=dict(model_loads=0,forward=0,body_steps=0,neural_steps=0,decoder_calls=0),sources_sha256=hashes,archive_checks=archive,arms=arms,feedback=feedback,normalization_note='Memory-only1ms hold expansion used for structural reuse; all command/readout/saturation values reported from true5ms blocks and logical decode from original ticks. No interpolation or decoder replay.',limitations=['New per-tick logical filter timestamp is NOT actual decoder call; actual completed-block decode/readout only at5ms boundaries.','Historical full-network first event only bounded by saved5ms window; historical subwindow logical decode not saved.','Native activation threshold .0001+1e-12, clipping saturation threshold1-1e-12.','Last50ms uses(200,250]ms4000savedstates; contact presence is not load bearing.','Model force and absolute calibration NOT_ESTABLISHED;16mappinggaps/candidate prohibition/animal reference limits remain; no standing or walking claim.'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,2,figsize=(16,17))
    for label,b in [('old_sensory5ms',oldbody),('new_sensory1ms',body)]:
        t=b['time']*1000;world=worlds[label]
        series=[(t,b['command'].max(1)),(t,b['activation'].max(1)),(t,b['thorax_position'][:,2]),(t,b['com_position'][:,2]),(t,b['com_velocity'][:,2]),(t,b['nonfoot_floor_count']),(t,np.count_nonzero(b['foot_contact'],axis=1)),(t[-4000:],world[:,6,2])]
        for ax,(x,y) in zip(axs.flat,series):ax.plot(x,y,label=label)
        axs[3,1].plot(t[-4000:],world[:,:6,2].sum(1),'--',label=label+' feet')
    titles=['Actual command maximum / command5ms','Native activation maximum','Thorax height / model units','COM height / model units','COM vz / model units','Nonfoot floor contacts','Foot contact legs (true floor ledger in JSON)','Last50ms vertical reaction / model units']
    for ax,title in zip(axs.flat,titles):ax.set_title(title);ax.set_xlabel('physical time / ms');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Sensory1ms / command5ms versus historical sensory5ms / command5ms — NOT standing certification');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(DESKTOP/(STEM+'.png'),dpi=140);plt.close(fig)
    # Explicit first15ms chain and sensory sampling panels, distinct actual/logical times.
    fig,axs=plt.subplots(2,2,figsize=(17,11))
    markers=[('first_motor_event_s','Motor event','o'),('first_saved_decode_s','Logical filter sample','s'),('first_effective_motor_readout_s','Actual block decode','D'),('first_write_boundary_s','Write boundary','v'),('first_body_command_sample_s','BODY command sample','>'),('first_activation_above_native_floor_s','Activation above floor','*')]
    for ax,(label,arm) in zip(axs[0],arms.items()):
        for j,leg in enumerate(LEGS):
            r=arm['perleg'][leg]
            for z,(key,title,marker) in enumerate(markers):
                if label=='old_sensory5ms' and key=='first_saved_decode_s':continue
                value=r[key]
                if value is not None:ax.scatter(value*1000,j+(z-2.5)*.065,marker=marker,label=title if j==0 else None,s=48)
        ax.set_yticks(range(6),[s.upper() for s in LEGS]);ax.invert_yaxis();ax.set_xlim(0,16);ax.set_xlabel('physical / neural time ms');ax.set_title(label+' event → completed decode → write → activation');ax.grid(alpha=.2);ax.legend(fontsize=7)
    times=np.arange(250)*.001;oldchannel=[r['channel_record']['per_leg']['lf']['hook_direction']['value'] for r in oldrecords['records']]
    newchannel=[r['channel_record']['per_leg']['lf']['hook_direction']['value'] for r in refresh]
    axs[1,0].step(np.arange(50)*5,oldchannel,where='post',label='old held5ms LF hook_direction');axs[1,0].step(times*1000,newchannel,where='post',label='new refreshed1ms LF hook_direction');axs[1,0].set_xlim(0,15);axs[1,0].set_title('Actual saved sensory channel samples (not interpolated)');axs[1,0].legend(fontsize=8)
    axs[1,1].step(times*1000,diff.sum(1),where='post');axs[1,1].set_xlim(0,15);axs[1,1].set_title('Current entries differing from historical held input');axs[1,1].set_ylabel('neural input entries')
    for ax in axs[1]:ax.set_xlabel('physical time ms');ax.grid(alpha=.2)
    fig.suptitle('Sensory1ms / command5ms — first15ms chain and observed refresh; logical ticks ≠ decoder-call times');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DESKTOP/(STEM+'_EARLY_CHAIN.png'),dpi=140);plt.close(fig)
    text=['# S4 sensory1ms / command5ms — saved-only comparison','', 'No dynamics, model loading, forward, neural simulation or decoder replay.','', '| Metric | Historical sensory5ms | New sensory1ms |','|---|---:|---:|']
    for k in ['total_network_events','unique_motor_node_events','first15ms_network_events','first15ms_unique_motor_events','first_saved_decode_s','first_effective_motor_readout_s','first_write_boundary_s','first_actual_body_command_sample_s','first_activation_above_native_floor_s','first_nonfoot_s']:
        text.append('| '+k+' | '+' | '.join(str(arms[a][k]) for a in arms)+' |')
    text+=['','## First all-network event (historical bound only)',json.dumps({label:arm['all_network_first_event'] for label,arm in arms.items()},indent=2),'','Logical sample times are not actual completed-block decoder-call times; both runs command5ms.','', '## Sensory first difference',json.dumps(feedback,indent=2,default=base.jsonable),'', '## Per-leg event → logical filter → actual decode → write → activation','| arm/leg | Event | Logical | Actual decode | Write | BODY sample | Above-floor activation |','|---|---:|---:|---:|---:|---:|---:|']
    for label,arm in arms.items():
        for leg,r in arm['perleg'].items():text.append('| '+label+'/'+leg+' | '+' | '.join(str(r[k]) for k in ['first_motor_event_s','first_saved_decode_s','first_effective_motor_readout_s','first_write_boundary_s','first_body_command_sample_s','first_activation_above_native_floor_s'])+' |')
    text+=['','Full first15ms per-leg event lists, decoded/written vectors and native-floor activation counts/peaks are in ANALYSIS.json.','', '## Saturation and validity']
    for label,arm in arms.items():text.append(label+': '+json.dumps({k:arm[k] for k in ['saturation_applied','native_control_floor_evidence','warning_max','nonfinite','activation_min','activation_max','activation_over1']},default=base.jsonable))
    text+=['','## Last50ms (200,250]ms — 4000 states; model force units']
    for label,arm in arms.items():
        last=arm['last50ms'];text+=['', '### '+label,'| Category | Contact samples | Mean world reaction xyz |','|---|---:|---|']
        for j,leg in enumerate(LEGS+['NONFOOT']):text.append(f"| {leg} | {last['geometric_contact_samples'][j]} | {last['world_reaction_xyz_stats']['mean'][j].tolist()} |")
        text+=[last['world_reaction_source'],json.dumps({k:last[k] for k in ['thorax_height','com_height','com_vz']},default=base.jsonable)]
    text+=['','## Limits']+report['limitations']+['','## Sources / outputs',str(ROOT/'ANALYSIS.json'),str(DESKTOP/(STEM+'.json')),str(DESKTOP/(STEM+'.png')),str(DESKTOP/(STEM+'_EARLY_CHAIN.png')),str(Path(__file__))]
    assert all(base.sha(f)==hashes[str(f)] for f in source),'Input changed during saved-only analysis'
    encoded=json.dumps(report,indent=2,default=base.jsonable,allow_nan=False)+'\n';(ROOT/'ANALYSIS.json').write_text(encoded);(DESKTOP/(STEM+'.json')).write_text(encoded);(DESKTOP/(STEM+'.md')).write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(exit_code=0,outputs=[str(ROOT/'ANALYSIS.json'),str(DESKTOP/(STEM+'.md')),str(DESKTOP/(STEM+'.png'))],execution=report['execution'])))
if __name__=='__main__':main()
