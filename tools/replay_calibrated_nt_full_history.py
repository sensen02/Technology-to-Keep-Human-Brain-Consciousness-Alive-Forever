"""Fixed 1/32 sourced-NT fresh full history: two arms, no body/tier/training.
run -> separate process verify -> parent review -> seal once.
"""
from pathlib import Path
import argparse, dataclasses, json, math, os, subprocess, sys, traceback
import numpy as np
from scipy import sparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_sourced_nt_increment as cal
old=cal.old
R=old.R; B=old.B; G=old.G
O=Path('/home/sensen/Desktop/engine修复验收/s4-calibrated-nt-full-history/20261102T023600Z')
H=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-full-prewarm-replay/20261101T214958Z')
C=cal.O
sha,load,arrays,dump,same=old.sha,old.load,old.arrays,old.dump,old.same
ARMS=['CANDIDATE','SELECTED_CURRENT_CUT']; GAIN=1/32
LIMITS='Fixed saved original-body sensory replay, not new physical closed loop. Physiological sign/calibration, physical support and walking NOT_ESTABLISHED. No gain reselection even if saturation fails.'

def context():
    c=old.context()
    records=load(B/'RUN/PREWARM_NEURAL_RECORDS.json')+load(B/'RUN/NEURAL_RECORDS.json')[:150]
    assert len(records)==1250
    drive=np.array([r['sensory_current_nA'] for r in records])
    raster=arrays(B/'NEURAL_RASTER.npz')['spikes'][:2500]
    assert drive.shape==(1250,12000) and raster.shape==(2500,12000)
    return c,records,drive,raster

def decode(e,mapping):
    rate=np.zeros(90); half=[]; pool=[]
    decay=math.exp(-.0005/.005)
    for t,sp in enumerate(e):
        inst=np.array([sp[r['local_indices']].mean()/.0005 if r['local_indices'] else 0. for r in mapping['rows']])
        rate=decay*rate+(1-decay)*inst;half.append(rate.copy())
        if t%2:pool.append([e[t-1:t+1,r['local_indices']].sum() for r in mapping['rows']])
    half=np.array(half);nxt=np.clip(half[1::2]/100,0,1)
    return dict(rates_halfstep_hz=half,rates_window_hz=half[1::2],pool_counts=np.array(pool),decoded_next=nxt,applied=np.vstack([np.zeros(90),nxt[:-1]]))

def accumulation(e,mats, checkpoints):
    ring=np.zeros((5,12000),bool);pos=0;ge=np.zeros(12000);gi=ge.copy();ad=ge.copy();ref=ge.copy();count=np.zeros(12000,np.int64);out={}
    for t,sp in enumerate(e):
        arrival=ring[pos].copy()
        ge+=mats['We']@arrival;gi+=mats['Wi']@arrival
        ge*=np.exp(-.5/3);gi*=np.exp(-.5/8)
        ad*=np.exp(-.5/100);ad[sp]+=.002
        ref[sp]=(t+1)*.5+2;count+=sp
        ring[pos]=sp;pos=(pos+1)%5
        if t+1 in checkpoints:out[t+1]=dict(_ring=ring.copy(),ring_i=np.array(pos),ge=ge.copy(),gi=gi.copy(),adaptation=ad.copy(),refractory_until_ms=ref.copy(),spike_count=count.copy())
    return out

def baseline_gate(c,records,drive,e):
    d=decode(e,c[3]);n=arrays(B/'RUN/NEURAL.npz')
    assert same(e[2200:].reshape(150,2,12000).sum(1),n['spike_counts'][:150])
    assert same(drive[1100:],n['sensory_current_nA'][:150])
    for key,field,inside in [('applied','applied_command',False),('decoded_next','decoded_next_command',False),('rates_window_hz','rates_hz',True),('pool_counts','pool_spike_counts',True)]:
        ref=np.array([r['decoder'][field] if inside else r[field] for r in records]);assert same(d[key],ref),key
    counts=e.reshape(1250,2,12000).sum(1)
    assert same(counts.sum(1),np.array([r['total_network_spikes'] for r in records]))
    assert same(counts[:,c[2]].sum(1),np.array([r['motor_spikes'] for r in records]))
    acc=accumulation(e,c[6],[2200])[2200];boundary=c[4]
    for k,v in acc.items():
        if k!='ring_i':assert same(v,boundary[k]),k
    assert same(d['rates_window_hz'][1099],boundary['decoder_rates']) and same(d['decoded_next'][1099],boundary['pending'])
    assert int(c[5]['net_scalars']['_ring_i'])==int(acc['ring_i'])==0
    # Reconstruct every current from original root888 channel records.
    rebuilt=np.zeros_like(drive)
    for leg,channels in load(B/'SENSORY_GROUPS.json').items():
        for channel,idx in channels.items():
            for w,r in enumerate(records):
                a=r['channels']['per_leg'][leg][channel]
                rebuilt[w,idx]=r['channels']['scale_nA_per_unit']*float(a['value'])*a['weight']
    assert same(rebuilt,drive)
    return d,dict(passed=True,identity='round13 true root888, never OLD root_id',raster_slice=[0,2500],prewarm_records=1100,release_records=150,all1250_decoder_and_counts_exact=True,root888_current_rebuilt_exact=True,boundary_ring_ge_gi_adaptation_refractory_counts_exact=True,v_evidence='Saved boundary v only; no original per-tick v trace, no baseline voltage integration/reproduction claim',baseline_networks=0,baseline_ticks=0,old_gate_role='Historical old-interface gate only, not new baseline identity')

def prepare():
    assert not O.exists(),'Refuse overwrite'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()=='6175f81be0363b97455ec08bc43c2a2a0348f58f'
    c,rec,drive,e=context();d,gate=baseline_gate(c,rec,drive,e)
    assert load(C/'SELECTION_TABLE.json')['selected_gain']==GAIN
    hashes={}
    for folder in [B,G,C,H]:hashes.update(cal.check_manifest(folder))
    for p in [Path(__file__),Path(old.__file__),Path(cal.__file__),R/'engine/neural_cond.py',R/'engine/embodied/motor_output.py']:
        hashes[str(p)]=sha(p)
    O.mkdir(parents=True)
    dump(O/'PREREGISTRATION.json',dict(stage='S3/S4',gain=GAIN,arms=ARMS,selection_table=str(C/'SELECTION_TABLE.json'),source_sha256=hashes,matrix='actual baseWe + 1/32 ADDED_We; Wi unchanged',fresh='ConductanceNetwork and MotorOutput fresh defaults neural t0, zero offsets, zero ring, zero pending; no old current-mask fix',schedule=dict(neural_start_absolute_body_s=.2,prewarm_ms=1100,release_ms=150,ticks_per_arm=2500,current_windows_per_arm=1250),cut='Selected963 current zero from neural t0; other currents identical, no edge cut or ring inheritance',planned_calls=dict(network_instances=2,neural_steps=5000,body_model=0,body_forward=0,body_steps=0,tier_build=0,training=0),limits=LIMITS,baseline_gate=gate))
    dump(O/'BASELINE_REUSE_GATE.json',gate)
    dump(O/'OLD_PREWARM_GATE_REFERENCE.json',dict(path=str(H/'OLD_PREWARM_GATE.json'),sha256=sha(H/'OLD_PREWARM_GATE.json'),content=load(H/'OLD_PREWARM_GATE.json'),identity='OLD root_id interface, historical only'))
    dump(O/'MAPPING.json',c[3]);dump(O/'SENSORY_GROUPS.json',load(B/'SENSORY_GROUPS.json'))
    np.savez_compressed(O/'BASELINE_REUSED.npz',spikes_packed=np.packbits(e,axis=1),counts=e.reshape(1250,2,12000).sum(1),motor_events=e[:,c[2]],**d)
    cut=drive.copy();cut[:,c[1]]=0
    np.savez_compressed(O/'INPUTS.npz',saved_current_nA=drive,cut_current_nA=cut,root_ids=c[0],selected_local=c[1],motor_local=c[2],selected_mask=np.isin(np.arange(12000),c[1]),ownership_mask=np.ones(12000,bool))
    for p in [Path(__file__),Path(old.__file__),Path(cal.__file__),R/'engine/neural_cond.py',R/'engine/embodied/motor_output.py']:
        (O/('EXECUTED_'+p.name)).write_bytes(p.read_bytes())
    return c,drive

def simulate(arm,c,drive,calls):
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.motor_output import MotorOutput
    net=ConductanceNetwork(12000,ConductanceParams(),dt_ms=.5);calls['network_instances']+=1
    mats=cal.matrix(GAIN,c);net.set_connectivity(mats['We'],mats['Wi'])
    dec=MotorOutput(c[3]);pending=np.zeros(90)
    for k in mats:sparse.save_npz(O/(arm+'_ACTUAL_'+k+'.npz'),getattr(net,k))
    dump(O/(arm+'_NETWORK_IDENTITY.json'),dict(params=dataclasses.asdict(net.p),n=net.n,dt_ms=net.dt,delay_steps=net.delay_steps,ring_slots=len(net._ring),voltage_method=net.voltage_method,decoder_config=dataclasses.asdict(dec.config),root_sha256=sha(B/'IDENTITIES.npz'),offset0=True,ownership='all default enabled'))
    np.savez_compressed(O/(arm+'_INITIAL.npz'),**old.snapshot(net,dec,pending))
    ev=np.zeros((2500,12000),bool);arrival=np.zeros_like(ev)
    inc=np.zeros((2500,12000));cnt=np.zeros((2500,12000),np.int32)
    half=np.zeros((2500,90));app=np.zeros((1250,90));nxt=app.copy();pool=np.zeros((1250,90),np.int64)
    stats={k:np.zeros((1250,4)) for k in ['v','ge','gi','adaptation']}
    delta=GAIN*c[7]['We'];positive=c[7]['We'].astype(bool).astype(np.int32)
    current=drive.copy()
    if arm==ARMS[1]:current[:,c[1]]=0
    for w in range(1250):
        app[w]=pending;rate=dec.rates.copy();samples={k:[] for k in stats}
        for sub in range(2):
            t=2*w+sub;arrival[t]=net._ring[net._ring_i]
            inc[t]=delta@arrival[t];cnt[t]=positive@arrival[t].astype(np.int32)
            ev[t]=net.step(current[w]);calls['neural_steps']+=1
            for k in stats:
                a=getattr(net,k);assert np.isfinite(a).all();samples[k].append(a.copy())
            inst=np.array([ev[t,g].mean()/.0005 if g.size else 0. for g in dec.groups]);rate=math.exp(-.1)*rate+(1-math.exp(-.1))*inst;half[t]=rate
        pending,tr=dec.consume(ev[2*w:2*w+2],spike_dt_s=.0005,start_s=w*.001,end_s=(w+1)*.001)
        assert same(rate,dec.rates);nxt[w]=pending;pool[w]=tr['pool_spike_counts']
        for k in stats:
            a=np.array(samples[k]);stats[k][w]=[a.min(),a.max(),a.mean(),np.sqrt((a*a).mean())]
        if w==1099:np.savez_compressed(O/(arm+'_BOUNDARY.npz'),**old.snapshot(net,dec,pending))
        if (w+1)%100==0:print(arm,w+1,'windows',flush=True);dump(O/'ACTUAL_CALLS.json',calls)
    np.savez_compressed(O/(arm+'_FINAL.npz'),**old.snapshot(net,dec,pending))
    np.savez_compressed(O/(arm+'_NEURAL.npz'),spikes_packed=np.packbits(ev,axis=1),arrivals_packed=np.packbits(arrival,axis=1),counts=ev.reshape(1250,2,12000).sum(1),motor_events=ev[:,c[2]],selected_events=ev[:,c[1]],selected_arrivals=arrival[:,c[1]],added_arrival_increment_uS=inc,added_arrival_event_counts=cnt,aggregate_event_counts=cnt.sum(1),**{k+'_window_min_max_mean_rms':v for k,v in stats.items()})
    np.savez_compressed(O/(arm+'_DECODE.npz'),rates_halfstep_hz=half,rates_window_hz=half[1::2],applied=app,decoded_next=nxt,pool_counts=pool,applied_source_interval=np.arange(-1,1249),decoded_source_interval=np.arange(1250))
    dump(O/'ACTUAL_CALLS.json',calls)

def unpack(n):return np.unpackbits(n['spikes_packed'],axis=1,count=12000).astype(bool)
def metrics():
    inp=arrays(O/'INPUTS.npz');motor=inp['motor_local'];base=arrays(O/'BASELINE_REUSED.npz');out={}
    for arm in ['BASELINE']+ARMS:
        n=base if arm=='BASELINE' else arrays(O/(arm+'_NEURAL.npz'));d=base if arm=='BASELINE' else arrays(O/(arm+'_DECODE.npz'));e=unpack(n)
        rows={}
        for name,a,b in [('prewarm',0,1100)]+[('release'+str(ms),1100,1100+ms) for ms in [5,15,50,150]]:
            s=e[2*a:2*b,motor];rows[name]=dict(MN_spikes=int(s.sum()),active_MN=int(s.any(0).sum()),network_spikes=int(e[2*a:2*b].sum()),applied_saturated=int((d['applied'][a:b]>=1).sum()),applied_elements=(b-a)*90)
        rows['differences_vs_original']={}
        for name,a,b in [('full',0,1250),('prewarm',0,1100),('release',1100,1250)]:
            rows['differences_vs_original'][name]=dict(first_MN_event_end_ms=old.first(e[2*a:2*b,motor],unpack(base)[2*a:2*b,motor],.5),first_applied_start_ms=old.first(d['applied'][a:b],base['applied'][a:b],1,-1),first_decoded_available_ms=old.first(d['decoded_next'][a:b],base['decoded_next'][a:b],1),MN_different_bits=int((e[2*a:2*b,motor]!=unpack(base)[2*a:2*b,motor]).sum()))
        rows['release_initial_pending_diff']=bool(np.any(d['applied'][1100]!=base['applied'][1100]))
        rows['release_first20ms_command_different_elements']=int((d['applied'][1100:1120]!=base['applied'][1100:1120]).sum())
        rows['original_first_nonfoot_ms']=18.9125
        out[arm]=rows
    return out

def verify():
    pr=load(O/'PREREGISTRATION.json')
    for p,h in pr['source_sha256'].items():assert sha(p)==h,p
    c,rec,drive,be=context();bd,gate=baseline_gate(c,rec,drive,be);inp=arrays(O/'INPUTS.npz')
    assert same(inp['saved_current_nA'],drive)
    cut=drive.copy();cut[:,c[1]]=0;assert same(cut,inp['cut_current_nA'])
    assert same(cut[:,~inp['selected_mask']],drive[:,~inp['selected_mask']])
    raw=cal.raw_graph(c);delta=raw.data*GAIN;audits={}
    for arm in ARMS:
        n=arrays(O/(arm+'_NEURAL.npz'));d=arrays(O/(arm+'_DECODE.npz'));e=unpack(n);ar=np.unpackbits(n['arrivals_packed'],axis=1,count=12000).astype(bool)
        mats={k:sparse.load_npz(O/(arm+'_ACTUAL_'+k+'.npz')) for k in ['We','Wi']}
        for k,v in cal.matrix(GAIN,c).items():assert old.matrix_exact(v,mats[k])
        init=arrays(O/(arm+'_INITIAL.npz'))
        for k in ['ge','gi','adaptation','refractory_until_ms','spike_count','_ring','threshold_offset_mV','decoder_rates','pending']:assert not np.any(init[k]),k
        assert np.all(init['v']==-65) and int(init['steps'])==0 and float(init['t_ms'])==0 and int(init['decoder_interval'])==-1
        dd=decode(e,c[3])
        for k,v in dd.items():assert same(v,d[k]),(arm,k)
        assert same(n['counts'],e.reshape(1250,2,12000).sum(1));assert same(n['motor_events'],e[:,c[2]]) and same(n['selected_events'],e[:,c[1]])
        ring=np.zeros((5,12000),bool);pos=0
        for t in range(2500):
            assert same(ar[t],ring[pos]);assert same(n['selected_arrivals'][t],ar[t,c[1]])
            active=ar[t,raw.col];counts=np.bincount(raw.row[active],minlength=12000);inc=np.bincount(raw.row[active],weights=delta[active],minlength=12000)
            assert same(counts,n['added_arrival_event_counts'][t]);assert same(inc,n['added_arrival_increment_uS'][t]);assert counts.sum()==n['aggregate_event_counts'][t]
            ring[pos]=e[t];pos=(pos+1)%5
        states=accumulation(e,mats,[2200,2500])
        for ticks,suffix in [(2200,'BOUNDARY'),(2500,'FINAL')]:
            state=arrays(O/(arm+'_'+suffix+'.npz'))
            for k,v in states[ticks].items():assert same(v,state[k]),(arm,suffix,k)
            assert same(state['pending'],dd['decoded_next'][ticks//2-1]) and same(state['decoder_rates'],dd['rates_halfstep_hz'][ticks-1])
            assert int(state['steps'])==ticks and float(state['t_ms'])==ticks*.5 and int(state['decoder_interval'])==ticks//2-1
            for k in ['v','ge','gi','adaptation']:assert np.isfinite(state[k]).all()
        assert same(d['applied_source_interval'],np.arange(-1,1249)) and same(d['decoded_source_interval'],np.arange(1250))
        if arm==ARMS[1]:assert not e[:,c[1]].any() and not ar[:,c[1]].any() and not n['aggregate_event_counts'].any()
        audits[arm]=dict(ticks=2500,pure_arrays=True,decoder_every_halfstep_exact=True,ring_delay_counts_actual_matrix_initial_boundary_final_exact=True,voltage_limit='No voltage reintegration; stored initial/boundary/final finite, window statistics retained')
    result=metrics();dump(O/'RESULT.json',dict(metrics=result,limits=LIMITS));dump(O/'INDEPENDENT_VERIFICATION.json',dict(passed=True,baseline=gate,arms=audits,zero_networks=True,zero_integration=True))
    dump(O/'JOB_LEDGER.json',dict(run='completed; see ACTUAL_CALLS',verify='fresh process pure arrays completed',failures=[],parent_review='required before seal'))
    text='# 唯一1/32新增边完整神经历史重放\n\n'+LIMITS+'\n\n第13真实新root888基准使用raster[0:2500]、1100条PREWARM和150条RUN记录；旧OLD gate仅旧接口历史证据。全窗counts/decoder与1100ms ring/ge/gi/adapt/ref/累计spikes逐值一致。原缺逐tick v，未声称基准电压重现。\n\n两臂各fresh2500ticks，神经t0对应原身体0.2s；1.1s交接无reset。selected963从t0电流为0对照没有旧ring残留。感觉仍是原身体固定序列，差异不能直接外推新现实闭环因果。\n\n统计如下（firstdiff为对应phase相对时间；事件取tick末，applied取窗口开始）：\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n全部5000ticks由独立新进程数组复算raster/counts/ring/delay/filter/applied/边界累计和实际矩阵，未重积分基准或候选。保存完整packed raster、逐窗12000counts、204MN事件、90pool逐半步rates、selected arrivals、新边增量及事件计数、全量初/边界/末状态与每窗状态统计。物理支撑与生理sign仍NOT_ESTABLISHED；饱和与弱效应如实保留，不重选。\n'
    (O/'REPORT.zh-CN.md').write_text(text)
    print(json.dumps(result,ensure_ascii=False),flush=True)

def seal():
    assert not (O/'MANIFEST_SHA256.json').exists()
    assert load(O/'PARENT_REVIEW.json')['passed'] and load(O/'INDEPENDENT_VERIFICATION.json')['passed']
    dump(O/'CLEANUP_LEDGER.json',dict(deleted=[],reason='All created outputs retained for evidence or provenance; no confirmed disposable files'))
    inventory={str(p):dict(importance='核心必需' if p.suffix in ['.npz','.py'] or p.name in ['PREREGISTRATION.json','INDEPENDENT_VERIFICATION.json'] else '重要辅助',purpose='Scientific arrays/executed source' if p.suffix in ['.npz','.py'] else 'Identity, report, provenance or review') for p in O.iterdir() if p.is_file()}
    dump(O/'FILE_INVENTORY.json',inventory)
    dump(O/'MANIFEST_SHA256.json',{str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in O.iterdir() if p.is_file()})
    for p in O.iterdir():p.chmod(0o444)
    O.chmod(0o555)
    print('sealed once',O,flush=True)

if __name__=='__main__':
    mode=argparse.ArgumentParser();mode.add_argument('mode',choices=['run','verify','seal']);a=mode.parse_args()
    if a.mode=='run':
        c,drive=prepare();calls=cal.calls_zero();dump(O/'ACTUAL_CALLS.json',calls)
        try:
            for arm in ARMS:simulate(arm,c,drive,calls)
        except BaseException:
            dump(O/'FAILURE.json',dict(traceback=traceback.format_exc(),calls=calls));raise
    elif a.mode=='verify':verify()
    else:seal()
