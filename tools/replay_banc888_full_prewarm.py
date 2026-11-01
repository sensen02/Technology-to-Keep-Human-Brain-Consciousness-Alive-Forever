"""Full-history fixed-body-input replay, separate from sealed boundary A/B experiment.
Old prewarm must reproduce every saved neural boundary array/rates/pending exactly.
Only then run C with corrected sensory and MN interfaces from neural time zero.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'tools'))
from replay_banc888_interface_candidates import arrays, dump, load, sha, inputs, SRC, ARCHIVE, CAND
AB=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-interface-replay/20261101T214247Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s4-banc888-full-prewarm-replay/20261101T214958Z')
KEYS=['threshold_offset_mV','_ring','v','ge','gi','adaptation','refractory_until_ms','spike_count']


def unchanged():
    pr=load(OUT/'PREREGISTRATION.json')
    for p,h in pr['source_sha256'].items():assert sha(p)==h,p
    return pr


def prepare():
    assert not OUT.exists()
    roots,old,new,oldstat,newstat,ma,mc=inputs()
    warm=load(SRC/'PREWARM_NEURAL_RECORDS.json');release=load(SRC/'NEURAL_RECORDS.json')
    cfg=load(SRC/'PREREGISTRATION.json');identity=load(ARCHIVE/'IDENTITY.json')
    assert len(warm)==1100 and len(release)==700
    assert cfg['neural_prewarm_s']==1.1 and cfg['neural_command_dt_s']==.001 and cfg['neural_dt_s']==.0005
    assert cfg['neural_sensory_channel_weights'] is None and cfg['neural_descending_config'] is None
    assert sha(SRC/'run_physical_walk_prewarm.py')==cfg['source_hashes'][str(REPO/'run_physical_walk_prewarm.py')]
    for p,h in cfg['source_hashes'].items():
        if p.startswith(str(REPO/'engine')):assert sha(p)==h,p
    for p,x in load(AB/'MANIFEST_SHA256.json')['files'].items():assert sha(p)==x['sha256'],p
    for p,h in identity['input_sha256'].items():assert sha(p)==h,p
    boundary=arrays(SRC/'NEURAL_BOUNDARY.npz')
    assert np.all(boundary['threshold_offset_mV']==0)
    assert np.all(np.array(warm[0]['applied_command'])==0)
    current_old=np.array([r['sensory_current_nA'] for r in warm+release]);assert current_old.shape==(1800,12000)
    assert np.array_equal(current_old[1100:],arrays(SRC/'NEURAL.npz')['sensory_current_nA'])
    rebuilt=np.zeros_like(current_old);corrected=np.zeros_like(current_old)
    scalars=[];groupnames=[];group_proof=[]
    for leg,chs in old.items():
        for ch,idx in chs.items():
            v=np.array([r['channels']['scale_nA_per_unit']*float(r['channels']['per_leg'][leg][ch]['value'])*r['channels']['per_leg'][leg][ch]['weight'] for r in warm+release])
            assert np.array_equal(current_old[:,idx],np.broadcast_to(v[:,None],(1800,len(idx))))
            if ch in ('club_vibration','multidendritic_noxious'):assert np.all(v==0)
            rebuilt[:,idx]=v[:,None];corrected[:,new[leg][ch]]=v[:,None]
            groupnames.append(leg+'/'+ch);scalars.append(v)
            group_proof.append(dict(leg=leg,channel=ch,old_n=len(idx),new_n=len(new[leg][ch]),all1800_old_members_same=True,exact_encoder_scalar=True))
    assert np.array_equal(current_old,rebuilt)
    oldowners={i:(l,c) for l,cs in old.items() for c,ids in cs.items() for i in ids}
    newowners={i:(l,c) for l,cs in new.items() for c,ids in cs.items() for i in ids}
    assert all(oldowners[i]==newowners[i] for i in set(oldowners)&set(newowners))
    added=sorted(set(newowners)-set(oldowners));assert len(added)==26 and not set(oldowners)-set(newowners)
    paths=list(load(AB/'PREREGISTRATION.json')['source_sha256'])
    paths += [str(AB/n) for n in ['MANIFEST_SHA256.json','PREREGISTRATION.json','BASELINE_DECODE.npz','A_DECODE.npz','B_DECODE.npz','A_NEURAL.npz','MAPPING_B.json']]
    paths += [str(SRC/n) for n in ['run_physical_walk_prewarm.py','WARMED_RUNTIME.npz','PREWARM_NEURAL_RECORDS.json','PREWITHDRAW_NEURAL_STATE.npz','NEURAL_BOUNDARY.npz','EXECUTION.json']]
    paths += [str(Path(__file__).resolve())]
    hashes={p:sha(p) for p in dict.fromkeys(paths)}
    OUT.mkdir(parents=True)
    dump(OUT/'PREREGISTRATION.json',dict(scope='Separate full-history fixed-body-sensory-input neural replay C; no new physical feedback.',
        source_sha256=hashes,repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        prior_sealed_AB=str(AB),prior_sealed_AB_unchanged=True,
        runner_evidence=dict(path=str(SRC/'run_physical_walk_prewarm.py'),sha256=sha(SRC/'run_physical_walk_prewarm.py'),
          source_lines='156 seed0/build zero background;179 run fresh runtime;180-183 save warmed state;186 continue same runtime',
          implementation_lines='continuous_handover24-35: default ConductanceParams,WeWi*.2,default MotorOutput;83-85: fresh runtime,pending0;97-100: saved boundary',
          old_prewarm_windows=1100,neural_dt_ms=.5,window_ms=1.,warmup_neural_steps=2200,neural_time_start_s=0.,body_time_start_s=float(warm[0]['absolute_start_s']),
          neural_boundary_s=1.1,body_boundary_s=float(release[0]['absolute_start_s']),background_current=0.,threshold_offsets_zero=True),
        old_gate='Start default resting neural state and zero decoder/pending, integrate saved1100 full currents. Every window compare old applied,next,rates,pool counts,total/MN spikes; at2200 compare all8 neural arrays,90rates,pending to NEURAL_BOUNDARY and PREWITHDRAW exactly. Source-derived clock steps2200,t1100ms,ring0 and WARMED_RUNTIME must match. If ANY mismatch save first difference and stop C; no tuning/retry.',
        C='Fresh default network+204MN decoder, corrected root888 sensory throughout saved1100 prewarm+700release windows. Save own2200-step boundary/rates/pending and continue SAME net+decoder into release; no reset/cold insertion. This is full-history candidate and changes boundary relative to old A/B.',
        input_rule='For all1800 windows use historical saved scale*float(channel normalized value)*weight; exact reconstruct old full current and all group members, validated disjoint groups, unmodelled zeros unchanged.',
        groups=group_proof,identity=identity,decoder=dict(filter_tau_s=.005,rate_for_full_control_hz=100.,max_control=1.,per_actuator_gain=None),
        selected_MN_old=202,selected_MN_C=204,mapped_actuators=74,unresolved=16,distinct_shared_pools=66,
        comparisons='C release early15ms/full700ms vs authenticated oldbaseline and prior A/B; differing prewarm histories and boundary explicitly disclosed. All commands causal previous interval. Archive full3600x204 selectedMN events and filtered neuron rates.',
        planned_calls=dict(model_load=0,body_step=0,tier_build=0,training=0,network_instances=2,neural_steps_old_gate=2200,neural_steps_C=3600,total_neural_steps_if_gate_passes=5800),
        verification='Independent new process array/scalar re-encoding and filtering, boundary reconstruction from saved event counts, original saved states/pool traces, no second neural integration.',
        limitations='Fixed historical body sensory sequence; candidate commands not applied to a physical body. Not physical closed-loop behavior, support or walking. Force-unit calibration NOT_ESTABLISHED.'))
    dump(OUT/'MAPPING_OLD.json',ma);dump(OUT/'MAPPING_C.json',mc)
    dump(OUT/'INPUT_REMAP_PROOF.json',dict(groups=group_proof,all1800_full_old_currents_exact=True,group_targets_disjoint=True,
        added=[dict(root_id=int(roots[i]),local_index=i,leg=newowners[i][0],channel=newowners[i][1]) for i in added],removed=[],
        old_targets=len(oldowners),new_targets=len(newowners),old_nonzero_any=int(np.any(current_old!=0,axis=0).sum()),new_nonzero_any=int(np.any(corrected!=0,axis=0).sum())))
    np.savez_compressed(OUT/'INPUTS.npz',old_current_nA=current_old,candidate_current_nA=corrected,group_names=np.array(groupnames),group_current_nA=np.array(scalars).T,
        root_ids=roots,added_indices=added,delta_nA=(corrected-current_old)[:,added],
        old_motor_indices=sorted({i for r in ma['rows'] for i in r['local_indices']}),candidate_motor_indices=sorted({i for r in mc['rows'] for i in r['local_indices']}),
        source_absolute_start_s=[r['absolute_start_s'] for r in warm+release],source_absolute_end_s=[r['absolute_end_s'] for r in warm+release])
    with (OUT/'INPUT_DELTA.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['full_window','phase','root_id','local_index','leg','channel','old_nA','new_nA','delta_nA','signed_dose_nA_s','abs_dose_nA_s'])
        for k in range(1800):
            for i in added:
                d=corrected[k,i]-current_old[k,i];leg,ch=newowners[i]
                w.writerow([k,'prewarm' if k<1100 else 'release',int(roots[i]),i,leg,ch,current_old[k,i],corrected[k,i],d,d*.001,abs(d)*.001])
    print(json.dumps(dict(prepared=str(OUT),all1800_currents_exact=True,planned_steps=5800),ensure_ascii=False))


def snapshot(net,dec,pending):
    return dict(**{k:getattr(net,k).copy() for k in KEYS},decoder_rates=dec.rates.copy(),pending=pending.copy(),
                steps=np.array(net.steps),t_ms=np.array(net.t_ms),ring_i=np.array(net._ring_i),decoder_elapsed_s=np.array(dec.elapsed_s),decoder_interval=np.array(dec.interval))


def first_difference(actual,expected,field):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape:return dict(field=field,actual_shape=list(a.shape),expected_shape=list(b.shape))
    bad=np.argwhere(a!=b)
    if not len(bad):return None
    ix=tuple(bad[0]);return dict(field=field,differing_elements=len(bad),first_index=list(ix),actual=a[ix],expected=b[ix],max_abs_error=float(np.max(np.abs(a.astype(float)-b.astype(float)))))


def simulate(label,drive,mapping,expected=None):
    from scipy.sparse import load_npz
    from engine.neural_cond import ConductanceNetwork,ConductanceParams
    from engine.embodied.motor_output import MotorOutput,MotorOutputConfig
    pr=load(OUT/'PREREGISTRATION.json');net=ConductanceNetwork(12000,ConductanceParams(**pr['identity']['conductance_params']),dt_ms=.5)
    net.set_connectivity(load_npz(ARCHIVE/'ACTUAL_We.npz'),load_npz(ARCHIVE/'ACTUAL_Wi.npz'))
    dec=MotorOutput(mapping,MotorOutputConfig());pending=np.zeros(90);n=len(drive)
    motor=np.array(sorted({i for row in mapping['rows'] for i in row['local_indices']}));mnrate=np.zeros(len(motor));decay=math.exp(-.0005/.005)
    events=np.empty((2*n,12000),bool);rates=np.empty((2*n,90));mnrates=np.empty((2*n,len(motor)))
    applied=np.empty((n,90));decoded=np.empty_like(applied);poolcounts=np.empty((n,90),int);first=None
    start_s=[];end_s=[]
    for k,I in enumerate(drive):
        applied[k]=pending
        for sub in range(2):
            j=2*k+sub;events[j]=net.step(I);mnrate=decay*mnrate+(1-decay)*events[j,motor]/.0005;mnrates[j]=mnrate
            inst=np.array([events[j,idx].mean()/.0005 if idx.size else 0. for idx in dec.groups])
            rates[j]=decay*(dec.rates if sub==0 else rates[j-1])+(1-decay)*inst
        # Keep original phase arithmetic for exact saved source-clock alignment.
        start=k*.001 if k<1100 else 1.1+(k-1100)*.001
        end=(k+1)*.001 if k<1100 else 1.1+(k-1099)*.001
        pending,trace=dec.consume(events[2*k:2*k+2],spike_dt_s=.0005,start_s=start,end_s=end)
        assert np.array_equal(rates[2*k+1],dec.rates)
        decoded[k]=pending;poolcounts[k]=trace['pool_spike_counts'];start_s.append(start);end_s.append(end)
        if expected is not None and first is None:
            checks=[('applied',applied[k],expected[k]['applied_command']),('next',decoded[k],expected[k]['decoded_next_command']),
                    ('rates',dec.rates,expected[k]['decoder']['rates_hz']),('pool_counts',poolcounts[k],expected[k]['decoder']['pool_spike_counts']),
                    ('total_network_spikes',events[2*k:2*k+2].sum(),expected[k]['total_network_spikes']),
                    ('motor_spikes',events[2*k:2*k+2,motor].sum(),expected[k]['motor_spikes'])]
            for field,a,b in checks:
                diff=first_difference(a,b,field)
                if diff is not None:first=dict(window=k,**diff);break
        if k==1099:
            b=snapshot(net,dec,pending);b['selected_motor_indices']=motor;b['selected_motor_rates_hz']=mnrate.copy()
            np.savez_compressed(OUT/f'{label}_BOUNDARY.npz',**b)
    state=snapshot(net,dec,pending)
    np.savez_compressed(OUT/f'{label}_FINAL.npz',**state)
    np.savez_compressed(OUT/f'{label}_NEURAL.npz',spikes_packed=np.packbits(events,axis=1),counts=events.reshape(n,2,12000).sum(1),
            selected_motor_indices=motor,selected_motor_root_ids=arrays(OUT/'INPUTS.npz')['root_ids'][motor],selected_motor_events=events[:,motor],
            selected_motor_rates_halfstep_hz=mnrates,event_neural_end_s=(np.arange(2*n)+1)*.0005,
            event_body_aligned_end_s=.2+(np.arange(2*n)+1)*.0005)
    np.savez_compressed(OUT/f'{label}_DECODE.npz',rates_halfstep_hz=rates,rates_window_hz=rates[1::2],applied=applied,decoded_next=decoded,pool_counts=poolcounts,
            applied_source_interval=np.arange(n)-1,decoded_source_interval=np.arange(n),neural_start_s=start_s,neural_end_s=end_s,
            absolute_start_s=arrays(OUT/'INPUTS.npz')['source_absolute_start_s'][:n],absolute_end_s=arrays(OUT/'INPUTS.npz')['source_absolute_end_s'][:n],
            saturated_applied=applied>=1.,saturated_next=decoded>=1.)
    return first


def run():
    pr=unchanged();assert not (OUT/'OLD_NEURAL.npz').exists() and not (OUT/'RESULT.json').exists()
    inp=arrays(OUT/'INPUTS.npz');started=time.monotonic()
    diff=simulate('OLD',inp['old_current_nA'][:1100],load(OUT/'MAPPING_OLD.json'),load(SRC/'PREWARM_NEURAL_RECORDS.json'))
    b=arrays(OUT/'OLD_BOUNDARY.npz');checks={};first=diff
    for name in ['NEURAL_BOUNDARY.npz','PREWITHDRAW_NEURAL_STATE.npz']:
        expected=arrays(SRC/name)
        for field in KEYS+['decoder_rates','pending']:
            d=first_difference(b[field],expected[field],name+':'+field);checks[name+':'+field]=d is None
            if first is None and d is not None:first=d
    warm=arrays(SRC/'WARMED_RUNTIME.npz')
    for field,value in dict(steps=2200,t_ms=1100.,ring_i=0,decoder_elapsed_s=1.1,decoder_interval=1099).items():
        d=first_difference(b[field],value,field);checks[field]=d is None
        if first is None and d is not None:first=d
    checks['warmed_runtime_rates']=np.array_equal(b['decoder_rates'],warm['decoder_rates'])
    checks['warmed_runtime_pending']=np.array_equal(b['pending'],warm['pending'])
    checks['warmed_runtime_time']=float(warm['neural_time_s'])==1.1
    gate=first is None and all(checks.values());dump(OUT/'OLD_PREWARM_GATE.json',dict(passed=gate,first_mismatch=first,boundary_checks=checks,
          all1100_window_traces_exact=diff is None,neural_steps=2200,network_instances=1))
    if not gate:
        dump(OUT/'RESULT.json',dict(gate_passed=False,C_executed=False,calls=dict(model_load=0,body_step=0,tier_build=0,training=0,network_instances=1,neural_steps=2200),first_mismatch=first))
        unchanged();print('OLD PREWARM EXACT GATE FAILED; C NOT RUN');return 2
    simulate('C',inp['candidate_current_nA'],load(OUT/'MAPPING_C.json'))
    answer=metrics();answer.update(gate_passed=True,C_executed=True,C_complete_decoder_history=True,
            calls=dict(model_load=0,body_step=0,tier_build=0,training=0,network_instances=2,old_prewarm_neural_steps=2200,C_neural_steps=3600,total_neural_steps=5800),
            elapsed_s=time.monotonic()-started,interpretation='Fixed saved body sensory sequence; C owns full preceding history and its boundary differs from OLD/A/B. No new body closed loop.')
    unchanged();dump(OUT/'RESULT.json',answer);print(json.dumps(answer,ensure_ascii=False,indent=2));return 0


def metrics():
    inp=arrays(OUT/'INPUTS.npz');c=arrays(OUT/'C_NEURAL.npz');csp=np.unpackbits(c['spikes_packed'],axis=1,count=12000).astype(bool)[2200:]
    bsp=np.unpackbits(arrays(ARCHIVE/'BASELINE_REPLAY.npz')['spikes_packed'],axis=1,count=12000).astype(bool)
    asp=np.unpackbits(arrays(AB/'A_NEURAL.npz')['spikes_packed'],axis=1,count=12000).astype(bool)
    dec={n:arrays(AB/f'{n}_DECODE.npz') for n in ['BASELINE','A','B']};cd=arrays(OUT/'C_DECODE.npz');dec['C']={k:v[1100:] for k,v in cd.items() if v.shape[0]==1800}
    answer={}
    for win in [15,700]:
        m={}
        for name,sp in [('BASELINE',bsp),('A',asp),('B',asp),('C',csp)]:
            d=dec[name];selected=inp['old_motor_indices'] if name in ['BASELINE','A'] else inp['candidate_motor_indices']
            m[name]=dict(network_spikes=int(sp[:2*win].sum()),old202_MN_spikes=int(sp[:2*win,inp['old_motor_indices']].sum()),
                candidate204_MN_spikes=int(sp[:2*win,inp['candidate_motor_indices']].sum()),active_readout_MN_spikes=int(sp[:2*win,selected].sum()),
                applied_nonzero=int((d['applied'][:win]>0).sum()),next_nonzero=int((d['decoded_next'][:win]>0).sum()),
                applied_saturated=int((d['applied'][:win]>=1).sum()),next_saturated=int((d['decoded_next'][:win]>=1).sum()))
        for name in ['BASELINE','A','B']:
            m['C_minus_'+name]={k:m['C'][k]-m[name][k] for k in m['C']}
            m['C_vs_'+name]=dict(applied_different_elements=int(np.count_nonzero(dec['C']['applied'][:win]!=dec[name]['applied'][:win])),
                applied_max_abs_delta=float(np.abs(dec['C']['applied'][:win]-dec[name]['applied'][:win]).max()),
                next_max_abs_delta=float(np.abs(dec['C']['decoded_next'][:win]-dec[name]['decoded_next'][:win]).max()))
        answer[f'release_first_{win}ms']=m
    cb=arrays(OUT/'C_BOUNDARY.npz');ob=arrays(OUT/'OLD_BOUNDARY.npz')
    answer['C_vs_old_boundary']={k:dict(differing_elements=int(np.count_nonzero(cb[k]!=ob[k])),max_abs_delta=float(np.abs(cb[k].astype(float)-ob[k].astype(float)).max())) for k in KEYS+['decoder_rates','pending']}
    return answer


def verify():
    # Pure saved-array/scalar process: no ConductanceNetwork or MotorOutput calls/imports.
    unchanged();inp=arrays(OUT/'INPUTS.npz');warm=load(SRC/'PREWARM_NEURAL_RECORDS.json');records=warm+load(SRC/'NEURAL_RECORDS.json')
    roots=arrays(ARCHIVE/'IDENTITIES.npz')['root_ids'];assert np.array_equal(roots,inp['root_ids'])
    for label,lookup in [('old',load(REPO/'outputs/neural_rebuild/sensory_channel_lookup.json')),('candidate',load(CAND/'sensory_channel_lookup_root888.candidate.json'))]:
        expected=np.zeros((1800,12000));lu=lookup['leg_channel_by_root']
        for i,r in enumerate(roots):
            pair=lu.get(str(int(r)))
            if pair is None:continue
            leg,ch=pair
            for k,rec in enumerate(records):
                row=rec['channels']['per_leg'][leg][ch];expected[k,i]=rec['channels']['scale_nA_per_unit']*float(row['value'])*row['weight']
        assert np.array_equal(expected,inp[f'{label}_current_nA'])
    with (OUT/'INPUT_DELTA.csv').open() as f:
        n=0
        for row in csv.DictReader(f):
            k=int(row['full_window']);i=int(row['local_index']);d=inp['candidate_current_nA'][k,i]-inp['old_current_nA'][k,i]
            assert int(row['root_id'])==roots[i] and float(row['delta_nA'])==d and float(row['signed_dose_nA_s'])==d*.001;n+=1
    assert n==1800*26
    assert load(OUT/'OLD_PREWARM_GATE.json')['passed']
    maxerr=0.;checked=0;decay=math.exp(-.0005/.005)
    for label,windows,mappingfile in [('OLD',1100,'MAPPING_OLD.json'),('C',1800,'MAPPING_C.json')]:
        a=arrays(OUT/f'{label}_NEURAL.npz');d=arrays(OUT/f'{label}_DECODE.npz');m=load(OUT/mappingfile)
        events=np.unpackbits(a['spikes_packed'],axis=1,count=12000).astype(bool)
        assert np.array_equal(events.reshape(windows,2,12000).sum(1),a['counts'])
        assert np.array_equal(events[:,a['selected_motor_indices']],a['selected_motor_events'])
        assert np.array_equal(a['selected_motor_root_ids'],roots[a['selected_motor_indices']])
        for ai,row in enumerate(m['rows']):
            idx=row['local_indices'];rate=0.
            for j,sample in enumerate(events):
                inst=(sum(int(sample[i]) for i in idx)/len(idx)/.0005) if idx else 0.
                rate=decay*rate+(1-decay)*inst;err=abs(rate-d['rates_halfstep_hz'][j,ai]);assert err<=1e-12;maxerr=max(maxerr,err);checked+=1
                if j%2:
                    k=j//2;assert d['decoded_next'][k,ai]==min(1.,max(0.,rate/100.))
                    assert d['pool_counts'][k,ai]==sum(int(events[t,i]) for t in [j-1,j] for i in idx)
        for mi,i in enumerate(a['selected_motor_indices']):
            rate=0.
            for j,sample in enumerate(events):
                rate=decay*rate+(1-decay)*float(sample[i])/.0005
                assert rate==a['selected_motor_rates_halfstep_hz'][j,mi]
        assert np.all(d['applied'][0]==0) and np.array_equal(d['applied'][1:],d['decoded_next'][:-1])
        assert np.array_equal(d['applied_source_interval'],np.arange(windows)-1)
        assert np.array_equal(d['decoded_source_interval'],np.arange(windows))
        assert np.array_equal(d['saturated_applied'],d['applied']>=1.)
        b=arrays(OUT/f'{label}_BOUNDARY.npz');fin=arrays(OUT/f'{label}_FINAL.npz')
        assert np.array_equal(b['spike_count'],a['counts'][:1100].sum(0))
        assert np.array_equal(fin['spike_count'],a['counts'].sum(0))
        assert np.array_equal(b['decoder_rates'],d['rates_window_hz'][1099])
        assert np.array_equal(b['pending'],d['decoded_next'][1099])
        assert np.array_equal(b['selected_motor_rates_hz'],a['selected_motor_rates_halfstep_hz'][2199])
        assert b['steps']==2200 and b['t_ms']==1100 and b['ring_i']==0 and b['decoder_interval']==1099
        assert fin['steps']==2*windows and fin['t_ms']==windows and fin['ring_i']==(2*windows)%5
        # Recover delay slots and refractory state exactly from saved event endpoints, no neural integration.
        for state,steps in [(b,2200),(fin,2*windows)]:
            ring=np.zeros((5,12000),bool)
            for j in range(steps-5,steps):ring[j%5]=events[j]
            assert np.array_equal(state['_ring'],ring)
            refractory=np.zeros(12000)
            for j in range(steps):refractory[events[j]]=(j+1)*.5+2.
            assert np.array_equal(state['refractory_until_ms'],refractory)
        if label=='OLD':
            for name in ['NEURAL_BOUNDARY.npz','PREWITHDRAW_NEURAL_STATE.npz']:
                original=arrays(SRC/name)
                for key in KEYS+['decoder_rates','pending']:assert np.array_equal(b[key],original[key])
            assert np.array_equal(d['applied'],np.array([r['applied_command'] for r in warm]))
            assert np.array_equal(d['decoded_next'],np.array([r['decoded_next_command'] for r in warm]))
            assert np.array_equal(d['rates_window_hz'],np.array([r['decoder']['rates_hz'] for r in warm]))
            assert np.array_equal(a['counts'].sum(1),np.array([r['total_network_spikes'] for r in warm]))
        else:
            assert np.array_equal(d['applied'][1100],b['pending'])
            assert np.array_equal(d['rates_window_hz'][1099],b['decoder_rates'])
            for j in range(2200,2202):
                for ai,row in enumerate(m['rows']):
                    idx=row['local_indices'];inst=events[j,idx].mean()/.0005 if idx else 0.
                    assert d['rates_halfstep_hz'][j,ai]==decay*d['rates_halfstep_hz'][j-1,ai]+(1-decay)*inst
    result=load(OUT/'RESULT.json');computed=metrics()
    assert all(result[k]==v for k,v in computed.items())
    unchanged();v=dict(passed=True,process_id=os.getpid(),method='Independent new process pure arrays+scalar encoder/filters; no network or decoder instantiation, no repeated neural integration.',
        exact_old_prewarm_boundary=True,exact_old_all1100_rates_commands=True,input_1800_windows_exact=True,
        all_pool_halfstep_rates_checked=checked,max_rate_error_hz=maxerr,selected202and204_MN_filters_checked=True,
        counts_events_ring_refractory_cumulative_spikes_exact=True,C_boundary_pending_rates_causal_continuity=True,
        source_hashes_unchanged=True,calls=dict(network_instances=0,neural_steps=0,body_steps=0,model_load=0,tier_build=0,training=0))
    dump(OUT/'INDEPENDENT_VERIFICATION.json',v);print(json.dumps(v,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','verify']);mode=ap.parse_args().mode
    result={'prepare':prepare,'run':run,'verify':verify}[mode]();sys.exit(result or 0)
