"""S1/S3/S4 archived-input interface experiment; never load a body or build a tier.
prepare -> run -> verify, in separate processes. No production admission or defaults changed.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SRC = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
ARCHIVE = Path('/home/sensen/Desktop/engine修复验收/s4-neural-boundary-replay/20261101T185129Z')
CAND = Path('/home/sensen/Desktop/engine修复验收/s4-banc888-interface-candidates/20261101T212622Z')
OUT = Path('/home/sensen/Desktop/engine修复验收/s4-banc888-interface-replay/20261101T214247Z')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    def cast(x):
        return x.tolist() if isinstance(x, np.ndarray) else x.item()
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False, default=cast) + '\n')


def load(p):
    return json.loads(Path(p).read_text())


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def archived_spikes():
    return np.unpackbits(arrays(ARCHIVE/'BASELINE_REPLAY.npz')['spikes_packed'], axis=1, count=12000).astype(bool)


def inputs():
    from engine.embodied.motor_output import build_motor_manifest
    from engine.embodied.sensory_channels import channel_index_map, ChannelSensoryEncoder
    identity = arrays(ARCHIVE/'IDENTITIES.npz')
    roots = identity['root_ids']
    old, oldstat = channel_index_map(12000, roots)
    new, newstat = channel_index_map(12000, roots, lookup_path=CAND/'sensory_channel_lookup_root888.candidate.json')
    # Full validation includes repeated indices within a group and cross-group ownership.
    ChannelSensoryEncoder(old, 12000)
    ChannelSensoryEncoder(new, 12000)
    mapping = load(SRC/'MAPPING.json')
    candidate = build_motor_manifest(load(CAND/'mn_pools_root888.candidate.json'), roots,
                                     [r['actuator'] for r in mapping['rows']])
    return roots, old, new, oldstat, newstat, mapping, candidate


def decode(spikes, mapping, initial, pending):
    from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
    dec = MotorOutput(mapping, MotorOutputConfig())
    dec.rates[:] = initial
    dec.elapsed_s = 1.1
    dec.interval = 1099
    decay = math.exp(-.0005/dec.config.filter_tau_s)
    applied, nexts, filters, counts, sources = [], [], [], [], []
    for k in range(700):
        applied.append(pending.copy())
        sp = spikes[2*k:2*k+2]
        rate = dec.rates.copy()
        for sample in sp:
            inst = np.array([sample[idx].mean()/.0005 if idx.size else 0. for idx in dec.groups])
            rate = decay*rate + (1-decay)*inst
            filters.append(rate.copy())
        pending, trace = dec.consume(sp, spike_dt_s=.0005, start_s=1.1+k*.001, end_s=1.1+(k+1)*.001)
        assert np.array_equal(rate, dec.rates)
        nexts.append(pending.copy())
        counts.append(trace['pool_spike_counts'])
        sources.append(trace['source_interval'])
    return dict(initial_rates=initial, initial_pending=applied[0], rates_halfstep_hz=np.array(filters),
                rates_window_hz=np.array(filters)[1::2], applied=np.array(applied), decoded_next=np.array(nexts),
                pool_counts=np.array(counts), applied_source_interval=np.arange(1099,1799),
                decoded_source_interval=np.array(sources), neural_start_s=1.1+np.arange(700)*.001,
                neural_end_s=1.1+(np.arange(700)+1)*.001,
                saturated_applied=np.array(applied)>=1., saturated_next=np.array(nexts)>=1.)


def prepare():
    assert not OUT.exists(), 'New evidence directory must not already exist'
    roots, old, new, oldstat, newstat, mapping, candidate = inputs()
    original = arrays(SRC/'NEURAL.npz')
    boundary = arrays(SRC/'NEURAL_BOUNDARY.npz')
    records = load(SRC/'NEURAL_RECORDS.json')
    prewarm = load(SRC/'PREWARM_NEURAL_RECORDS.json')
    assert boundary['decoder_rates'].shape == (90,)
    assert len(prewarm) == 1100 and len(records) == 700
    ident = load(ARCHIVE/'IDENTITY.json')
    # Verify sealed material, not only RESULT.first_mismatch being null.
    for p, h in load(ARCHIVE/'MANIFEST_SHA256.json').items():
        assert sha(p) == h, p
    for p, item in load(CAND/'MANIFEST_SHA256.json')['files'].items():
        assert sha(p) == item['sha256'], p
    for p, h in ident['input_sha256'].items():
        assert sha(p) == h, p
    assert ident['weight_multiplier'] == .2 and ident['inferred_boundary_scalars'] == dict(steps=2200,t_ms=1100.,ring_i=0)
    base = arrays(ARCHIVE/'BASELINE_REPLAY.npz')
    spikes = archived_spikes()
    expected_next = np.array([r['decoded_next_command'] for r in records])
    expected_rates = np.array([r['decoder']['rates_hz'] for r in records])
    checks = dict(counts_700x12000_exact=np.array_equal(spikes.reshape(700,2,12000).sum(1), original['spike_counts']),
                  applied_700x90_exact=np.array_equal(base['applied'], original['commands']),
                  decoded_next_700x90_exact=np.array_equal(base['decoded_next'],expected_next),
                  records_applied_exact=np.array_equal(original['commands'],np.array([r['applied_command'] for r in records])),
                  initial_pending_exact=np.array_equal(base['applied'][0],boundary['pending']))
    red = decode(spikes,mapping,boundary['decoder_rates'],boundary['pending'].copy())
    checks.update(independent_decoder_applied_exact=np.array_equal(red['applied'],base['applied']),
                  independent_decoder_next_exact=np.array_equal(red['decoded_next'],base['decoded_next']),
                  original_rates_700x90_exact=np.array_equal(red['rates_window_hz'],expected_rates))
    assert all(checks.values()), checks
    old_I = original['sensory_current_nA']
    rebuilt = np.zeros_like(old_I)
    new_I = np.zeros_like(old_I)
    group_scalars, groupnames, group_checks, owner_old, owner_new = [], [], [], {}, {}
    for leg, channels in old.items():
        for ch, idx in channels.items():
            vals = np.array([r['channels']['scale_nA_per_unit']*float(r['channels']['per_leg'][leg][ch]['value'])*r['channels']['per_leg'][leg][ch]['weight'] for r in records])
            assert np.array_equal(old_I[:,idx],np.broadcast_to(vals[:,None],(700,len(idx))))
            if ch in ('club_vibration','multidendritic_noxious'):
                assert np.all(vals==0)
            rebuilt[:,idx] = vals[:,None]
            new_I[:,new[leg][ch]] = vals[:,None]
            group_scalars.append(vals);groupnames.append(leg+'/'+ch)
            group_checks.append(dict(leg=leg,channel=ch,old_n=len(idx),new_n=len(new[leg][ch]),all_700_members_same_exact=True,encoder_scalar_exact=True,declared_zero=ch in ('club_vibration','multidendritic_noxious')))
            for i in idx: owner_old[i]=(leg,ch)
            for i in new[leg][ch]: owner_new[i]=(leg,ch)
    assert np.array_equal(rebuilt,old_I)
    assert all(owner_old[i]==owner_new[i] for i in set(owner_old)&set(owner_new))
    added = sorted(set(owner_new)-set(owner_old));removed=sorted(set(owner_old)-set(owner_new))
    assert len(added)==26 and not removed
    oldmotor=sorted({i for r in mapping['rows'] for i in r['local_indices']})
    newmotor=sorted({i for r in candidate['rows'] for i in r['local_indices']})
    assert len(oldmotor)==202 and len(newmotor)==204
    initial_B=boundary['decoder_rates'].copy();changed=[]
    for a,b in zip(mapping['rows'],candidate['rows']):
        oi,ni=set(a['local_indices']),set(b['local_indices'])
        assert oi <= ni
        if oi != ni:
            j=a['actuator_index']; initial_B[j] *= len(oi)/len(ni)
            changed.append(dict(actuator=a['actuator'],actuator_index=j,old_indices=sorted(oi),new_indices=sorted(ni),
                                added_root_ids=roots[sorted(ni-oi)],old_pool_rate_hz=boundary['decoder_rates'][j],new_pool_rate_hz=initial_B[j],
                                rule='Retain old sum of filtered per-cell rates n_old*old_mean; assign new MN contribution exactly zero; divide by n_new.'))
    assert len(changed)==2
    assert candidate['usable_actuators']==74 and len([r for r in candidate['rows'] if r['status']=='unresolved'])==16
    assert load(CAND/'mn_pools_root888.candidate.json')['status']=='CANDIDATE_NOT_RUNTIME'
    sources=list(ident['input_sha256'])+[str(ARCHIVE/n) for n in ['ACTUAL_We.npz','ACTUAL_Wi.npz','IDENTITIES.npz','IDENTITY.json','BASELINE_REPLAY.npz','MANIFEST_SHA256.json']]
    sources += [str(SRC/n) for n in ['BODY.npz','NEURAL_RECORDS.json','PREWARM_NEURAL_RECORDS.json','PREWITHDRAW_NEURAL_STATE.npz']]
    sources += [str(CAND/n) for n in ['sensory_channel_lookup_root888.candidate.json','mn_pools_root888.candidate.json','MANIFEST_SHA256.json']]
    sources += [str(REPO/n) for n in ['outputs/neural_rebuild/sensory_channel_lookup.json','outputs/neural_rebuild/neural_rebuild_11_mn_pools.json','engine/embodied/sensory_channels.py','engine/embodied/runtime_pool_guard.py','GOAL_TRACKER.zh-CN.md']]
    sources += [str(Path(__file__).resolve())]
    hashes={p:sha(p) for p in dict.fromkeys(sources)}
    OUT.mkdir(parents=True)
    dump(OUT/'PREREGISTRATION.json',dict(scope='S1/S3/S4 saved-body-input neural replay, no new closed-loop behavior',
          repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          source_sha256=hashes,identity=ident,baseline_gate=checks,baseline_policy='Reuse sealed exact baseline after actual full array comparisons and exact redecoding; no baseline neural integration.',
          physical_boundary_s=1.3,neural_boundary_s=1.1,neural_steps_boundary=2200,ring_index_boundary=0,
          baseline_substep_limit='Original historical 0.5ms spike raster was not saved; sealed replay raster is authenticated and reproduces all historical 1ms counts/rates/commands exactly.',
          arms=dict(A='root888 sensory, old MN; restore all saved neural arrays + source-derived clocks, original90 rates and pending.',
                    B='Same A neural trajectory, independent fresh decoder with root888 MN; two newly admitted MN initial filtered contribution ZERO as explicit state intervention. First interval applies original pending.'),
          decoder_boundary=dict(shape=list(boundary['decoder_rates'].shape),all12000_rates_available=False,complete_prewarm_neuron_events_available=False,
                                full_decoder_continuity_B=False,changed_pools=changed,
                                source_note='Prewarm has currents/channel scalars, total and old-pool counts; no full neuron spike events. No prewarm reintegration permitted here.'),
          input_policy='Use exact saved encoder scale*value*weight per leg/channel, prove each old group same scalar and full old current reconstruction exact. New groups disjoint validated with production encoder constructor; unmodelled channels zero.',
          old_sensory=oldstat,new_sensory=newstat,groups=group_checks,
          timing='For each1ms window: apply saved previous pending; two0.5ms neural steps; filter and decode next command at end. End command valid next interval. B uses A substep raster, not1ms counts alone.',
          outcomes=['early15ms and full700ms network/202oldMN/204candidateMN spikes','applied/next command nonzero count and saturation with definition ctrl>0 and ctrl>=1','A-minus-baseline sensory effect, B-minus-A readout effect'],
          planned_calls=dict(physical_model_load=0,body_step=0,tier_build=0,training=0,network_instances=1,neural_step=1400),
          stop_rule='Stop before candidates if any exact baseline/source/input group gate fails; never treat mismatch:null alone as a comparison.',
          limitations='B changes readout and declared boundary initial contributions only; no feedback from commands. BODY .5ms rows are historical observations, no new physical trajectory. No standing/walking interpretation; physical force calibration NOT_ESTABLISHED.'))
    dump(OUT/'BASELINE_GATE.json',dict(checks=checks,compared_elements=dict(counts=8400000,applied=63000,next=63000,rates=63000),authenticated_archived_source=True))
    dump(OUT/'MAPPING_A.json',mapping);dump(OUT/'MAPPING_B.json',candidate)
    dump(OUT/'INPUT_REMAP_PROOF.json',dict(groups=group_checks,full_original_current_exact=True,
         old_group_memberships=len(owner_old),new_group_memberships=len(owner_new),old_nonzero_any=int(np.any(old_I!=0,axis=0).sum()),new_nonzero_any=int(np.any(new_I!=0,axis=0).sum()),
         no_group_duplicates_or_overlap=True,retained_identities_unchanged=True,
         added=[dict(root_id=int(roots[i]),local_index=i,leg=owner_new[i][0],channel=owner_new[i][1]) for i in added],removed=removed))
    np.savez_compressed(OUT/'INPUTS.npz',current_nA=new_I,group_current_nA=np.array(group_scalars).T,group_names=np.array(groupnames),
                        root_ids=roots,added_indices=added,removed_indices=np.array(removed,dtype=int),delta_nA=(new_I-old_I)[:,added],
                        initial_rates_A=boundary['decoder_rates'],initial_rates_B=initial_B,pending=boundary['pending'],old_motor_indices=oldmotor,new_motor_indices=newmotor)
    np.savez_compressed(OUT/'BASELINE_DECODE.npz',**red)
    with (OUT/'INPUT_DELTA.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['window','root_id','local_index','leg','channel','change','old_nA','new_nA','delta_nA','signed_dose_nA_s','absolute_dose_nA_s'])
        for k in range(700):
            for i in added+removed:
                leg,ch=(owner_new if i in owner_new else owner_old)[i];d=new_I[k,i]-old_I[k,i]
                w.writerow([k,int(roots[i]),i,leg,ch,'added' if i in owner_new else 'removed',old_I[k,i],new_I[k,i],d,d*.001,abs(d)*.001])
    with (OUT/'GROUP_WINDOW_DOSE.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['window','leg_channel','old_targets','new_targets','current_per_target_nA','old_dose_nA_s','new_dose_nA_s','delta_dose_nA_s'])
        for j,g in enumerate(group_checks):
            for k in range(700):
                v=group_scalars[j][k]*.001;w.writerow([k,groupnames[j],g['old_n'],g['new_n'],group_scalars[j][k],v*g['old_n'],v*g['new_n'],v*(g['new_n']-g['old_n'])])
    print(json.dumps(dict(prepared=str(OUT),baseline_gate=checks,decoder_rates=90,added_targets=26,planned_new_neural_steps=1400),ensure_ascii=False))


def assert_sources():
    prereg=load(OUT/'PREREGISTRATION.json')
    for p,h in prereg['source_sha256'].items():
        assert sha(p)==h,p
    return prereg


def run():
    prereg=assert_sources()
    assert not (OUT/'RESULT.json').exists() and not (OUT/'A_NEURAL.npz').exists()
    assert all(load(OUT/'BASELINE_GATE.json')['checks'].values())
    from scipy.sparse import load_npz
    from engine.neural_cond import ConductanceNetwork, ConductanceParams
    boundary=arrays(SRC/'NEURAL_BOUNDARY.npz');drive=arrays(OUT/'INPUTS.npz')
    ident=prereg['identity'];p=ConductanceParams(**ident['conductance_params'])
    net=ConductanceNetwork(12000,p,dt_ms=.5,voltage_method='backward_euler',threshold_offset_mV=boundary['threshold_offset_mV'])
    net.set_connectivity(load_npz(ARCHIVE/'ACTUAL_We.npz'),load_npz(ARCHIVE/'ACTUAL_Wi.npz'))
    for key in ['_ring','v','ge','gi','adaptation','refractory_until_ms','spike_count']:
        getattr(net,key)[:]=boundary[key]
        assert np.array_equal(getattr(net,key),boundary[key])
    net.steps=2200;net.t_ms=1100.;net._ring_i=0
    initial={k:getattr(net,k).copy() for k in boundary if hasattr(net,k)}
    np.savez_compressed(OUT/'RESTORED_NEURAL_BOUNDARY.npz',**initial,steps=net.steps,t_ms=net.t_ms,ring_i=net._ring_i)
    start=time.monotonic();spikes=np.empty((1400,12000),bool)
    for k in range(700):
        for sub in range(2):
            spikes[2*k+sub]=net.step(drive['current_nA'][k])
    assert net.steps==3600 and net.t_ms==1800. and net._ring_i==0
    np.savez_compressed(OUT/'A_NEURAL.npz',spikes_packed=np.packbits(spikes,axis=1),counts=spikes.reshape(700,2,12000).sum(1),
                        motor_events_204=spikes[:,drive['new_motor_indices']],motor_local_indices=drive['new_motor_indices'],
                        motor_root_ids=drive['root_ids'][drive['new_motor_indices']],neural_event_end_s=1.1+(np.arange(1400)+1)*.0005,
                        body_aligned_event_end_s=1.3+(np.arange(1400)+1)*.0005)
    final={k:getattr(net,k).copy() for k in boundary if hasattr(net,k)}
    np.savez_compressed(OUT/'FINAL_NEURAL_STATE.npz',**final,steps=net.steps,t_ms=net.t_ms,ring_i=net._ring_i)
    for label in ['A','B']:
        d=decode(spikes,load(OUT/f'MAPPING_{label}.json'),drive[f'initial_rates_{label}'],drive['pending'].copy())
        d['absolute_start_s']=np.array([r['absolute_start_s'] for r in load(SRC/'NEURAL_RECORDS.json')])
        d['absolute_end_s']=np.array([r['absolute_end_s'] for r in load(SRC/'NEURAL_RECORDS.json')])
        np.savez_compressed(OUT/f'{label}_DECODE.npz',**d)
    result=metrics()
    result.update(calls=dict(physical_model_load=0,body_steps=0,tier_build=0,training=0,network_instances=1,neural_steps=1400,baseline_neural_steps=0,B_neural_steps=0),
                  elapsed_s=time.monotonic()-start,baseline_gate_passed=True,
                  B_neural_trajectory='Exact same A file/raster by construction; independent MotorOutput decoder only',B_complete_decoder_state_continuity=False)
    assert_sources();dump(OUT/'RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def metrics():
    inp=arrays(OUT/'INPUTS.npz');a=arrays(OUT/'A_NEURAL.npz');asp=np.unpackbits(a['spikes_packed'],axis=1,count=12000).astype(bool)
    bsp=archived_spikes();dec={n:arrays(OUT/(n+'_DECODE.npz')) for n in ['BASELINE','A','B']}
    answer={}
    for windows in [15,700]:
        part={}
        for name,sp in [('BASELINE',bsp),('A',asp),('B',asp)]:
            d=dec[name];part[name]=dict(network_spikes=int(sp[:2*windows].sum()),
                 old202_MN_spikes=int(sp[:2*windows,inp['old_motor_indices']].sum()),
                 candidate204_MN_spikes=int(sp[:2*windows,inp['new_motor_indices']].sum()),
                 added2_MN_spikes=int(sp[:2*windows,np.setdiff1d(inp['new_motor_indices'],inp['old_motor_indices'])].sum()),
                 applied_nonzero=int((d['applied'][:windows]>0).sum()),next_nonzero=int((d['decoded_next'][:windows]>0).sum()),
                 applied_saturated=int((d['applied'][:windows]>=1).sum()),next_saturated=int((d['decoded_next'][:windows]>=1).sum()))
        part['A_minus_baseline']={k:part['A'][k]-part['BASELINE'][k] for k in part['A']}
        part['B_minus_A']={k:part['B'][k]-part['A'][k] for k in part['A']}
        for label,left,right in [('A_vs_baseline','A','BASELINE'),('B_vs_A','B','A')]:
            x,y=dec[left],dec[right]
            part[label]=dict(applied_differing_elements=int(np.count_nonzero(x['applied'][:windows]!=y['applied'][:windows])),
                 next_differing_elements=int(np.count_nonzero(x['decoded_next'][:windows]!=y['decoded_next'][:windows])),
                 applied_max_abs_delta=float(np.abs(x['applied'][:windows]-y['applied'][:windows]).max()),
                 next_max_abs_delta=float(np.abs(x['decoded_next'][:windows]-y['decoded_next'][:windows]).max()))
        part['A_vs_baseline']['spike_bits_differing']=int(np.count_nonzero(asp[:2*windows]!=bsp[:2*windows]))
        answer[f'first_{windows}ms']=part
    return answer


def verify():
    # Intentionally no neural model or decoder imports; only saved arrays + scalar formulas.
    assert_sources();pr=load(OUT/'PREREGISTRATION.json');inp=arrays(OUT/'INPUTS.npz')
    roots=arrays(ARCHIVE/'IDENTITIES.npz')['root_ids'];assert np.array_equal(inp['root_ids'],roots)
    old_I=arrays(SRC/'NEURAL.npz')['sensory_current_nA'];records=load(SRC/'NEURAL_RECORDS.json')
    maps=[load(REPO/'outputs/neural_rebuild/sensory_channel_lookup.json')['leg_channel_by_root'],load(CAND/'sensory_channel_lookup_root888.candidate.json')['leg_channel_by_root']]
    rebuilt=[]
    for lookup in maps:
        v=np.zeros((700,12000))
        for i,root in enumerate(roots):
            if str(int(root)) in lookup:
                leg,ch=lookup[str(int(root))]
                for k,r in enumerate(records):
                    cr=r['channels'];row=cr['per_leg'][leg][ch]
                    v[k,i]=cr['scale_nA_per_unit']*float(row['value'])*row['weight']
        rebuilt.append(v)
    assert np.array_equal(rebuilt[0],old_I) and np.array_equal(rebuilt[1],inp['current_nA'])
    assert np.array_equal((rebuilt[1]-rebuilt[0])[:,inp['added_indices']],inp['delta_nA'])
    changed_indices=np.flatnonzero(np.any(rebuilt[1]!=rebuilt[0],axis=0))
    assert set(changed_indices)<=set(inp['added_indices'])
    with (OUT/'INPUT_DELTA.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==700*26
    for row in rows:
        k=int(row['window']);i=int(row['local_index']);d=rebuilt[1][k,i]-rebuilt[0][k,i]
        assert int(row['root_id'])==roots[i] and float(row['delta_nA'])==d and float(row['signed_dose_nA_s'])==d*.001
    with (OUT/'GROUP_WINDOW_DOSE.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==700*42
    for row in rows:
        k=int(row['window']);j=list(inp['group_names']).index(row['leg_channel']);v=inp['group_current_nA'][k,j]
        assert float(row['current_per_target_nA'])==v
        assert float(row['delta_dose_nA_s'])==v*.001*(int(row['new_targets'])-int(row['old_targets']))
    boundary=arrays(SRC/'NEURAL_BOUNDARY.npz');restored=arrays(OUT/'RESTORED_NEURAL_BOUNDARY.npz')
    for k in ['threshold_offset_mV','_ring','v','ge','gi','adaptation','refractory_until_ms','spike_count']:
        assert np.array_equal(restored[k],boundary[k])
    assert restored['steps']==2200 and restored['t_ms']==1100 and restored['ring_i']==0
    sp=arrays(OUT/'A_NEURAL.npz');raster=np.unpackbits(sp['spikes_packed'],axis=1,count=12000).astype(bool)
    assert np.array_equal(raster.reshape(700,2,12000).sum(1),sp['counts'])
    assert np.array_equal(raster[:,inp['new_motor_indices']],sp['motor_events_204'])
    assert np.array_equal(sp['motor_root_ids'],roots[inp['new_motor_indices']])
    final=arrays(OUT/'FINAL_NEURAL_STATE.npz')
    assert np.array_equal(final['spike_count']-boundary['spike_count'],sp['counts'].sum(0))
    assert final['steps']==3600 and final['t_ms']==1800 and final['ring_i']==0
    decay=math.exp(-.0005/.005);maxerr=0.
    for label,events in [('BASELINE',archived_spikes()),('A',raster),('B',raster)]:
        d=arrays(OUT/f'{label}_DECODE.npz');mapping=load(OUT/('MAPPING_A.json' if label=='BASELINE' else f'MAPPING_{label}.json'))
        for ai,row in enumerate(mapping['rows']):
            idx=row['local_indices'];rate=float(d['initial_rates'][ai])
            for j,sample in enumerate(events):
                instantaneous=(sum(int(sample[i]) for i in idx)/len(idx)/.0005) if idx else 0.
                rate=decay*rate+(1-decay)*instantaneous
                err=abs(rate-d['rates_halfstep_hz'][j,ai]);maxerr=max(maxerr,err)
                assert err<=1e-12
                if j%2:
                    k=j//2;assert d['decoded_next'][k,ai]==min(1.,max(0.,rate/100.))
                    assert d['pool_counts'][k,ai]==sum(int(events[t,i]) for t in [j-1,j] for i in idx)
        assert np.array_equal(d['applied'][0],boundary['pending'])
        assert np.array_equal(d['applied'][1:],d['decoded_next'][:-1])
        assert np.array_equal(d['applied_source_interval'],np.arange(1099,1799))
        assert np.array_equal(d['decoded_source_interval'],np.arange(1100,1800))
        assert np.array_equal(d['saturated_next'],d['decoded_next']>=1.)
    ma=load(OUT/'MAPPING_A.json');mb=load(OUT/'MAPPING_B.json')
    for a,b in zip(ma['rows'],mb['rows']):
        n0,n1=len(a['local_indices']),len(b['local_indices']);ai=a['actuator_index']
        expected=boundary['decoder_rates'][ai]*(n0/n1) if n0!=n1 else boundary['decoder_rates'][ai]
        assert inp['initial_rates_B'][ai]==expected
        assert set(a['local_indices'])<=set(b['local_indices'])
    assert sum(bool(r['local_indices']) for r in mb['rows'])==74
    assert len({tuple(r['local_indices']) for r in ma['rows'] if r['local_indices']})==66
    assert len({tuple(r['local_indices']) for r in mb['rows'] if r['local_indices']})==66
    base=arrays(OUT/'BASELINE_DECODE.npz');original=arrays(SRC/'NEURAL.npz')
    assert np.array_equal(archived_spikes().reshape(700,2,12000).sum(1),original['spike_counts'])
    assert np.array_equal(base['applied'],original['commands'])
    assert np.array_equal(base['decoded_next'],np.array([r['decoded_next_command'] for r in records]))
    result=load(OUT/'RESULT.json');m=metrics()
    assert all(result[k]==v for k,v in m.items())
    verification=dict(passed=True,process_id=os.getpid(),verification='Independent new process, pure array/scalar reconstruction; no network initialization/integration, no MotorOutput import.',
                      input_700x12000_exact=True,baseline_counts_applied_next_exact=True,
                      all_three_decoder_halfstep_rates_checked=1400*90*3,max_rate_error_hz=maxerr,
                      candidate204_events_exact=True,candidate_counts_exact=True,all_commands_causal=True,
                      source_hashes_unchanged=True,calls=dict(model_load=0,body_step=0,network_instances=0,tier_build=0,neural_step=0,training=0))
    dump(OUT/'INDEPENDENT_VERIFICATION.json',verification);print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','verify']);mode=ap.parse_args().mode
    {'prepare':prepare,'run':run,'verify':verify}[mode]()
