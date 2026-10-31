"""S286 offline saved-data verifier; never imports simulation/network modules.

Archive identity, exact historical legacy replay, independent command scheduling,
and native contact/kinematic metrics are separate from behavior success.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

REFERENCE = Path('/home/sensen/Desktop/engine修复验收/s284-friction-sensitivity/20261026T161000Z/baseline')
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(str(type(value)))


def load_npz(path):
    # Each archive is opened once; arrays survive the closed ZIP handle.
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def delta_summary(left, right, time_ms):
    diff = np.abs(left.astype(float) - right.astype(float))
    flat = diff.reshape(len(diff), -1)
    rows = np.flatnonzero(np.any(flat != 0, axis=1))
    return dict(max_abs=float(diff.max()) if diff.size else 0.,
                changed_values=int(np.count_nonzero(diff)),
                first_diff_index=int(rows[0]) if rows.size else None,
                first_diff_time_ms=float(time_ms[rows[0]]) if rows.size else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args(argv).output
    checks, metrics = [], {}
    verifier_sha_start=sha(Path(__file__))

    def check(name, value, **extra):
        checks.append(dict(name=name, passed=bool(value), **extra))

    try:
        prereg = load_json(out / 'PREREG.json')
        arms = prereg['arms']
        check('round S286', prereg['round'] == 'S286')
        check('expected legacy/causal arms', arms == ['legacy', 'causal_carry'])
        for path, expected in prereg['source_hashes'].items():
            check('source hash ' + path, Path(path).is_file() and sha(path) == expected)
        check('source hash table nonempty', bool(prereg['source_hashes']))
        complete = load_json(out / 'COMPLETE.json')
        check('capture complete', complete.get('capture_complete') is True and complete.get('source_unchanged') is True)
        check('no failed marker', not (out / 'FAILED.json').exists())
        ledger = load_json(out / 'CALL_LEDGER.json')
        expected_calls = dict(model_load=2, tier_build=1, network_init=2, live_step=8000,
                              neural_tick=1600, live_fwd_position=2, live_forward=0,
                              diagnostic_forward=8002)
        check('exact simulation ledger', ledger == expected_calls, actual=ledger)
        check('COMPLETE ledger', complete.get('calls') == ledger)
        data = {}
        for arm in arms:
            dest = out / arm
            tr = load_npz(dest / 'TRAJECTORY.npz')
            ne = load_npz(dest / 'NEURAL.npz')
            co = load_npz(dest / 'CTRL.npz')
            meta = load_json(dest / 'MODEL_METADATA.json')
            data[arm] = dict(tr=tr, ne=ne, co=co, meta=meta,
                             params=load_json(dest / 'PARAMETERS.json'))
            t = tr['time']
            check(arm + ' state dimensions', tr['qpos'].shape == (4001,49) and tr['qvel'].shape == (4001,48))
            check(arm + ' control dimensions', co['ctrl'].shape == (4000,90) and ne['ctrl'].shape == (4000,90))
            check(arm + ' neural/rate dimensions', ne['time_ms'].shape == (800,) and ne['spikes'].shape[0] == 800 and ne['rates_hz'].shape == (400,90) and ne['raw_rates_hz'].shape == (400,2,90))
            check(arm + ' time axis', np.allclose(t, np.arange(4001)*.0001, rtol=0, atol=1e-11))
            check(arm + ' neural axis', np.array_equal(ne['time_ms'], np.arange(1,801)*.5))
            check(arm + ' no external force', not np.any(tr['external_applied']) and not np.any(tr['qfrc_applied']))
            check(arm + ' control artifact agreement', np.array_equal(co['ctrl'], ne['ctrl']) and np.array_equal(co['ctrl'], tr['ctrl'][1:]))
            check(arm + ' control range', np.all((co['ctrl']>=0) & (co['ctrl']<=1)))
            check(arm + ' names count', len(meta['names'])==90 and co['names'].tolist()==meta['names'])
            check(arm + ' finite arrays', all(np.isfinite(v).all() for v in tr.values() if v.dtype.kind in 'biufc') and all(np.isfinite(v).all() for v in ne.values() if v.dtype.kind in 'biufc'))
            # Packed contacts are sufficient for independent physical-force reconstruction.
            raw = tr['contact_raw']; offsets = tr['contact_offsets']; counts = tr['contact_count']
            check(arm + ' packed contact layout', raw.ndim==2 and raw.shape[1]==21 and offsets.shape==(4002,) and np.array_equal(np.diff(offsets),counts) and offsets[-1]==len(raw))
            contacts = load_json(dest/'CONTACTS.json')
            check(arm+' JSON contacts agree', len(contacts)==4001 and all(np.array_equal(np.asarray(rows,float).reshape(-1,21),raw[int(offsets[i]):int(offsets[i+1])]) for i,rows in enumerate(contacts)))
            floor = int(meta['floor']); geom_leg = {int(k):v for k,v in meta['geom_leg'].items()}
            foot=np.zeros((4001,6,3)); nonfoot=np.zeros((4001,3)); nonfoot_pen=np.zeros(4001,int)
            first_contact=None;first_margin=None;first_pen=None
            for i in range(4001):
                for c in raw[int(offsets[i]):int(offsets[i+1])]:
                    g1,g2=int(c[0]),int(c[1])
                    if floor not in (g1,g2): continue
                    other=g2 if g1==floor else g1
                    force=(1 if g1==floor else -1)*(c[6:15].reshape(3,3).T @ c[15:18])
                    if other in geom_leg:
                        foot[i,LEGS.index(geom_leg[other])]+=force
                    else:
                        nonfoot[i]+=force
                        if first_contact is None:first_contact=float(t[i]*1000)
                        if c[2]>=0 and first_margin is None:first_margin=float(t[i]*1000)
                        if c[2]<0:
                            nonfoot_pen[i]+=1
                            if first_pen is None:first_pen=float(t[i]*1000)
            check(arm+' physical foot force reconstruction',np.allclose(foot,tr['foot_force_world'],rtol=0,atol=2e-12))
            check(arm+' nonfoot penetration reconstruction',np.array_equal(nonfoot_pen,tr['body_penetrating_contacts']))
            quat=tr['qpos'][:,3:7];angle=2*np.arccos(np.clip(abs((quat*quat[0]).sum(axis=1)),0,1))
            spike_times=ne['time_ms'][np.any(ne['spikes'],axis=1)];late=t>=.35-1e-12
            metrics[arm]=dict(target_spikes_0_200=int(ne['spikes'][ne['time_ms']<=200].sum()),target_spikes_200_400=int(ne['spikes'][ne['time_ms']>200].sum()),target_last_spike_ms=float(spike_times[-1]) if spike_times.size else None,COM_dz400_native=float(tr['com'][-1,2]-tr['com'][0,2]),root_dz400_native=float(tr['qpos'][-1,2]-tr['qpos'][0,2]),first_nonfoot_contact_including_margin_ms=first_contact,first_nonfoot_margin_dist_ge0_ms=first_margin,first_nonfoot_penetration_dist_lt0_ms=first_pen,max_orientation_relative_initial_rad=float(angle.max()),max_joint_limit_excess_native=float(tr['max_limit'].max()),foot_Fz_350_400_mean_native=float(foot[late,:,2].sum(axis=1).mean()),nonfoot_Fz_350_400_mean_native=float(nonfoot[late,2].mean()),foot_force_world_native_per_leg_last=foot[-1].tolist(),nonfoot_force_world_native_last=nonfoot[-1].tolist(),support_walking='NOT_ESTABLISHED',force_units='native; absolute physical calibration NOT_ESTABLISHED')
            i200=int(np.argmin(abs(t-.2)))
            metrics[arm]['sample200']={'time_ms':float(t[i200]*1000),'foot_force_world_native_per_leg':foot[i200].tolist(),'foot_Fz_total_native':float(foot[i200,:,2].sum()),'nonfoot_force_world_native':nonfoot[i200].tolist(),'COM_dz_native':float(tr['com'][i200,2]-tr['com'][0,2]),'root_dz_native':float(tr['qpos'][i200,2]-tr['qpos'][0,2]),'orientation_rad':float(angle[i200]),'max_limit_excess_native':float(tr['max_limit'][i200])}
            data[arm].update(foot=foot,nonfoot=nonfoot,angle=angle)
        reference_tr=load_npz(REFERENCE/'TRAJECTORY.npz');reference_ne=load_npz(REFERENCE/'NEURAL.npz')
        for filename,reference,actual in [('TRAJECTORY',reference_tr,data['legacy']['tr']),('NEURAL',reference_ne,data['legacy']['ne'])]:
            check('legacy '+filename+' has every historical key',set(reference)<=set(actual))
            for key,value in reference.items():
                check('legacy exact '+filename+'/'+key,key in actual and value.dtype==actual[key].dtype and np.array_equal(value,actual[key]),historical_shape=list(value.shape),actual_shape=list(actual[key].shape) if key in actual else None)
        for key in ('qpos','qvel','act','ctrl'):
            check('initial '+key+' same',np.array_equal(data['legacy']['tr'][key][0],data['causal_carry']['tr'][key][0]))
        check('neural parameters same',data['legacy']['params']==data['causal_carry']['params'])
        # COMMAND_TIMING adapter is intentionally explicit; filled from runner schema.
        verify_timing(out,data,check,metrics)
        times=np.arange(4000)*.1
        metrics['paired_commands']=[dict(index=j,actuator=data['legacy']['meta']['names'][j],**delta_summary(data['legacy']['co']['ctrl'][:,j:j+1],data['causal_carry']['co']['ctrl'][:,j:j+1],times)) for j in range(90)]
        metrics['target_spikes_paired']=delta_summary(data['legacy']['ne']['spikes'],data['causal_carry']['ne']['spikes'],data['legacy']['ne']['time_ms'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,2,figsize=(13,11))
        for arm in arms:
            a=data[arm];tr=a['tr'];tt=tr['time']*1000
            axes[0,0].plot(tt,tr['com'][:,2]-tr['com'][0,2],label=arm)
            axes[0,1].plot(tt,tr['qpos'][:,2]-tr['qpos'][0,2],label=arm)
            axes[1,0].plot(tt,a['foot'][:,:,2].sum(axis=1),label=arm+' feet')
            axes[1,0].plot(tt,a['nonfoot'][:,2],ls='--',label=arm+' nonfoot')
            axes[1,1].plot(tt,a['angle'],label=arm)
            axes[2,0].plot(a['ne']['time_ms'],a['ne']['spikes'].sum(axis=1),label=arm)
        axes[2,1].plot(times,np.max(abs(data['legacy']['co']['ctrl']-data['causal_carry']['co']['ctrl']),axis=1),label='max across90')
        for ax,title in zip(axes.flat,['COM dz native','root dz native','Physical Fz native (no calibration)','Orientation relative initial rad','Target-only spikes/tick','Paired command absolute difference']):
            ax.set_title(title);ax.set_xlabel('individual clock ms');ax.legend(fontsize=7);ax.grid(alpha=.3)
        fig.suptitle('S286 command timing support audit — saved data only; no support/walking success claim');fig.tight_layout();fig.savefig(out/'OUTCOMES.png',dpi=140);plt.close(fig)
        check('no simulator/network modules imported',not any(k=='mujoco' or k.startswith('engine.') for k in sys.modules))
        check('verifier own source unchanged during analysis',sha(Path(__file__))==verifier_sha_start)
        result=dict(passed=all(c['passed'] for c in checks),checks=checks,failed=[c['name'] for c in checks if not c['passed']],verifier_source_path=str(Path(__file__).resolve()),verifier_sha256=verifier_sha_start,offline_calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0),behavior_claim='NOT_ESTABLISHED')
        (out/'METRICS.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=serial)+'\n')
        (out/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serial)+'\n')
        print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed']),ensure_ascii=False))
        return 0 if result['passed'] else 1
    except Exception as exc:
        print('offline verifier error: '+repr(exc),file=sys.stderr)
        return 2


def verify_timing(out,data,check,metrics):
    expected_starts=np.arange(400)*.001
    expected_ends=(np.arange(400)+1)*.001
    for arm in ('legacy','causal_carry'):
        trace=load_npz(out/arm/'COMMAND_TIMING.npz')
        required={'initial_carry','decoded','source_start_s','source_end_s','valid_from_s','applied','applied_start_s','applied_end_s','applied_valid_from_s','applied_source_index'}
        check(arm+' complete command trace',required<=set(trace))
        if not required<=set(trace):raise ValueError('missing timing trace fields: '+repr(required-set(trace)))
        decoded=trace['decoded'];initial=trace['initial_carry'];applied=trace['applied']
        check(arm+' command trace shapes',decoded.shape==(400,90) and initial.shape==(90,) and applied.shape==(400,90))
        initial_saved=load_npz(out/arm/'INITIAL_CARRY.npz')
        check(arm+' initial carry equals original initial control',np.array_equal(initial,data[arm]['tr']['ctrl'][0]) and np.array_equal(initial,initial_saved['command']) and np.array_equal(initial,np.asarray([0.]*90)))
        check(arm+' initial carry valid at0',float(initial_saved['valid_from_s'])==0. and int(initial_saved['source_interval'])==-1)
        check(arm+' contiguous neural intervals',np.allclose(trace['source_start_s'],expected_starts,rtol=0,atol=1e-12) and np.allclose(trace['source_end_s'],expected_ends,rtol=0,atol=1e-12))
        check(arm+' decoded valid at end',np.array_equal(trace['valid_from_s'],trace['source_end_s']))
        check(arm+' contiguous applied body intervals',np.allclose(trace['applied_start_s'],expected_starts,rtol=0,atol=1e-11) and np.allclose(trace['applied_end_s'],expected_ends,rtol=0,atol=1e-11))
        if arm=='legacy':
            reconstructed=decoded.copy();source_index=np.arange(400);valid=trace['valid_from_s']
        else:
            reconstructed=np.concatenate([initial[None,:],decoded[:-1]],axis=0);source_index=np.arange(400)-1;valid=np.concatenate([[0.],trace['valid_from_s'][:-1]])
        check(arm+' independently reconstructed interval command',np.array_equal(applied,reconstructed))
        check(arm+' independently reconstructed source index',np.array_equal(trace['applied_source_index'],source_index))
        check(arm+' independently reconstructed applied valid_from',np.allclose(trace['applied_valid_from_s'],valid,rtol=0,atol=1e-12))
        expanded=np.repeat(reconstructed,10,axis=0)
        check(arm+' all4000 controls rebuilt independently',np.array_equal(expanded,data[arm]['co']['ctrl']) and np.array_equal(expanded,data[arm]['ne']['ctrl']) and np.array_equal(expanded,data[arm]['tr']['ctrl'][1:]))
        decoded_json=load_json(out/arm/'DECODED_COMMANDS.json');applied_json=load_json(out/arm/'COMMAND_APPLICATIONS.json')
        check(arm+' JSON dense command agreement',len(decoded_json)==400 and len(applied_json)==400 and np.array_equal(np.asarray([r['command'] for r in decoded_json]),decoded) and np.array_equal(np.asarray([r['command'] for r in applied_json]),applied))
        check(arm+' JSON dense timestamp agreement',np.array_equal(np.asarray([r['valid_from_s'] for r in decoded_json]),trace['valid_from_s']) and np.array_equal(np.asarray([r['valid_from_s'] for r in applied_json]),trace['applied_valid_from_s']) and np.array_equal(np.asarray([r['source_interval'] for r in applied_json]),trace['applied_source_index']))
        violation=trace['applied_start_s']+1e-11<valid
        check(arm+' expected valid_from violations',int(violation.sum())==(400 if arm=='legacy' else 0),violations=int(violation.sum()))
        if arm=='causal_carry':
            check('causal final decoded preserved pending', 'pending_final' in trace and np.array_equal(trace['pending_final'],decoded[-1]))
            check('causal final decoded not selected during400ms',np.all(trace['applied_source_index']<399) and int(trace['applied_source_index'][-1])==398 and trace['source_end_s'][-1]<=.4+1e-11)
            check('causal final pending has no actual application',bool(trace['pending_at_capture_end'][-1]) and int(np.count_nonzero(trace['pending_at_capture_end']))==1 and np.isnan(trace['actual_applied_start_s'][-1]) and np.isnan(trace['actual_applied_end_s'][-1]) and decoded_json[-1]['actual_applied_start_s'] is None and decoded_json[-1]['actual_applied_end_s'] is None)
            check('causal pending planned after capture',abs(float(trace['planned_applied_start_s'][-1])-.4)<1e-12 and abs(float(trace['planned_applied_end_s'][-1])-.401)<1e-12)
        metrics[arm]['timing_violations']=int(violation.sum())
        metrics[arm]['last_applied_source_index']=int(source_index[-1])
        metrics[arm]['last_pending_decoded_preserved']=arm=='causal_carry'
        metrics[arm]['decoded_interval_command_vs_other_note']='pending control may numerically equal applied control; source indices, not inequality, establish non-application'



if __name__=='__main__':
    raise SystemExit(main())
