"""Identity-explicit archived neural replay; no physical model or controller."""
import argparse, hashlib, json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.sparse import save_npz
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.embodied.adapters import TierConfig, build_neural_tier
from engine.neural_cond import ConductanceNetwork, ConductanceParams
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig
SRC=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()


def dump(p,a):
    p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
    assert (out/'PREREGISTRATION.json').exists() and not (out/'RESULT.json').exists()
    cfg=TierConfig(background_mean_nA=0.,background_sd_nA=0.)
    inputs=[SRC/n for n in ['NEURAL_BOUNDARY.npz','NEURAL.npz','HANDOVER.json','MAPPING.json','PREREGISTRATION.json']]
    inputs += [Path(__file__),ROOT/'engine/neural_cond.py',ROOT/'engine/embodied/motor_output.py',ROOT/'engine/embodied/continuous_handover.py',ROOT/'engine/embodied/adapters.py',ROOT/cfg.data_path]
    hashes={str(p):sha(p) for p in inputs}
    prior=json.loads((SRC/'PREREGISTRATION.json').read_text())
    for n in ['engine/neural_cond.py','engine/embodied/motor_output.py','engine/embodied/continuous_handover.py']:
        assert hashes[str(ROOT/n)]==prior['source_hashes'][str(ROOT/n)]
    hand=json.loads((SRC/'HANDOVER.json').read_text());origin=float(hand['neural_origin_s'])
    mapping=json.loads((SRC/'MAPPING.json').read_text())
    with np.load(SRC/'NEURAL.npz') as z: currents=z['sensory_current_nA'];expected=z['spike_counts'];commands=z['commands']
    with np.load(SRC/'NEURAL_BOUNDARY.npz') as z: boundary={k:z[k].copy() for k in z.files}
    built=build_neural_tier(cfg); roots=np.asarray(built['data']['root_ids'])[built['global_index']]
    assert len(roots)==12000
    for row in mapping['rows']:
        assert [int(roots[i]) for i in row['local_indices']]==row['tier_root_ids']
    motor=np.asarray(sorted({i for row in mapping['rows'] for i in row['local_indices']}),dtype=int)
    We=built['edges']['We']*.2;Wi=built['edges']['Wi']*.2
    save_npz(out/'ACTUAL_We.npz',We);save_npz(out/'ACTUAL_Wi.npz',Wi)
    np.savez_compressed(out/'IDENTITIES.npz',root_ids=roots,motor_indices=motor,global_index=built['global_index'])
    params=ConductanceParams();start_steps=round(origin/.0005)
    assert start_steps==2200 and currents.shape==(700,12000) and expected.shape==currents.shape
    dump(out/'IDENTITY.json',dict(input_sha256=hashes,tier_config=asdict(cfg),tier_fingerprint=built['fingerprint'],
        conductance_params=asdict(params),weight_multiplier=.2,full_actual_weights_saved=True,
        inferred_boundary_scalars=dict(steps=start_steps,t_ms=origin*1000,ring_i=start_steps%5),
        scalar_inference='source-derived, must pass replay gate; not claimed as originally saved',
        motor_cells=len(motor),n_neurons=len(roots),matrix_nnz=dict(exc=We.nnz,inh=Wi.nnz)))

    def restore():
        net=ConductanceNetwork(12000,params,dt_ms=.5,threshold_offset_mV=boundary['threshold_offset_mV'])
        net.set_connectivity(We,Wi)
        for key in ['_ring','v','ge','gi','adaptation','refractory_until_ms','spike_count']:
            assert getattr(net,key).shape==boundary[key].shape
            getattr(net,key)[:]=boundary[key]
        net.steps=start_steps;net.t_ms=origin*1000;net._ring_i=start_steps%len(net._ring)
        dec=MotorOutput(mapping,MotorOutputConfig())
        dec.rates[:]=boundary['decoder_rates'];dec.elapsed_s=origin;dec.interval=1099
        return net,dec,boundary['pending'].copy()

    def run(label,drive,expected_counts=None):
        net,dec,pending=restore();steps=len(drive)*2
        spikes=np.zeros((steps,12000),bool);applied=[];decoded=[];probe=[];mismatch=None;cmd_error=0.
        for k,I in enumerate(drive):
            applied.append(pending.copy())
            if expected_counts is not None: cmd_error=max(cmd_error,float(np.abs(pending-commands[k]).max()))
            for sub in range(2):
                j=2*k+sub
                if j<30:
                    probe.append(dict(time_from_release_ms=j*.5,
                        voltage=net.v[motor].copy(),ge=net.ge[motor].copy(),gi=net.gi[motor].copy(),
                        adaptation=net.adaptation[motor].copy(),current=I[motor].copy(),
                        refractory=(net.t_ms+1e-10<net.refractory_until_ms[motor]),
                        gap=params.threshold_mV+net.threshold_offset_mV[motor]-net.v[motor]))
                spikes[j]=net.step(I)
            obs=spikes[2*k:2*k+2].sum(0)
            if expected_counts is not None and not np.array_equal(obs,expected_counts[k]) and mismatch is None:
                ids=np.flatnonzero(obs!=expected_counts[k])
                mismatch=dict(interval=k,differing_cells=len(ids),first_local_indices=ids[:20],
                              first_root_ids=roots[ids[:20]],actual=obs[ids[:20]],expected=expected_counts[k,ids[:20]])
            pending,trace=dec.consume(spikes[2*k:2*k+2],spike_dt_s=.0005,start_s=origin+k*.001,end_s=origin+(k+1)*.001)
            decoded.append(pending.copy())
        np.savez_compressed(out/(label+'.npz'),spikes_packed=np.packbits(spikes,axis=1),
            n_neurons=np.array(12000),applied=applied,decoded_next=decoded,
            motor_voltage=[p['voltage'] for p in probe],motor_ge=[p['ge'] for p in probe],
            motor_gi=[p['gi'] for p in probe],motor_adaptation=[p['adaptation'] for p in probe],
            motor_current=[p['current'] for p in probe],motor_refractory=[p['refractory'] for p in probe],motor_gap=[p['gap'] for p in probe])
        result=dict(neural_steps=steps,first_mismatch=mismatch,max_applied_command_error=cmd_error,
            motor_spikes_5ms=int(spikes[:10,motor].sum()),motor_spikes_15ms=int(spikes[:30,motor].sum()),
            total_motor_spikes=int(spikes[:,motor].sum()),first15_motor_cells_firing=int(spikes[:30,motor].any(0).sum()),
            first15_total_network_spikes=int(spikes[:30].sum()),
            first15_sensory_current_nonzero_cells=int(np.any(drive[:15]!=0,axis=0).sum()),
            initial_motor_voltage_range=[float(netv) for netv in [probe[0]['voltage'].min(),probe[0]['voltage'].max()]],
            initial_motor_gap_range=[float(v) for v in [probe[0]['gap'].min(),probe[0]['gap'].max()]],
            initial_motor_refractory_count=int(probe[0]['refractory'].sum()))
        return result,spikes[:30].copy()

    baseline,base_sp=run('BASELINE_REPLAY',currents,expected)
    gate=baseline['first_mismatch'] is None and baseline['max_applied_command_error']<=1e-12
    results={'baseline':baseline}
    if gate:
        zero,zsp=run('ZERO_SENSORY',np.zeros_like(currents[:15]))
        hold,hsp=run('HOLD_FIRST_SENSORY',np.repeat(currents[:1],15,axis=0))
        zero['spike_bits_different_from_baseline_first15']=int(np.count_nonzero(zsp!=base_sp))
        hold['spike_bits_different_from_baseline_first15']=int(np.count_nonzero(hsp!=base_sp))
        results.update(zero_sensory=zero,hold_first_sensory=hold)
    assert hashes=={str(p):sha(p) for p in inputs}
    result=dict(gate_passed=gate,results=results,body_steps=0,
        total_neural_steps=sum(r['neural_steps'] for r in results.values()),
        limitations='Saved-current neural-only replay, not a closed-loop body intervention. Full historical substep spikes absent; aggregate reproduction does not uniquely prove their timing.')
    dump(out/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()))
    return 0 if gate else 2


if __name__=='__main__':
    sys.exit(main())
