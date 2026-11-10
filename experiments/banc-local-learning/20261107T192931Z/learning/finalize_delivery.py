"""Summarize fixed comparisons and mirror scoped learning artifacts only."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
DEST=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/learning')
CASES=('oldinstant','oldhistory','newinstant','newhistory')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}


def main():
    verified=json.loads((HERE/'INDEPENDENT_VERIFICATION.json').read_text())
    results={case:json.loads((HERE/'cases'/case/'RESULT.json').read_text()) for case in CASES}
    pilot=json.loads((HERE/'PILOT_VERIFICATION.json').read_text()) if (HERE/'PILOT_VERIFICATION.json').exists() else None
    comparison=dict(primary_fixed_before_results='newhistory',source_quality='failed slow4 sameepisode temporaldiagnostics only',cases={case:dict(metrics=r['metrics'],primary=r['primary'],fit_solves=r['fit_solves'],temporal_better_than_zero=r['temporal_better_than_zero']) for case,r in results.items()},independent_verification_checks=verified['checks_passed'],total_new_fit_solves=sum(r['fit_solves'] for r in results.values()),new_neural_steps=0 if pilot is None else pilot['actual_neural_replay_steps'],new_physical_steps=0,biological_calibration='NOT_ESTABLISHED')
    (HERE/'COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    ix=np.arange(4);width=.34
    axes[0].bar(ix-width/2,[results[c]['metrics']['train_RMS_rad_s'] for c in CASES],width,label='Training')
    axes[0].bar(ix+width/2,[results[c]['metrics']['temporal_RMS_rad_s'] for c in CASES],width,label='Temporal sameepisode')
    axes[0].axhline(results['newhistory']['metrics']['zero_temporal_RMS_rad_s'],color='red',ls='--',label='Temporal zero-output baseline')
    axes[0].set_xticks(ix,CASES);axes[0].set(ylabel='RMS applied-rate error (rad/s)',title='Primary fixed before results: newhistory; no case beats temporal zero baseline')
    axes[0].legend(fontsize=9)
    trace=load(HERE/'cases/newhistory/PREDICTIONS.npz')
    axes[1].plot(trace['time_s'],trace['target_rate'][:,6],lw=1,label='Failed-reference applied LF knee mean rate')
    axes[1].plot(trace['time_s'],trace['predicted_rate'][:,6],lw=1,label='Fixed newhistory prediction')
    axes[1].axvline(6,color='black',ls='--',label='Temporal diagnostic begins')
    axes[1].set(xlabel='Saved time (s)',ylabel='Rate (rad/s)');axes[1].legend(fontsize=9)
    fig.suptitle('Mechanism/history engineering comparison | not walking or biological calibration')
    fig.savefig(HERE/'CASE_COMPARISON.png',dpi=160);plt.close(fig)
    table='\n'.join('| '+case+' | '+str(results[case]['fit_solves'])+' | '+str(results[case]['eligible_weight_count'])+' | '+f"{results[case]['metrics']['train_RMS_rad_s']:.9f}"+' | '+f"{results[case]['metrics']['temporal_RMS_rad_s']:.9f}"+' |' for case in CASES)
    pilot_text='Parent physical execution and subsequent saved neural/history verification are pending; no physical or neural replay result is claimed by this report yet.' if pilot is None else (
        f"Parent produced the learned pilot; this task independently replayed its saved neural/input/history trace: **{pilot['status']}**, {pilot['checks_passed']} checks, {pilot['actual_neural_replay_steps']} neural steps and zero physical steps. "
        f"Exact local inputs/encoder currents, start MN rates/volts, causal features, handover and full neural/predictor final states match; matrix predictions allow1e-10 accumulation tolerance. "
        f"Voltage minimum {pilot['voltage_range_mV'][0]:.12g}mV, -90mV sanity pass={pilot['voltage_minus90_sanity_pass']}; biological calibration remains NOT_ESTABLISHED. "
        "Old185910Z hold INITIAL/body/dt/warmup-command contract matches exactly and its postwarm commands arezero; it is a historical matched body-only control, not a new-adaptation neuralhold experiment. Parent retains native physical replay and physical interpretation.")
    report=f'''# Fixed four-case mechanism and causal MN-history comparison

Primary **newhistory** was fixed in preregistration before fitting; it was not selected on temporal results. All four cases fail to improve on the temporal zero-output baseline **3.661845726 rad/s**. The same failed slow4 source remains a failed reference, and the temporal split is part of the same episode, not independent-animal generalization. This work serves S3 local learning/S4 interface discipline.

| Case | New fits | Eligible coefficients | Training RMS rad/s | Temporal RMS rad/s |
|---|---:|---:|---:|---:|
{table}

Oldinstant weights are exactly cached from185910Z and were not refitted. The other three cases each use42 fixed boundedridge fits: **126 new solves**, lambda1, bounds[-1,1], tol1e-9, maxiter100, no bias. No neural steps were used to generate learning features here; both old and new traces were reused. No physical execution was performed by this agent.

The new dynamics trace was frozen by its owning agent at SHA256 `8a73c8a221b0f752e7c53f9e129fcf9747c42dc3ead6209a7ce912f430e70847`; module SHA256 `c5106d46a62bedd31faccdbe981764106c23c4dc491890e6e4d21ef531718250`. Mechanism: adaptation conductance with EA−80mV, spike increment .002/15uS and tau100ms, replacing the old subtractive adaptation current. Neuron identities, original source edge masks, old556 encoder and zeroDN drive remain unchanged. The old voltage sanity failures are preserved historical evidence; the new mechanism is an engineering hypothesis, not physiology calibration.

## Causal runtime and identities

Implementation: `{HERE/'runtime.py'}`. `Readout(model_path).observe(mn_rates323)` appends the current START-bin MNfilteredHz; `.action()` returns `30*tanh(feature@W)` in actual42 joint order. Feature order is lagmajor,323 canonical MN within each block, /100Hz. History lags are[0,1,4,10] bins, corresponding to0,5,20,50ms. Pre-start inputs are exactzero and there is no future wrapping. Every output joint uses only same-leg MN features. Model roots/local indices/leg labels and all42 joint names are checked against frozen source files at initialization; malformed identities, crossleg weights or nonzero bias are rejected.

The runtime keeps11 samples for history models and1 forinstant. `.state_dict()` returns history_rates_hz, observations_seen, lags_bins and bin_ms as numeric arrays; `.load_state_dict()` validates and restores them. Fresh or fully silent history produces exactzero action. Immediatezero current input with nonzero past history can retain a remembered action. A last nonzero sample at t0 is still eligible at exactaget50ms and absent by t55ms; no feature refers to data older than50ms. Parent was informed of this endpoint convention before results.

Primary model: `{HERE/'cases/newhistory/MODEL.npz'}`; SHA256 `{verified['case_model_sha256']['newhistory']}`. Desktop equivalent: `{DEST/'cases/newhistory/MODEL.npz'}`. Models include MNroot/local/leg/joint identity, lags, scales, source hashes and fixedprimary metadata.

## Independent verification and runner

`{HERE/'verify_cases.py'}` does not import the fitter, neural module, optimization library or physical runner. **{verified['checks_passed']} checks passed**, rebuilding lag features by explicit per-row past indexing; comparing full1817-sample runtime features/actions; checking all model identities, same-leg masks, bounds, zero bias, exactsource labels, causal prefix invariance, stationary inputs, impulse/drain behavior, invalid inputs and source hashes. Ridge objectives/KKT were independently recomputed without optimization. Largest projected residual across all cases is {max(x['max_projected_residual'] for x in verified['cases']):.12g}, below preregistered1e-6. There was no performance-based refit or primary substitution. Temporary invalid-model fixtures were removed immediately after checks.

Physical runner: `{HERE/'run_closed_loop.py'}`. It was written and syntax-checked, then handed to parent; this agent never executes it. Default is fixednewhistory learnedcontrol, optional `--control hold` remains available but is not launched automatically. It preserves original sourceINITIAL/fullbody/nativephysics/PD128/caps and1s warmup+2s posthandover schedule. START MN rates enter readout history before action, history survives handover; physical sensory currents are held and advanced neurally after the same physical5ms interval. HANDOVER stores `neural_g_adapt_uS` and `predictor_*` state; NEURAL_FINAL stores explicit `g_adapt_uS`; READOUT_FINAL preserves predictor history. No adaptationconductance is mislabeled as nA. Old BODY/NATIVE/COMMANDS/POLICY schemas are preserved with extra readout feature/history metadata.

{pilot_text}

Runner reproduction (parent scope only):

```bash
PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python {HERE/'run_closed_loop.py'} --control learned
```

Independent fit/runtime verification:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python {HERE/'verify_cases.py'} --phase all
```

Completed outputs are protected against accidental overwrite by fit/verifier entry points. Numerical plot `{DEST/'CASE_COMPARISON.png'}` compares the four fixed fits and the primary LF knee trace; the actual generated image was opened and inspected, with scope recorded in VISUAL_REVIEW.json. All sources/models remain engineering artifacts; no stable walking, causal support dependence or complete brain claim is made. Only this learning subtree and its Desktop mirror were modified; tracker remains parent-owned.
'''
    (HERE/'REPORT.md').write_text(report)
    names=['PREREG.json','SOURCE_FREEZE.json','PILOT_REPLAY_PREREG.json','runtime.py','fit_cases.py','verify_cases.py','run_closed_loop.py','verify_pilot.py','finalize_delivery.py','INDEPENDENT_VERIFICATION.json','COMPARISON.json','REPORT.md','CASE_COMPARISON.png']
    for case in CASES:
        names.extend('cases/'+case+'/'+name for name in ('MODEL.npz','PREDICTIONS.npz','RESULT.json'))
    if pilot is not None:names.append('PILOT_VERIFICATION.json')
    for optional in ('EXECUTION_RECORD.json','VISUAL_REVIEW.json'):
        if (HERE/optional).exists():names.append(optional)
    records=[]
    for name in names:
        source,dest=HERE/name,DEST/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        assert sha(source)==sha(dest)
        records.append(dict(source=str(source),desktop=str(dest),sha256=sha(source),bytes=source.stat().st_size,byte_equal=True))
    delivery=dict(status='PASS_BYTE_IDENTICAL',files=records,files_including_record=len(records)+1,total_new_fit_solves=126,new_neural_steps=0 if pilot is None else pilot['actual_neural_replay_steps'],new_physical_steps=0,record_note='No recursive self hash; delivery record copied and compared byte for byte.')
    text=json.dumps(delivery,indent=2,ensure_ascii=False)+'\n'
    for folder in (HERE,DEST):(folder/'DELIVERY_VERIFICATION.json').write_text(text)
    assert sha(HERE/'DELIVERY_VERIFICATION.json')==sha(DEST/'DELIVERY_VERIFICATION.json')
    print(json.dumps({k:delivery[k] for k in ('status','files_including_record','total_new_fit_solves','new_neural_steps','new_physical_steps')}))

if __name__=='__main__':main()
