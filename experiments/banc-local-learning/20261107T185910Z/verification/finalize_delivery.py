"""Create verification report/plot and mirror only this task's outputs."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
DEST=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/verification')
DATA=DEST.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}

def main():
    readout=json.loads((HERE/'VERIFY_READOUT.json').read_text())
    closed=json.loads((HERE/'VERIFY_CLOSED_LOOP.json').read_text()) if (HERE/'VERIFY_CLOSED_LOOP.json').exists() else None
    hold=json.loads((HERE/'VERIFY_OUTPUT_HOLD.json').read_text()) if (HERE/'VERIFY_OUTPUT_HOLD.json').exists() else None
    trace=load(DATA/'readout/TRAIN_TRACE.npz')
    joint_names=load(DATA/'readout/MODEL.npz')['joint_names']
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(3,1,figsize=(11,9),constrained_layout=True)
    t=trace['time_s'];axes[0].plot(t,trace['target_rate'][:,6],label='Failed-reference LF knee applied mean rate',lw=1)
    axes[0].plot(t,trace['predicted_rate'][:,6],label='MN-only learned prediction',lw=1)
    axes[0].axvline(6.,c='black',ls='--',label='Temporal diagnostic begins')
    axes[0].set(ylabel='Rate (rad/s)',title=str(joint_names[6])+'; same episode temporal split, not animal generalization');axes[0].legend(fontsize=8)
    axes[1].plot(t,trace['mn_rates_hz'].mean(axis=1),label='Mean of 323 MN filtered rates')
    axes[1].set(ylabel='Filtered rate (Hz)');axes[1].legend(fontsize=8)
    axes[2].plot(t,trace['mn_voltages_mV'].min(axis=1),label='Minimum recorded MN START voltage')
    axes[2].axhline(-90,c='red',ls='--',label='Preregistered sanity floor (-90mV)')
    axes[2].set(xlabel='Saved physical time (s)',ylabel='Voltage (mV)');axes[2].legend(fontsize=8)
    fig.suptitle('Independent numerical replay | CALIBRATION_FAILED | no walking acceptance')
    fig.savefig(HERE/'READOUT_VERIFICATION.png',dpi=160)
    plt.close(fig)
    m=readout['metrics']
    rows='\n'.join('| '+key+' | '+str(value)+' |' for key,value in m.items())
    closed_text='Closed-loop neural/input verification has not yet run; no result is claimed.' if closed is None else (
        f"Closed-loop replay: `{closed['status']}`, {closed['samples']} policy samples, {closed['actual_neural_replay_steps']} additional neural steps, "
        f"{closed['checks_passed']} checks. Full saved final voltage, excitatory/inhibitory conductance, adaptation, refractory state, delay ring/index, spike counts, filtered rates and step count compare exactly. "
        f"Saved handover neural fields compare exactly. Minimum interval-end voltage {closed['voltage_range_mV'][0]:.12g} mV; -90mV sanity passes: {closed['voltage_minus90_sanity_pass']}. "
        "The parent owns independent native physical replay; this verifier performed no physical steps.")
    if hold is not None:
        closed_text += (f"\n\nOutput-hold control: `{hold['status']}`, {hold['samples']} samples / {hold['actual_neural_replay_steps']} neural steps / {hold['checks_passed']} checks. "
                       f"Qualified inputs, start MN states, predicted readout, saved handover and full final neural state compare exactly. Applied commands after1s are exactly zero while neural processing continues. "
                       f"Voltage minimum {hold['voltage_range_mV'][0]:.12g} mV also fails -90mV sanity. Total verification replay steps including readout and both pilots: "
                       f"{readout['actual_neural_replay_steps']+closed['actual_neural_replay_steps']+hold['actual_neural_replay_steps']}; no new physical steps or learning.")
    content=f'''# Independent saved readout and neural-input verification

Result: **{readout['status']}**. Independent readout verification passed {readout['checks_passed']} checks and replayed all {readout['samples']} samples / {readout['actual_neural_replay_steps']} neural steps. No new learning, optimization, physical steps, MuJoCo calls or tracked-file edits occurred.

Source/data provenance, exact q42/qvel42/footload rows, five-row applied-rate label means, qualified encoder currents, start-of-interval MN rates and MN voltages were checked. Recorded sensory currents were then held for each 5ms neural interval with zero descending current, total conductance .02uS and dt .5ms. The verifier imports frozen LocalCircuit and SensoryEncoder but never imports the trainer or physical runner. This is deterministic implementation replay, not a second independently implemented neural integrator; separate dynamics verification remains the source for the latter.

The same-leg-only 323x42 weight mask, [-1,1] coefficient bounds, exact zero bias, silent-MN zero action, actual42 names, root888 IDs, local indices and leg labels pass. Predictions, aggregate and per-joint RMS were independently recalculated. The constrained ridge objective uses lambda1; no fit was repeated. The maximum projected KKT gradient residual is {readout['max_projected_gradient_residual']:.12g}, below the preregistered 1e-6 bound. Raw active-set gradient residuals and each joint's objective are retained in VERIFY_READOUT.json.

| Metric | Recomputed rad/s |
|---|---:|
{rows}

Temporal prediction improves on zero: **{readout['temporal_better_than_zero']}**. The source is the failed slow4 reference and both splits belong to the same episode. These are engineering fit diagnostics, not independent-animal validation or successful movement labels.

The readout replay minimum interval-end voltage is **{readout['voltage_range_mV'][0]:.12g} mV**, which fails the original -90mV sanity threshold. Frozen configuration remains **CALIBRATION_FAILED**, using the preregistered .02 fixed fallback. Exact numerical replay does not establish calibrated physiology, stable walking or complete brain function. The plot shows recorded MN start voltages only; the reported full-network interval-end minimum is a separate measurement and must not be inferred from that plot alone.

{closed_text}

## Outputs and reproduction

Primary verifier: `{HERE/'verify_saved.py'}`.

Preregistration: `{HERE/'PREREG.json'}`. Readout result: `{HERE/'VERIFY_READOUT.json'}`. Replay-derived readout final state: `{HERE/'REPLAY_READOUT_FINAL.npz'}`. The trainer did not archive a full readout final state, so that snapshot is not claimed equal to an unavailable original.

Desktop report: `{DEST/'REPORT.md'}`. Plot: `{DEST/'READOUT_VERIFICATION.png'}`. Byte-identical delivery hashes: `{HERE/'DELIVERY_VERIFICATION.json'}`.

Reproduction command (writes protected fresh result paths; existing result prevents accidental overwrite):

```bash
PYTHONDONTWRITEBYTECODE=1 /run/media/sensen/Data2/cell_wound_prototype/venv_body/bin/python {HERE/'verify_saved.py'} readout
```

For the two pilot replays use `{HERE/'verify_pilots.py'}` with argument `closed_loop` or `output_hold` under the same interpreter. Scope extension was saved before replay in `{HERE/'PILOT_EXTENSION_PREREG.json'}`. Original MODEL/TRAIN_TRACE/POLICY/source artifacts are read-only to this task. System python3 lacked scipy during an initial dependency-only probe; the existing venv_body interpreter was used without installing packages. All three verification jobs exited0. The plotted readout figure was actually opened and inspected; it displays reference/prediction mismatch, sustained MN activity and the MN-only voltage trace with a separately labeled sanity threshold.
'''
    (HERE/'REPORT.md').write_text(content)
    DEST.mkdir(parents=True,exist_ok=True)
    names=['PREREG.json','verify_saved.py','finalize_delivery.py','VERIFY_READOUT.json','REPLAY_READOUT_FINAL.npz','REPORT.md','READOUT_VERIFICATION.png']
    if closed is not None:names.append('VERIFY_CLOSED_LOOP.json')
    if hold is not None:names.extend(['VERIFY_OUTPUT_HOLD.json','PILOT_EXTENSION_PREREG.json','verify_pilots.py'])
    records=[]
    for name in names:
        src,dst=HERE/name,DEST/name
        shutil.copy2(src,dst)
        assert sha(src)==sha(dst)
        records.append(dict(source=str(src),desktop=str(dst),sha256=sha(src),bytes=src.stat().st_size,byte_equal=True))
    record=dict(status='PASS_BYTE_IDENTICAL',files=records,files_including_delivery_record=len(records)+1,
                actual_neural_replay_steps=readout['actual_neural_replay_steps']+(0 if closed is None else closed['actual_neural_replay_steps'])+(0 if hold is None else hold['actual_neural_replay_steps']),new_physical_steps=0,new_learning_updates=0,
                note='Delivery record self hash omitted; direct byte equality checked after mirroring.')
    text=json.dumps(record,indent=2,ensure_ascii=False)+'\n'
    for folder in (HERE,DEST):(folder/'DELIVERY_VERIFICATION.json').write_text(text)
    assert sha(HERE/'DELIVERY_VERIFICATION.json')==sha(DEST/'DELIVERY_VERIFICATION.json')
    print(json.dumps({k:record[k] for k in ('status','files_including_delivery_record','actual_neural_replay_steps')}))

if __name__=='__main__':main()
