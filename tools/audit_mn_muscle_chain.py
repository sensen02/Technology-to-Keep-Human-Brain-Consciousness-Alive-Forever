"""Audit existing MN->muscle records; static observations only, no integrations."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf
ROOT = Path(__file__).resolve().parents[1]
SRC = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def records(path):
    """Stream objects in a JSON array without retaining 114MB sensory records."""
    decoder = json.JSONDecoder()
    with path.open() as f:
        buf = f.read(1048576).lstrip()
        assert buf[0] == '['
        buf = buf[1:]
        while True:
            buf = buf.lstrip()
            if buf.startswith(','):
                buf = buf[1:].lstrip()
            if buf.startswith(']'):
                return
            try:
                obj, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                more = f.read(1048576)
                if not more:
                    raise
                buf += more
                continue
            yield obj
            buf = buf[end:]


def dump(path, obj):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert, allow_nan=False))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path)
    out = ap.parse_args().out
    assert (out/'PREREGISTRATION.json').exists()
    assert not (out/'AUDIT.json').exists(), 'new evidence directory required'
    files = [SRC/n for n in ('MODEL.xml','MAPPING.json','PREREGISTRATION.json','NEURAL.npz',
                            'NEURAL_RECORDS.json','NEURAL_BOUNDARY.npz','BODY.npz','HANDOVER_AFTER.npz')]
    files += [Path(__file__), ROOT/'engine/embodied/motor_output.py', ROOT/'engine/embodied/continuous_handover.py']
    hashes = {str(p):sha(p) for p in files}
    assert hashes[str(SRC/'MODEL.xml')] == '3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
    prereg = json.loads((SRC/'PREREGISTRATION.json').read_text())
    source_match = {name:sha(Path(name)) == h for name,h in prereg['source_hashes'].items()
                    if name.endswith(('motor_output.py','continuous_handover.py'))}
    assert all(source_match.values()), 'inspect historical decoder implementation before replay'
    mapping = json.loads((SRC/'MAPPING.json').read_text()); rows = mapping['rows']
    groups = [np.asarray(r['local_indices'],dtype=int) for r in rows]
    names = [r['actuator'] for r in rows]
    with np.load(SRC/'NEURAL.npz') as z: counts=z['spike_counts'].copy(); commands=z['commands'].copy()
    with np.load(SRC/'NEURAL_BOUNDARY.npz') as z: rates=z['decoder_rates'].copy(); pending=z['pending'].copy()
    assert counts.shape == (700,12000) and np.all((counts>=0)&(counts<=2))
    decay = np.exp(-.0005/.005)
    rate_series=[]; lower=[]; upper=[]; pool_counts=[]; next_commands=[]
    bound_violation=cmd_error=0.; motor_indices=sorted({i for g in groups for i in g})
    motor_total=0
    for k,rec in enumerate(records(SRC/'NEURAL_RECORDS.json')):
        assert rec['interval']==k and rec['phase']=='neural_only'
        assert not rec['assistance_kept_during_neural']
        assert rec['motor_output_connected'] and rec['external_force_max']==0 and rec['auxiliary_force_max']==0
        c=counts[k]
        pc=np.array([c[g].sum() if len(g) else 0 for g in groups])
        assert np.array_equal(pc,rec['decoder']['pool_spike_counts'])
        assert c.sum()==rec['total_network_spikes']
        assert c[motor_indices].sum()==rec['motor_spikes']
        motor_total+=rec['motor_spikes']
        applied=np.asarray(rec['applied_command'])
        cmd_error=max(cmd_error,float(np.max(np.abs(applied-pending))),float(np.max(np.abs(applied-commands[k]))))
        # A count=2 neuron fired both ticks; count=1 could be either tick.
        twos=np.array([(c[g]==2).sum()/len(g) if len(g) else 0. for g in groups])
        ones=np.array([(c[g]==1).sum()/len(g) if len(g) else 0. for g in groups])
        lo=decay**2*rates+(1-decay)/.0005*(twos*(decay+1)+ones*decay)
        hi=decay**2*rates+(1-decay)/.0005*(twos*(decay+1)+ones)
        actual=np.asarray(rec['decoder']['rates_hz'])
        bound_violation=max(bound_violation,float(np.maximum(lo-actual,actual-hi).max()))
        nextcmd=np.asarray(rec['decoded_next_command'])
        cmd_error=max(cmd_error,float(np.max(np.abs(np.clip(actual/100,0,1)-nextcmd))))
        assert rec['applied_source_interval']==1099+k
        rates=actual; pending=nextcmd
        rate_series.append(actual); lower.append(lo); upper.append(hi); pool_counts.append(pc); next_commands.append(nextcmd)
    assert len(rate_series)==700 and cmd_error<=1e-12 and bound_violation<=1e-9
    rate_series=np.asarray(rate_series); pool_counts=np.asarray(pool_counts)
    with np.load(SRC/'BODY.npz') as z: body={k:z[k].copy() for k in z.files}
    t0=float(np.load(SRC/'HANDOVER_AFTER.npz')['time'])
    bi=np.flatnonzero(body['time']>=t0-1e-10)
    rel=body['time'][bi]-t0
    ticks=np.rint(rel/.0005).astype(int)
    ci=np.clip((ticks-1)//2,0,699)
    body_cmd_error=float(np.max(np.abs(body['command'][bi]-commands[ci])))
    assert body_cmd_error<=1e-12
    assert not body['assist'][bi].any() and not body['external'][bi].any()
    per_actuator=[]
    for j,r in enumerate(rows):
        entry={k:r[k] for k in ('actuator','actuator_index','status','pool_root_ids','tier_root_ids','local_indices')}
        entry['windows']={}
        for label,end in [('early15ms',.015),('all700ms',.700)]:
            ni=int(round(end/.001)); bsel=bi[rel<=end+1e-10]
            entry['windows'][label]=dict(pool_spike_count=int(pool_counts[:ni,j].sum()),
                max_rate_hz=float(rate_series[:ni,j].max()), mean_applied_ctrl=float(commands[:ni,j].mean()),
                saturated_applied_intervals=int((commands[:ni,j]>=1).sum()),
                max_activation=float(body['activation'][bsel,j].max()),
                mean_abs_total_muscle_force=float(np.abs(body['muscle_force'][bsel,j]).mean()))
        per_actuator.append(entry)
    model=_load_mjcf(str(SRC/'MODEL.xml')).compile()
    assert (model.nq,model.nv,model.nu,model.nmesh)==(49,48,90,71)
    assert [model.actuator(j).name for j in range(model.nu)]==names
    assert np.array_equal(model.actuator_actadr,np.arange(90))
    snapshots=[]; mats=[]; active=[]; bias=[]; capacities=[]; actual_torque=[]
    force_error=torque_error=0.
    for ms in (0,5,15):
        i=int(np.argmin(abs(body['time']-(t0+ms/1000))))
        d=mujoco.MjData(model)
        d.qpos[:]=body['qpos'][i]; d.qvel[:]=body['qvel'][i]; d.ctrl[:]=body['command'][i]
        d.act[:]=body['activation'][i]; d.time=body['time'][i]
        mujoco.mj_forward(model,d)
        f=d.actuator_force.copy(); qf=d.qfrc_actuator.copy()
        force_error=max(force_error,float(np.abs(f-body['muscle_force'][i]).max()))
        mat=np.zeros((90,48))
        for j in range(90):
            start=int(d.moment_rowadr[j]); nn=int(d.moment_rownnz[j])
            mat[j,d.moment_colind[start:start+nn]]=d.actuator_moment[start:start+nn]
        torque_error=max(torque_error,float(np.abs(mat.T@f-qf).max()))
        d.act[:]=0; mujoco.mj_forward(model,d); f0=d.actuator_force.copy()
        d.act[:]=1; mujoco.mj_forward(model,d); gain=d.actuator_force.copy()-f0
        af=f-f0; terms=mat*af[:,None]; cap=mat*gain[:,None]
        snapshots.append(dict(relative_ms=ms,body_sample=i,absolute_s=body['time'][i],
            active_torque_norm=float(np.linalg.norm(terms.sum(0))),
            active_force_abs_sum=float(np.abs(af).sum()),
            passive_force_abs_sum=float(np.abs(f0).sum()),
            capacity_positive=cap.clip(0).sum(0),capacity_negative=cap.clip(max=0).sum(0),
            loaded_feet=int((body['foot_force'][i,:,2]>0).sum()),
            foot_fz=body['foot_force'][i,:,2],nonfoot_count=int(body['nonfoot_count'][i])))
        mats.append(mat); active.append(af); bias.append(f0); capacities.append(cap); actual_torque.append(terms)
    assert force_error<=1e-8 and torque_error<=1e-8
    np.savez_compressed(out/'CHAIN_ARRAYS.npz',rate_hz=rate_series,rate_lower=lower,rate_upper=upper,
        pool_counts=pool_counts,applied_command=commands,decoded_next=next_commands,
        body_sample_indices=bi,body_command_interval=ci,moment=mats,active_force=active,
        passive_force=bias,unit_activation_torque=capacities,actual_active_torque=actual_torque)
    selected=np.flatnonzero(np.max(commands,axis=0)>1e-6)
    summary=dict(model_sha256=hashes[str(SRC/'MODEL.xml')],n_neural_windows=700,
        mapped_actuator_rows=sum(bool(len(g)) for g in groups),unique_motor_cells=len(motor_indices),
        nonzero_command_actuators_gt1e6=int(len(selected)),motor_spikes= motor_total,
        command_error=cmd_error,body_command_error=body_cmd_error,rate_bound_violation=bound_violation,
        max_saved_force_error=force_error,max_torque_reconstruction_error=torque_error,
        retained_command_on_first_window_norm=float(np.linalg.norm(commands[0])),
        decoded_peak_rate_hz=float(rate_series.max()),
        active_arrays_exclude_passive_muscle_force=True,
        model_load=1,forward=9,body_step=0,neural_step=0,
        unresolved_actuator_names=[r['actuator'] for r in rows if not r['local_indices']],
        note='Pool counts double count shared pools across muscle rows; motor_spikes is unique-neuron count. Substep timing unavailable, bounds only. No gait success.')
    assert hashes=={str(p):sha(p) for p in files}
    dump(out/'AUDIT.json',dict(summary=summary,input_sha256=hashes,source_code_matches=source_match,
        per_actuator=per_actuator,static_snapshots=snapshots,
        joint_names=[model.joint(int(model.dof_jntid[k])).name for k in range(model.nv)]))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
