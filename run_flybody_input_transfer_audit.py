"""Physical observation -> exact sensory roots -> one-step membrane transfer.
手工研究原型，不是成品. No connectivity propagation or standing claim.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
import run_flybody_loop as loop
from run_sensory_tuning_annotation_audit import sha, exact_groups

HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs/model_branches/flybody_loop/flybody_input_transfer_audit.json'


def audit():
    pointer = HERE/'outputs/neural_muscle_closure/current_run.json'
    p = json.loads(pointer.read_text())
    bundle = pointer.parent/p['path']
    manifest_path = bundle/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    identity_path = bundle/'neural_identity.json'
    assert sha(manifest_path) == p['manifest_sha256']
    assert sha(identity_path) == manifest['files']['neural_identity.json']['sha256']
    identity = json.loads(identity_path.read_text())
    lookup = HERE/'outputs/neural_rebuild/sensory_channel_lookup.json'
    cells = exact_groups(identity, json.loads(lookup.read_text()))
    assert len(cells) == 3788
    groups = identity['sensory_groups']
    n = len(identity['tier_root_ids'])
    body = loop.build_body('flybody_servo')
    body['reference'] = body['seated'].copy()
    model = body['model']
    cfg = loop.closure.ChannelConfig()
    encoder = loop.closure.ChannelSensoryEncoder(groups, n, cfg)
    records = []
    for leg in loop.LEGS:
        key = leg.lower()
        for probe, channel, roles, expected in [
            ('coxa_position', 'hairplate_angle', {'hairplate': 'coxa'}, .125/cfg.reference_angle_rad*cfg.current_nA_per_unit),
            ('tibia_velocity', 'hook_direction', {'claw': 'tibia'}, 1.234/cfg.reference_velocity_rad_s*cfg.current_nA_per_unit)]:
            data = mujoco.MjData(model)
            data.qpos[:] = body['reference']
            mujoco.mj_forward(model, data)
            baseline, _ = encoder.encode(loop.observe(body, data, body['weight_per_leg'], roles))
            joint = body['role_hinge'][key]['coxa' if probe == 'coxa_position' else 'tibia']
            jid = model.joint(joint).id
            if probe == 'coxa_position':
                data.qpos[int(model.jnt_qposadr[jid])] += .125
            else:
                data.qvel[int(model.jnt_dofadr[jid])] = 1.234
            mujoco.mj_forward(model, data)
            current, _ = encoder.encode(loop.observe(body, data, body['weight_per_leg'], roles))
            ids = np.asarray(groups[key][channel], int)
            delta = current-baseline
            assert np.allclose(delta[ids], expected)
            # Isolate only the measured target-channel delta for a membrane equation probe.
            isolated = np.zeros(n)
            isolated[ids] = delta[ids]
            net0 = loop.ConductanceNetwork(n, loop.ConductanceParams(), dt_ms=.5)
            net1 = loop.ConductanceNetwork(n, loop.ConductanceParams(), dt_ms=.5)
            sp0 = net0.step(current_nA=np.zeros(n))
            sp1 = net1.step(current_nA=isolated)
            dv = net1.v-net0.v
            outside = np.ones(n, bool)
            outside[ids] = False
            assert np.all(dv[ids] > 0)
            assert np.all(dv[outside] == 0)
            records.append({'leg': key, 'probe': probe, 'channel': channel, 'roles': roles,
                            'target_cells': len(ids), 'target_root_ids': [identity['tier_root_ids'][int(i)] for i in ids],
                            'expected_delta_current_nA_per_cell': expected,
                            'measured_delta_current_nA_per_cell': float(delta[ids][0]),
                            'mean_one_step_delta_voltage_mV': float(dv[ids].mean()),
                            'target_spikes_after_one_step': int(sp1[ids].sum()),
                            'baseline_spikes_after_one_step': int(sp0[ids].sum()),
                            'outside_target_voltage_changed': int(np.count_nonzero(dv[outside])),
                            'observation_current_summary': loop.channel_current_summary(current, groups)})
    result = {'label': '手工研究原型，不是成品', 'status': 'PASS_INPUT_AND_MEMBRANE_TRANSFER_NOT_CONNECTOME_PROPAGATION',
              'n_neurons': n, 'sensory_roots_verified': len(cells), 'probes': records,
              'limitations': ['Single-step target-current isolation with NO connectivity; tests membrane input acceptance only.',
                             'Uses exact runtime sensory roots, but does not test connectome propagation, motor spikes or standing.',
                             'Coxa role is a diagnostic mapping, not validated for all hairplates.',
                             'No new gain, polarity, synapse or motor input was introduced.'],
              'sha256': {str(x): sha(x) for x in [identity_path, manifest_path, lookup, Path(loop.__file__), Path(__file__)]}}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'probes': len(records), 'output': str(OUT)}, indent=2))
    return result


if __name__ == '__main__':
    audit()
