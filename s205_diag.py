"""S205 Part B: instrumented 0.05 s arm that stores per-window external current, per-cell spike counts
and the pool cells' membrane potential, so the zero-spike pools can be diagnosed as
no-current / driven-but-subthreshold / driven-and-gated / fired.
"""
import json
import importlib
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype')
RUN = Path('/home/sensen/Desktop/engine修复验收/s205-unresolved-mn-and-silent-pools/20261025T075517207961Z')
S154 = Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
DT = 2.5e-5
CMD_EVERY = 200
NT_MS = 0.5
NT_STEPS = CMD_EVERY // 20
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')

out = RUN / 'diag'
assert not out.exists()
out.mkdir(parents=True)
import mujoco
sys.path.insert(0, str(ROOT))
L = importlib.import_module('run_flybody_loop')
from engine.embodied import adapters as ad
from engine.embodied.motor_output import MotorOutput, MotorOutputConfig, build_motor_manifest
from engine.embodied.sensory_channels import (ChannelSensoryEncoder, ChannelConfig, ChannelQuantities,
                                              channel_index_map)

body = L.build_body('canonical')
model = body['model']
model.opt.timestep = DT
z = np.load(S154 / 'init80ms/TRAJECTORY.npz')
q0 = z['qpos'][0].copy()
act0 = z['act'][0].copy()
joints = {}
for leg in LEGS:
    ids = [j for j in range(model.njnt)
           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower().startswith('joint_' + leg.lower())]
    tib = [j for j in ids if 'tibia' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or '').lower()]
    joints[leg] = dict(q=[int(model.jnt_qposadr[j]) for j in ids],
                       tib_q=int(model.jnt_qposadr[tib[0]]), tib_v=int(model.jnt_dofadr[tib[0]]))

tier_cfg = ad.TierConfig(background_mean_nA=0, background_sd_nA=0)
built = ad.build_neural_tier(tier_cfg)
tier = ad.NeuralTier(built, tier_cfg, ad.OwnershipLedger())
n = int(built['n'])
groups, _ = channel_index_map(n, built['tier']['root_ids'])
names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
pools = json.loads((ROOT / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
manifest = build_motor_manifest(pools, built['tier']['root_ids'], names)
dec = MotorOutput(manifest, MotorOutputConfig())
tier.reset()
enc = ChannelSensoryEncoder(groups, n, ChannelConfig())
enc.set_reference_pose(q0)
for leg in LEGS:
    j = joints[leg]
    enc.assign(leg, j['tib_q'], j['tib_q'], j['q'])
threshold_mV = float(tier.net.p.threshold_mV)

pool_cells = sorted({int(c) for r in manifest['rows'] for c in r['local_indices']})
d = mujoco.MjData(model)
d.qpos[:] = q0
d.qvel[:] = 0.0
d.act[:] = act0
d.ctrl[:] = 0.0
mujoco.mj_forward(model, d)

ncmd = 10
cur_hist = np.zeros((ncmd, n), np.float32)
count_hist = np.zeros((ncmd, n), np.uint8)
v_hist = np.zeros((ncmd, len(pool_cells), NT_STEPS), np.float32)
spike_hist = np.zeros((ncmd, NT_STEPS, n), bool)
ctrl_hist = np.zeros((ncmd, model.nu))
for k in range(ncmd):
    q = {}
    for leg in LEGS:
        j = joints[leg]
        q[leg.lower()] = ChannelQuantities(
            tibia_joint_dev_rad=float(d.qpos[j['tib_q']] - q0[j['tib_q']]),
            tibia_joint_vel_rad_s=float(d.qvel[j['tib_v']]),
            multi_joint_dev_rad=float(np.mean(d.qpos[j['q']] - q0[j['q']])),
            contact_force_norm=0.0, foot_contact=0.0)
    current, _ = enc.encode(q)
    cur_hist[k] = current
    for sc in range(NT_STEPS):
        sp = tier.net.step(tier.background_nA + current)
        spike_hist[k, sc] = sp
        count_hist[k] += sp.astype(np.uint8)
        v_hist[k, :, sc] = tier.net.v[pool_cells]
    pend, _ = dec.consume(spike_hist[k], spike_dt_s=NT_MS / 1000.0,
                          start_s=k * 0.005, end_s=(k + 1) * 0.005)
    d.ctrl[:] = np.asarray(pend, float)
    ctrl_hist[k] = d.ctrl.copy()
    for _j in range(CMD_EVERY):
        mujoco.mj_step(model, d)

np.savez_compressed(out / 'DIAG.npz', current=cur_hist, counts=count_hist, v=v_hist,
                    pool_cells=np.asarray(pool_cells), ctrl=ctrl_hist, threshold_mV=threshold_mV,
                    spikes_packed=np.packbits(spike_hist.reshape(ncmd, -1), axis=1), n_neurons=n)
(out / 'SUMMARY.json').write_text(json.dumps(dict(
    steps=ncmd * CMD_EVERY, duration_s=ncmd * CMD_EVERY * DT, rate_full=100.0, threshold_mV=threshold_mV,
    n_pool_cells=len(pool_cells), ncmd=ncmd,
    spikes_per_window=[int(c.sum()) for c in count_hist]), indent=2))
print(json.dumps(dict(threshold_mV=threshold_mV, pool_cells=len(pool_cells),
                      spikes=[int(c.sum()) for c in count_hist])))
