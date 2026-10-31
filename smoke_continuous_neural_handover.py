"""Live 10ms API smoke on a restored FAILED warmup state; NOT walking evidence.

Important auxiliary integration test. Restores recorded state once for a standalone
smoke; actual warmup+handover runner must pass its SAME live MjData without restore.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import mujoco
from flygym.compose.fly.musculoskeletal import _load_mjcf
import run_neural_muscle_closure as c
from engine.embodied.runtime_pool_guard import load_runtime_pool
from engine.embodied.continuous_handover import run_neural_continuous,state_snapshot
from run_integrated_repaired_closure import identities
from run_one_second_motor_pair import dump
P=Path(__file__).parent
SOURCE=Path('/home/sensen/Desktop/engine修复验收/s5-physical-warmup/20261028T051000Z')
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-neural-handover-smoke/20261028T054000Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 sources=[SOURCE/'MODEL.xml',SOURCE/'HANDOVER_STATE.npz',SOURCE/'BODY.npz',Path(__file__),P/'engine/embodied/continuous_handover.py',c.POOLS]
 hashes={str(p):sha(p) for p in sources}
 dump(OUT/'PREREGISTRATION.json',dict(scope='10ms live API smoke restored from FAILED warmup state. Does not meet walking goal; not continuous preceding warmup execution.',duration_s=.01,source_hashes=hashes,seed=0,criteria='nonzero qvel preserved; exact boundary qpos/qvel/act/time/warmstart; .15 to .16 time; no applied auxiliary/external force; first causal command interval zero'))
 counts={'model_load':0,'forward':0,'body_step':0,'neural_step':0,'network_instances':0};error=None;start=time.time();rows=[]
 try:
  m=_load_mjcf(str(SOURCE/'MODEL.xml')).compile();counts['model_load']+=1;d=mujoco.MjData(m)
  z=np.load(SOURCE/'HANDOVER_STATE.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];d.act[:]=z['activation'];d.ctrl[:]=z['command'];d.time=float(z['time']);d.qacc_warmstart[:]=z['qacc_warmstart'];d.qfrc_applied[:]=z['assist'];mujoco.mj_forward(m,d);counts['forward']+=1
  # Restore saved integrator warmstart AFTER derived-field refresh to be explicit.
  d.qacc_warmstart[:]=z['qacc_warmstart'];assert np.any(d.qvel)
  ref=np.load(SOURCE/'BODY.npz')['qpos'][0].copy();joints,geom=identities(m)
  np.random.seed(0);built=c.build_neural_tier(c.TierConfig(background_mean_nA=0,background_sd_nA=0));manifest=c.build_motor_manifest(load_runtime_pool(c.POOLS),built['tier']['root_ids'],[m.actuator(i).name for i in range(m.nu)]);groups,_=c.channel_index_map(int(built['n']),built['tier']['root_ids'])
  np.savez_compressed(OUT/'BEGIN.npz',**state_snapshot(d));dump(OUT/'MAPPING.json',manifest)
  def observe(model,data):rows.append(state_snapshot(data))
  result=run_neural_continuous(m,d,built,manifest,groups,.01,sensory_reference_qpos=ref,joints=joints,geom_leg=geom,body_observer=observe,retain_currents=True)
  counts.update(network_instances=result['counts']['network_instances'],body_step=result['counts']['body_steps'],neural_step=result['counts']['neural_steps']);counts['forward']+=result['counts']['forward']
  np.savez_compressed(OUT/'BOUNDARY_BEFORE.npz',**result['handover_before']);np.savez_compressed(OUT/'BOUNDARY_AFTER.npz',**result['handover_after_withdrawal']);np.savez_compressed(OUT/'END.npz',**state_snapshot(d));np.savez_compressed(OUT/'BODY.npz',**{key:np.array([row[key] for row in rows]) for key in rows[0]});np.savez_compressed(OUT/'NEURAL.npz',commands=result['commands'],spike_counts=result['spike_counts'],sensory_current=np.array([r['sensory_current_nA'] for r in result['records']]))
  dump(OUT/'RECORDS.json',result['records'])
  assert result['state_continuity_exact'] and counts['body_step']==800 and counts['neural_step']==20
  assert abs(d.time-float(z['time'])-.01)<1e-10
  assert not result['commands'][0].any()
  assert all(not row['qfrc_applied'].any() and not row['xfrc_applied'].any() for row in rows)
  dump(OUT/'VERIFICATION.json',dict(passed=True,physical_samples=len(rows),boundary_exact=True,nonzero_transferred_qvel_max=float(abs(z['qvel']).max()),activation_max_at_handover=float(abs(z['activation']).max()),absolute_start_s=float(z['time']),absolute_end_s=d.time,auxiliary_force_max=0.,external_force_max=0.,warnings=int(sum(w.number for w in d.warning)),motor_spikes=sum(r['motor_spikes'] for r in result['records']),walking_certified=False,scope='LIVE API SMOKE ONLY; source warmup FAILED and restored from disk, not same-execution walking warmup'))
 except BaseException as exc:
  error={'type':type(exc).__name__,'message':str(exc)};raise
 finally:
  dump(OUT/'EXECUTION.json',dict(counts=counts,error=error,exit_code=0 if error is None else 1,runtime_s=time.time()-start))
  dump(OUT/'SOURCE_UNCHANGED.json',dict(passed=all(sha(Path(k))==v for k,v in hashes.items())))
  (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes());(OUT/'continuous_handover.py').write_bytes((P/'engine/embodied/continuous_handover.py').read_bytes())
 print('LIVE_HANDOVER_API_SMOKE_PASS',OUT,flush=True)
if __name__=='__main__':main()
