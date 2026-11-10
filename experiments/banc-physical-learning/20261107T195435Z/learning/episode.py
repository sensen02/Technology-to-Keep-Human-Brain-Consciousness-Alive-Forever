"""Fullbody checkpoint episode runner; physical execution gated by optimizer CLI.

One graph/model per persistent worker. Actor receives only MN filtered Hz.
Reward reads privileged body state separately and never affects within-episode action.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import numpy as np
import mujoco as mj
from scoring import score_episode,heading_from_points,WEIGHT

HERE=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
FOUNDATION=HERE.parent/'foundation'
OUTPUT=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning')
OLD=ROOT/'experiments/banc-local-learning/20261107T185910Z'
CHECKPOINT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop')
PLANT=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
sys.path[:0]=[str(OLD/'encoder'),str(PLANT),str(ROOT/'tools')]
from encoder import SensoryEncoder
from incremental_actuator import IncrementalActuator,DEFAULT_MODEL_PATH
from plant_probe import Plant
spec=importlib.util.spec_from_file_location('_physical_learning_foundation',FOUNDATION/'runtime.py')
foundation=importlib.util.module_from_spec(spec);sys.modules[spec.name]=foundation;spec.loader.exec_module(foundation)


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


class PhysicalInvalid(Exception):
    """Only preregistered numerical nonfinite/warning physical trial failures."""


class EpisodeRunner:
    def __init__(self):
        self.plant=Plant();self.m=self.plant.m
        self.encoder=SensoryEncoder();self.circuit=foundation.load_circuit()
        self.handover=load(CHECKPOINT/'HANDOVER.npz')
        self.source_initial=load(CHECKPOINT/'INITIAL.npz')
        self.assist_native=load(CHECKPOINT/'NATIVE.npz')['rows']
        assist=self.assist_native[(self.assist_native[:,0]>=.8)&(self.assist_native[:,0]<1.)]
        self.assist=dict(foot_support_mean=float(assist[:,1:7].sum(axis=1).mean()/WEIGHT),com_z_mean=float(assist[:,10].mean()))
        del self.assist_native
        self.head=self.m.body('Head').id
        self.zero=np.zeros(20000)
        self.spec=mj.mjtState.mjSTATE_INTEGRATION
        self.source_hashes={str(p):sha(p) for p in (Path(__file__),HERE/'scoring.py',HERE/'search_rules.py',HERE/'PREREG.json',FOUNDATION/'runtime.py',FOUNDATION/'BASE.npz',FOUNDATION/'NEURAL_START.npz',CHECKPOINT/'HANDOVER.npz',CHECKPOINT/'INITIAL.npz',CHECKPOINT/'NATIVE.npz',DEFAULT_MODEL_PATH,OLD/'encoder/encoder.py')}
        if self.m.opt.timestep!=.0000125:raise ValueError('Original native timestep changed')
        if not np.array_equal(self.encoder.root_ids,self.circuit.root_ids):raise ValueError('Encoder/circuit identity mismatch')
        if not np.isin(self.encoder.local_indices,self.circuit.allowed_sensory_local).all():raise ValueError('Unqualified sensory port')

    def material_slip(self,d):
        p=self.plant;m=self.m;loads=np.zeros(6);weighted=np.zeros(6)
        for contact_index,contact in enumerate(d.contact):
            ga,gb=int(contact.geom1),int(contact.geom2)
            if not(p.planes[ga] or p.planes[gb]):continue
            geom=gb if p.planes[ga] else ga;leg=p.feet[geom]
            if leg<0:continue
            force=np.zeros(6);mj.mj_contactForce(m,d,contact_index,force)
            fz=max(0.,float((contact.frame.reshape(3,3).T@force[:3])[2])*(1 if p.planes[ga] else -1))
            jac=np.zeros((3,m.nv));mj.mj_jac(m,d,jac,np.zeros_like(jac),contact.pos,int(m.geom_bodyid[geom]))
            loads[leg]+=fz;weighted[leg]+=fz*np.linalg.norm((jac@d.qvel)[:2])
        return np.r_[d.time,loads,weighted]

    def run(self,task):
        folder=OUTPUT/task['relative_path']
        if folder.exists():raise FileExistsError('Never overwrite trial: '+str(folder))
        folder.mkdir(parents=True)
        theta=np.asarray(task['theta'],float).reshape(6,4,7)
        horizon=float(task['duration_s']);final=bool(task['final_validation']);ticks=int(round(horizon/.005));expected_steps=ticks*400
        p=self.plant;m=self.m;c=self.circuit;started=time.monotonic();physical_steps=0;forward_calls=0;neural_before=c.net.steps
        native=np.empty((expected_steps,20));body=[];commands=[];policies=[];slip=[];loads5=[];rates5=[]
        finite=True;failure=None;warning=False;software_exception=False;contact_rows=0;saturated=0;voltage_min=float('inf');voltage_max=-float('inf')
        d=None;actuator=None;actor=None
        try:
            # Episode boundary only: full integrationstate restore at actualtime1s.
            d=mj.MjData(m);mj.mj_setState(m,d,self.handover['physics_state'],self.spec);mj.mj_forward(m,d);forward_calls+=1
            foundation.restore_circuit(c)
            neural_before=c.net.steps
            if abs(d.time-1)>1e-8 or c.net.t_ms!=1000.:raise ValueError('Canonical clock mismatch')
            actuator=IncrementalActuator(m,d,kp=p.limits/.2*128,kd=p.limits*.005*np.sqrt(128),max_rate_rad_s=30.)
            actuator.target_rad=self.handover['actuator_target'].copy()
            actuator.initial_target_rad=actuator.target_rad.copy()
            actor=foundation.Actor(theta)
            if not np.array_equal(actor.joint_names,[m.joint(int(j)).name for j in p.jids]):raise ValueError('Actor actual42 order mismatch')
            if not np.array_equal(actor.mn_root_ids,c.root_ids[c.mn_local]):raise ValueError('Actor MN identity mismatch')
            initial=np.zeros(mj.mj_stateSize(m,self.spec));mj.mj_getState(m,d,initial,self.spec)
            heading=heading_from_points(d.xipos[self.head],d.xipos[p.thorax]);start_com=d.subtree_com[p.thorax].copy()
            np.savez_compressed(folder/'INITIAL.npz',state=initial,target=actuator.target_rad,kp=actuator.kp,kd=actuator.kd,limits=actuator.torque_limit,qa=p.qa,va=p.va,joint_bounds=m.jnt_range[p.jids])
            np.savez_compressed(folder/'NEURAL_INITIAL.npz',**foundation.snapshot(c))
            np.savez_compressed(folder/'ACTOR.npz',theta=theta,weights=actor.weights,mn_root_ids=actor.mn_root_ids,mn_local=actor.mn_local,mn_leg=actor.mn_leg,joint_names=actor.joint_names,basis_sha256=sha(FOUNDATION/'BASE.npz'))
            if final:mj.mj_saveModel(m,str(folder/'MODEL.mjb'),None)
            protocol=dict(condition=task['condition'],episode_id=task['episode_id'],evaluation_seed=int(task['evaluation_seed']),duration_s=horizon,start_time_s=float(d.time),physics_dt=m.opt.timestep,command_dt=.001,policy_dt=.005,slip_dt=.0005 if final else .005,checkpoint_episode=True,warmup_in_this_trace=False,previous_warmup_source=str(CHECKPOINT),controller='qualified556sensory->fixed20kconductancecore->MN-onlysameleglowrankactor->finitePD128',actor_inputs='MN323filteredHz only; no body/root/time/teacher/target',reward_inputs='privileged physicalbody evaluation only; no withintrial feedback into actor',assist=self.assist,initial_heading_xy=heading.tolist(),initial_COM=start_com.tolist(),source_hashes=self.source_hashes,body_source=str(DEFAULT_MODEL_PATH),body_source_sha256=sha(DEFAULT_MODEL_PATH),native_retention='all native20fields retained for this trial',run_source_sha256=sha(__file__),final_validation=final)
            dump(folder/'PROTOCOL.json',protocol)
            def state():return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),qacc_warmstart=d.qacc_warmstart.copy(),com=d.subtree_com[p.thorax].copy())
            body.append(state());initial_contact=p.contact(d)
            for tick in range(ticks):
                mn=c.filtered_rates_hz[c.mn_local].copy();rate=actor.action(mn)
                foot=initial_contact[1:7].copy() if physical_steps==0 else native[physical_steps-1,1:7].copy()
                q=d.qpos[p.qa].copy();vel=d.qvel[p.va].copy();current=self.encoder.encode(q,vel,foot)
                policies.append(dict(time_s=float(d.time),mn_rates_hz=mn,predicted_rate=rate.copy(),local_physical_input=np.r_[q,vel,foot],sensory_current_nA=current[self.encoder.local_indices].copy(),mn_voltages_mV=c.net.v[c.mn_local].copy(),teacher_warmup=False))
                rates5.append(rate.copy())
                for offset in range(5):
                    command_time=float(d.time);actuator.update(rate,.001);commands.append(np.r_[command_time,rate,actuator.target_rad])
                    for sub in range(80):
                        d.ctrl[:]=actuator.torque(d);saturated+=int(np.count_nonzero(np.abs(d.ctrl)>=actuator.torque_limit-1e-12))
                        if d.qfrc_applied.any() or d.xfrc_applied.any():raise RuntimeError('Unexpected external applied force')
                        mj.mj_step(m,d);physical_steps+=1;mj.mj_forward(m,d);forward_calls+=1
                        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():finite=False;raise PhysicalInvalid('Nonfinite physical state')
                        if d.warning.number.any():warning=True;raise PhysicalInvalid('MuJoCo warning')
                        native[physical_steps-1]=p.contact(d);contact_rows=physical_steps
                        if final and sub%40==39:slip.append(self.material_slip(d))
                    body.append(state())
                if not final:slip.append(self.material_slip(d))
                loads5.append(native[physical_steps-1,1:7].copy())
                c.advance(current,self.zero,5.)
                voltage_min=min(voltage_min,float(c.net.v.min()));voltage_max=max(voltage_max,float(c.net.v.max()))
                if abs(c.net.t_ms/1000-d.time)>1e-8:raise RuntimeError('Physical/neural clock mismatch')
            score=score_episode(native[:physical_steps],np.asarray(slip),np.asarray(loads5),np.asarray(rates5),start_com,d.subtree_com[p.thorax].copy(),heading,self.assist,.0005 if final else .005)
        except Exception as exc:
            software_exception=not isinstance(exc,PhysicalInvalid)
            failure=traceback.format_exc();finite=finite and (d is not None and np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            score=dict(reward=None,feasible=False,physical_gate={'passed':False,'reason':'incomplete or invalid episode'},reward_terms={},duration_s=physical_steps*.0000125)
        # Save actual partial data on failure; failed current step may have invalid contact.
        native_rows=native[:contact_rows].copy()
        np.savez_compressed(folder/'NATIVE.npz',rows=native_rows)
        np.savez_compressed(folder/'COMMANDS.npz',rows=np.asarray(commands))
        np.savez_compressed(folder/'SLIP.npz',rows=np.asarray(slip))
        np.savez_compressed(folder/'SCORE_INPUTS.npz',loads5ms=np.asarray(loads5),rates5ms=np.asarray(rates5),start_com=start_com if 'start_com' in locals() else np.zeros(3),end_com=d.subtree_com[p.thorax].copy() if d is not None else np.zeros(3),heading=heading if 'heading' in locals() else np.zeros(2))
        if body:np.savez_compressed(folder/'BODY.npz',**{key:np.asarray([b[key] for b in body]) for key in body[0]})
        if policies:np.savez_compressed(folder/'POLICY.npz',**{key:np.asarray([r[key] for r in policies]) for key in policies[0]})
        np.savez_compressed(folder/'NEURAL_FINAL.npz',**foundation.snapshot(c))
        valid=failure is None and finite and not warning and physical_steps==expected_steps
        result=dict(status='COMPLETE_CHECKPOINT_EPISODE_NOT_WALKING' if valid else 'FAILED_EPISODE_PRESERVED',valid=valid,**score,episode_id=task['episode_id'],condition=task['condition'],theta=theta.tolist(),evaluation_seed=int(task['evaluation_seed']),native_steps=physical_steps,live_forward_calls=forward_calls,neural_steps=int(c.net.steps-neural_before),neural_end_steps=int(c.net.steps),neural_end_time_ms=float(c.net.t_ms),sim_s=float(d.time) if d is not None else None,release_s=physical_steps*.0000125,finite=bool(finite),warnings=d.warning.number.tolist() if d is not None else [],failure=failure,software_exception=software_exception,contact_rows_saved=contact_rows,neural_voltage_range_mV=[voltage_min,voltage_max] if np.isfinite(voltage_min) else None,MN_spikes_since_checkpoint=int((c.net.spike_count-load(FOUNDATION/'NEURAL_START.npz')['spike_count'])[c.mn_local].sum()),torque_saturated_entries=saturated,torque_total_entries=physical_steps*42,wall_s=time.monotonic()-started,folder=str(folder),files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        dump(folder/'RESULT.json',result)
        if software_exception:
            raise RuntimeError('Software exception requires study abort; preserved '+str(folder/'RESULT.json')+'\n'+failure)
        return dict(episode_id=task['episode_id'],folder=str(folder),valid=valid,feasible=bool(score['feasible']) and valid,reward=score['reward'],native_steps=physical_steps,neural_steps=result['neural_steps'],live_forward_calls=forward_calls,theta=theta.tolist(),result_sha256=sha(folder/'RESULT.json'),wall_s=result['wall_s'],failure=failure)


_WORKER=None

def initialize_worker():
    global _WORKER
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1' or os.environ.get('OMP_NUM_THREADS')!='1':raise RuntimeError('Persistentworkers require singlethread BLAS/OMP')
    _WORKER=EpisodeRunner()

def run_task(task):
    if _WORKER is None:raise RuntimeError('Worker not initialized')
    return _WORKER.run(task)
