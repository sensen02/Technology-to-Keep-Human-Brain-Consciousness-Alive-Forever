"""Matched descending-neuron physical pair for one fixed saved Actor: DN 0 vs 0.05 nA.

WRITTEN BUT NOT EXECUTED by the diagnostic agent. It refuses to run without an
explicit --execute flag, and the default invocation only prints the plan.

Adapted from the previous fixed-study episode runner
/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/learning/episode.py
with exactly one functional difference: the descending current injected into the
frozen conductance core is named_current(0.0) or named_current(0.05) instead of a
hard zero. Everything else (canonical 1 s checkpoint, body/model, Actor, encoder,
core, 5 ms policy / 1 ms command / 12.5 us native timing, no teacher/reference/
root/pose write after initialization) is identical between the two conditions.

Actor choice is deterministic and never swapped silently: the saved nonzero
support-feasible candidate with the maximum training reward, tie-broken by trial
index. The zero incumbent can never be selected.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import numpy as np

if os.environ.get('MUJOCO_GL') is None:os.environ['MUJOCO_GL']='egl'
HERE=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
ROUND=HERE.parent
PREV=ROOT/'experiments/banc-physical-learning/20261107T195435Z'
FOUNDATION=PREV/'foundation'
PREV_LEARNING_WS=PREV/'learning'
PREV_LEARNING=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning')
TRAINING=PREV_LEARNING/'training'
CHECKPOINT=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop')
OLD=ROOT/'experiments/banc-local-learning/20261107T185910Z'
PLANT=ROOT/'experiments/s5-ann-incremental/20261107T010000Z'
DESCENDING=ROUND/'descending'
OUT=Path('/home/sensen/Desktop/engine修复验收/banc-descending-learning/20261107T203049Z/physical')
CONDITIONS=(('dn_zero',0.0),('dn_on',0.05))
HORIZON_S=2.0
NATIVE_DT=0.0000125
POLICY_DT=0.005
COMMAND_DT=0.001
SLIP_DT=0.0005
COMMAND_SHA256_EXPECTED='ba84bca2b7e3ee710c3412467db2a4eadcd839dd38d027d44241b7b9363b455c'
EXPECTED_TRIAL=17
EXPECTED_REWARD=-1.0144429917827316
SELECTION_RULE='among the 48 saved nonzero candidates in the 195435Z fixed physical search, keep those with RESULT.valid==True and RESULT.feasible==True (the preregistered support/score gate) and a nonzero theta; select the maximum saved training reward; tie-break by the smaller trial index; the zero incumbent trial_000 and the incumbent-control trials 013/026/039/052 are never eligible'


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


def load_command_module():
    path=DESCENDING/'command.py'
    digest=sha(path)
    if digest!=COMMAND_SHA256_EXPECTED:raise RuntimeError('Frozen descending command module changed: '+digest)
    spec=importlib.util.spec_from_file_location('_frozen_banc_descending_command',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module,digest


def select_actor():
    """Deterministic selection over saved scores only; no physics, no model load."""
    eligible=[]
    for generation in range(4):
        for slot in range(12):
            trial=1+13*generation+slot
            folder=TRAINING/f'trial_{trial:03d}'
            result=json.loads((folder/'RESULT.json').read_text())
            actor=load(folder/'ACTOR.npz')
            if not (result['valid'] and result['feasible']):continue
            if not np.any(actor['theta']):continue
            eligible.append(dict(trial=trial,generation=generation,folder=str(folder),reward=float(result['reward']),
                                 theta=np.asarray(actor['theta'],float).copy(),theta_max_abs=float(np.abs(actor['theta']).max()),
                                 result_sha256=sha(folder/'RESULT.json'),actor_sha256=sha(folder/'ACTOR.npz')))
    if not eligible:raise RuntimeError('No eligible nonzero support-feasible candidate')
    eligible.sort(key=lambda candidate:(-candidate['reward'],candidate['trial']))
    best=eligible[0]
    if best['trial']!=EXPECTED_TRIAL or abs(best['reward']-EXPECTED_REWARD)>1e-9:
        raise RuntimeError('Deterministic selection changed; refusing to silently swap Actors: trial '+str(best['trial'])
                           +' reward '+repr(best['reward']))
    return best,eligible


def build_plan(eligible,best,command_digest):
    ticks=int(round(HORIZON_S/POLICY_DT))
    return dict(status='PLAN_ONLY_NOT_EXECUTED',requires='--execute',
                selection_rule=SELECTION_RULE,eligible_candidates=len(eligible),
                selected=dict(trial=best['trial'],generation=best['generation'],reward=best['reward'],
                              theta_max_abs=best['theta_max_abs'],folder=best['folder'],
                              result_sha256=best['result_sha256'],actor_sha256=best['actor_sha256'],
                              actor_theta_sha256=hashlib.sha256(np.ascontiguousarray(best['theta'],dtype=float).tobytes()).hexdigest()),
                conditions=[dict(name=name,descending_amplitude_nA=amplitude) for name,amplitude in CONDITIONS],
                descending_command=dict(path=str(DESCENDING/'command.py'),sha256=command_digest,
                                        local_indices=[1414,13170],root_ids=[720575941433155799,720575941566493282],
                                        note='engineering probe only; laterality/homology/biological function NOT established'),
                timing=dict(policy_dt_s=POLICY_DT,command_dt_s=COMMAND_DT,native_dt_s=NATIVE_DT,slip_dt_s=SLIP_DT,
                            horizon_s=HORIZON_S,ticks=ticks,
                            native_steps_per_condition=int(ticks*POLICY_DT/NATIVE_DT),
                            forward_calls_per_condition=int(ticks*POLICY_DT/NATIVE_DT+2),
                            neural_advance_calls=ticks,neural_steps=int(ticks*5)),
                canonical_checkpoint=dict(path=str(CHECKPOINT),handover_sha256=sha(CHECKPOINT/'HANDOVER.npz'),
                                          initial_sha256=sha(CHECKPOINT/'INITIAL.npz'),
                                          physics_state_sha256=hashlib.sha256(np.ascontiguousarray(load(CHECKPOINT/'HANDOVER.npz')['physics_state']).tobytes()).hexdigest(),
                                          start_time_s=1.0,end_time_s=1.0+HORIZON_S),
                outputs={name:str(OUT/name) for name,_ in CONDITIONS},
                files_per_condition=['INITIAL.npz','MODEL.mjb','BODY.npz','NATIVE.npz','COMMANDS.npz','SLIP.npz',
                                     'RESULT.json','PROTOCOL.json','ACTOR.npz','NEURAL_INITIAL.npz','NEURAL_FINAL.npz',
                                     'POLICY.npz','DN_INPUT.npz'],
                physics_steps_if_executed=int(len(CONDITIONS)*ticks*POLICY_DT/NATIVE_DT),
                neural_steps_if_executed=int(len(CONDITIONS)*ticks*5),
                no_teacher_or_pose_write_after_init=True)


def run():
    import mujoco as mj
    sys.path[:0]=[str(OLD/'encoder'),str(PLANT),str(ROOT/'tools'),str(PREV_LEARNING_WS)]
    from encoder import SensoryEncoder
    from incremental_actuator import IncrementalActuator,DEFAULT_MODEL_PATH
    from plant_probe import Plant
    from scoring import score_episode,heading_from_points,WEIGHT
    spec=importlib.util.spec_from_file_location('_dn_pair_foundation',FOUNDATION/'runtime.py')
    foundation=importlib.util.module_from_spec(spec);sys.modules[spec.name]=foundation;spec.loader.exec_module(foundation)
    command,command_digest=load_command_module()
    best,eligible=select_actor()
    plan=build_plan(eligible,best,command_digest)
    for name,_ in CONDITIONS:
        if (OUT/name).exists():raise FileExistsError('Never overwrite a physical condition: '+str(OUT/name))
    (OUT).mkdir(parents=True,exist_ok=True)
    for name,_ in CONDITIONS:(OUT/name).mkdir()
    dump(OUT/'DN_PAIR_PLAN.json',plan)
    plant=Plant();m=plant.m;encoder=SensoryEncoder();circuit=foundation.load_circuit()
    handover=load(CHECKPOINT/'HANDOVER.npz');assist_native=load(CHECKPOINT/'NATIVE.npz')['rows']
    assist_rows=assist_native[(assist_native[:,0]>=.8)&(assist_native[:,0]<1.)]
    assist=dict(foot_support_mean=float(assist_rows[:,1:7].sum(axis=1).mean()/WEIGHT),com_z_mean=float(assist_rows[:,10].mean()))
    del assist_native,assist_rows
    head=m.body('Head').id
    state_spec=mj.mjtState.mjSTATE_INTEGRATION
    if m.opt.timestep!=NATIVE_DT:raise ValueError('Original native timestep changed')
    if not np.array_equal(encoder.root_ids,circuit.root_ids):raise ValueError('Encoder/circuit identity mismatch')
    if not np.isin(encoder.local_indices,circuit.allowed_sensory_local).all():raise ValueError('Unqualified sensory port')
    descending_local=np.asarray(circuit.descending_local)
    theta=np.asarray(best['theta'],float).reshape(168)
    source_hashes={str(p):sha(p) for p in (Path(__file__),FOUNDATION/'runtime.py',FOUNDATION/'BASE.npz',FOUNDATION/'NEURAL_START.npz',
                                          CHECKPOINT/'HANDOVER.npz',CHECKPOINT/'INITIAL.npz',CHECKPOINT/'NATIVE.npz',
                                          DEFAULT_MODEL_PATH,OLD/'encoder/encoder.py',DESCENDING/'command.py',
                                          Path(best['folder'])/'ACTOR.npz',Path(best['folder'])/'RESULT.json')}
    summaries=[]
    for condition,amplitude in CONDITIONS:
        folder=OUT/condition
        dn=command.named_current(amplitude)
        if dn.shape!=(20000,) or dn.dtype!=np.float64:raise ValueError('Descending current shape/dtype mismatch')
        nonzero_local=np.flatnonzero(dn!=0)
        if not set(nonzero_local.tolist())<={int(x) for x in command.LOCAL_INDICES}:raise ValueError('Descending vector outside declared ports')
        if np.any(dn[~np.isin(np.arange(20000),descending_local)]!=0):raise ValueError('Descending current outside circuit descending identities')
        started=time.monotonic();physical_steps=0;forward_calls=0;neural_before=int(circuit.net.steps)
        native=np.empty((int(round(HORIZON_S/POLICY_DT))*int(round(POLICY_DT/NATIVE_DT)),20))
        body=[];commands=[];policies=[];slip=[];loads5=[];rates5=[]
        finite=True;failure=None;warning=False;software_exception=False;contact_rows=0;saturated=0
        voltage_min=float('inf');voltage_max=-float('inf');d=None
        try:
            # Identical canonical initialization for both conditions.
            d=mj.MjData(m);mj.mj_setState(m,d,handover['physics_state'],state_spec);mj.mj_forward(m,d);forward_calls+=1
            foundation.restore_circuit(circuit)
            if abs(d.time-1)>1e-8 or circuit.net.t_ms!=1000.:raise ValueError('Canonical clock mismatch')
            actuator=IncrementalActuator(m,d,kp=plant.limits/.2*128,kd=plant.limits*.005*np.sqrt(128),max_rate_rad_s=30.)
            actuator.target_rad=handover['actuator_target'].copy();actuator.initial_target_rad=actuator.target_rad.copy()
            actor=foundation.Actor(theta)
            if not np.array_equal(actor.joint_names,[m.joint(int(j)).name for j in plant.jids]):raise ValueError('Actor actual42 order mismatch')
            if not np.array_equal(actor.mn_root_ids,circuit.root_ids[circuit.mn_local]):raise ValueError('Actor MN identity mismatch')
            initial=np.zeros(mj.mj_stateSize(m,state_spec));mj.mj_getState(m,d,initial,state_spec)
            heading=heading_from_points(d.xipos[head],d.xipos[plant.thorax]);start_com=d.subtree_com[plant.thorax].copy()
            np.savez_compressed(folder/'INITIAL.npz',state=initial,target=actuator.target_rad,kp=actuator.kp,kd=actuator.kd,
                                limits=actuator.torque_limit,qa=plant.qa,va=plant.va,joint_bounds=m.jnt_range[plant.jids])
            np.savez_compressed(folder/'NEURAL_INITIAL.npz',**foundation.snapshot(circuit))
            np.savez_compressed(folder/'ACTOR.npz',theta=actor.theta.reshape(6,4,7),weights=actor.weights,mn_root_ids=actor.mn_root_ids,
                                mn_local=actor.mn_local,mn_leg=actor.mn_leg,joint_names=actor.joint_names,basis_sha256=sha(FOUNDATION/'BASE.npz'))
            np.savez_compressed(folder/'DN_INPUT.npz',current=dn,nonzero_local_indices=nonzero_local,
                                nonzero_local_values=dn[nonzero_local],root_ids=np.asarray(command.ROOT_IDS,dtype=np.int64),
                                local_indices=np.asarray(command.LOCAL_INDICES,dtype=np.int64),descending_local=descending_local,
                                amplitude_nA=np.array(float(amplitude)),vector_sha256=np.array(hashlib.sha256(np.ascontiguousarray(dn).tobytes()).hexdigest()))
            mj.mj_saveModel(m,str(folder/'MODEL.mjb'),None)
            dump(folder/'PROTOCOL.json',dict(condition=condition,episode_id=condition+'_'+str(best['trial']),
                descending_amplitude_nA=float(amplitude),descending_root_ids=[int(x) for x in command.ROOT_IDS],
                descending_local_indices=[int(x) for x in command.LOCAL_INDICES],
                activation_semantics='constant current injected at the two named descending ports for the whole 2 s release; zero means the numeric no-injection control',
                paired_condition_difference='only the descending current vector differs between dn_zero and dn_on',
                actor_selection=dict(rule=SELECTION_RULE,trial=best['trial'],generation=best['generation'],reward=best['reward'],
                                     theta_max_abs=best['theta_max_abs'],saved_folder=best['folder'],
                                     result_sha256=best['result_sha256'],actor_sha256=best['actor_sha256'],
                                     eligible_candidates=len(eligible),not_the_zero_incumbent=True,not_an_accepted_trained_policy=True),
                duration_s=HORIZON_S,start_time_s=float(d.time),end_time_s=float(d.time)+HORIZON_S,
                physics_dt=NATIVE_DT,command_dt=COMMAND_DT,policy_dt=POLICY_DT,slip_dt=SLIP_DT,
                checkpoint_episode=True,warmup_in_this_trace=False,previous_warmup_source=str(CHECKPOINT),
                controller='qualified556sensory->fixed20kconductancecore(+named descending current)->MN-onlysameleglowrankactor->finitePD128',
                actor_inputs='MN323filteredHz only; no body/root/time/teacher/target/phase',
                nn_inputs='qualified556 sensory current + named descending current',
                reward_inputs='privileged physical body evaluation only; no within-trial feedback into the actor',
                assist=assist,initial_COM=start_com.tolist(),initial_heading_xy=heading.tolist(),
                source_hashes=source_hashes,body_source=str(DEFAULT_MODEL_PATH),body_source_sha256=sha(DEFAULT_MODEL_PATH),
                native_retention='all native20 fields retained',run_source_sha256=sha(__file__),
                no_teacher_reference_root_or_pose_write_after_initialization=True,
                biological_status='named DNp09 candidate pair injected as an engineering probe; laterality, homology and function NOT established'))
            def snapshot_body():return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),
                                            qacc_warmstart=d.qacc_warmstart.copy(),com=d.subtree_com[plant.thorax].copy())
            body.append(snapshot_body());initial_contact=plant.contact(d)
            ticks=int(round(HORIZON_S/POLICY_DT));expected_steps=ticks*int(round(POLICY_DT/NATIVE_DT))
            for tick in range(ticks):
                mn=circuit.filtered_rates_hz[circuit.mn_local].copy();rate=actor.action(mn)
                foot=initial_contact[1:7].copy() if physical_steps==0 else native[physical_steps-1,1:7].copy()
                q=d.qpos[plant.qa].copy();velocity=d.qvel[plant.va].copy();current=encoder.encode(q,velocity,foot)
                policies.append(dict(time_s=float(d.time),mn_rates_hz=mn,predicted_rate=rate.copy(),
                                     local_physical_input=np.r_[q,velocity,foot],
                                     sensory_current_nA=current[encoder.local_indices].copy(),
                                     mn_voltages_mV=circuit.net.v[circuit.mn_local].copy(),teacher_warmup=False))
                rates5.append(rate.copy())
                for offset in range(5):
                    command_time=float(d.time);actuator.update(rate,COMMAND_DT);commands.append(np.r_[command_time,rate,actuator.target_rad])
                    for sub in range(int(round(COMMAND_DT/NATIVE_DT))):
                        d.ctrl[:]=actuator.torque(d)
                        saturated+=int(np.count_nonzero(np.abs(d.ctrl)>=actuator.torque_limit-1e-12))
                        if d.qfrc_applied.any() or d.xfrc_applied.any():raise RuntimeError('Unexpected external applied force')
                        mj.mj_step(m,d);physical_steps+=1;mj.mj_forward(m,d);forward_calls+=1
                        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():finite=False;raise PhysicalInvalid('Nonfinite physical state')
                        if d.warning.number.any():warning=True;raise PhysicalInvalid('MuJoCo warning')
                        native[physical_steps-1]=plant.contact(d);contact_rows=physical_steps
                        if sub%int(round(SLIP_DT/NATIVE_DT))==int(round(SLIP_DT/NATIVE_DT))-1:slip.append(material_slip(m,plant,d))
                    body.append(snapshot_body())
                loads5.append(native[physical_steps-1,1:7].copy())
                circuit.advance(current,dn,5.)
                voltage_min=min(voltage_min,float(circuit.net.v.min()));voltage_max=max(voltage_max,float(circuit.net.v.max()))
                if abs(circuit.net.t_ms/1000-d.time)>1e-8:raise RuntimeError('Physical/neural clock mismatch')
            score=score_episode(native[:physical_steps],np.asarray(slip),np.asarray(loads5),np.asarray(rates5),
                                start_com,d.subtree_com[plant.thorax].copy(),heading,assist,SLIP_DT)
        except Exception as exc:
            software_exception=not isinstance(exc,PhysicalInvalid)
            failure=traceback.format_exc();finite=finite and (d is not None and np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
            score=dict(reward=None,feasible=False,physical_gate={'passed':False,'reason':'incomplete or invalid episode'},reward_terms={},duration_s=physical_steps*NATIVE_DT)
        np.savez_compressed(folder/'NATIVE.npz',rows=native[:contact_rows].copy())
        np.savez_compressed(folder/'COMMANDS.npz',rows=np.asarray(commands))
        np.savez_compressed(folder/'SLIP.npz',rows=np.asarray(slip))
        if body:np.savez_compressed(folder/'BODY.npz',**{key:np.asarray([b[key] for b in body]) for key in body[0]})
        if policies:np.savez_compressed(folder/'POLICY.npz',**{key:np.asarray([r[key] for r in policies]) for key in policies[0]})
        np.savez_compressed(folder/'NEURAL_FINAL.npz',**foundation.snapshot(circuit))
        valid=failure is None and finite and not warning and physical_steps==expected_steps
        result=dict(status='COMPLETE_MATCHED_DN_PAIR_EPISODE_NOT_BEHAVIOUR_EVIDENCE' if valid else 'FAILED_EPISODE_PRESERVED',valid=valid,
                    **score,episode_id=condition+'_'+str(best['trial']),condition=condition,descending_amplitude_nA=float(amplitude),
                    actor_trial=int(best['trial']),actor_theta_sha256=hashlib.sha256(np.ascontiguousarray(theta).tobytes()).hexdigest(),
                    dn_vector_sha256=hashlib.sha256(np.ascontiguousarray(dn).tobytes()).hexdigest(),
                    dn_root_ids=[int(x) for x in command.ROOT_IDS],dn_local_indices=[int(x) for x in command.LOCAL_INDICES],
                    evaluation_seed=20261107,native_steps=physical_steps,live_forward_calls=forward_calls,
                    neural_steps=int(circuit.net.steps-neural_before),neural_end_steps=int(circuit.net.steps),neural_end_time_ms=float(circuit.net.t_ms),
                    sim_s=float(d.time) if d is not None else None,release_s=physical_steps*NATIVE_DT,finite=bool(finite),
                    warnings=d.warning.number.tolist() if d is not None else [],failure=failure,software_exception=software_exception,
                    contact_rows_saved=contact_rows,neural_voltage_range_mV=[voltage_min,voltage_max] if np.isfinite(voltage_min) else None,
                    torque_saturated_entries=saturated,torque_total_entries=physical_steps*42,wall_s=time.monotonic()-started,
                    folder=str(folder),files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        dump(folder/'RESULT.json',result)
        summaries.append(dict(condition=condition,descending_amplitude_nA=float(amplitude),valid=valid,
                              native_steps=physical_steps,neural_steps=result['neural_steps'],forward_calls=forward_calls,
                              wall_s=result['wall_s'],folder=str(folder)))
        if software_exception:
            raise RuntimeError('Software exception requires abort; preserved '+str(folder/'RESULT.json')+'\n'+failure)
    # Matched-pair equality: only the descending vector may differ.
    def array_sha(path,key):
        a=np.ascontiguousarray(load(path)[key]);return hashlib.sha256(a.tobytes()).hexdigest()
    match=dict(status='CHECKED',initial_physics_state_identical=array_sha(OUT/'dn_zero/INITIAL.npz','state')==array_sha(OUT/'dn_on/INITIAL.npz','state'),
               initial_actuator_target_identical=array_sha(OUT/'dn_zero/INITIAL.npz','target')==array_sha(OUT/'dn_on/INITIAL.npz','target'),
               initial_neural_v_identical=array_sha(OUT/'dn_zero/NEURAL_INITIAL.npz','v')==array_sha(OUT/'dn_on/NEURAL_INITIAL.npz','v'),
               initial_neural_spike_count_identical=array_sha(OUT/'dn_zero/NEURAL_INITIAL.npz','spike_count')==array_sha(OUT/'dn_on/NEURAL_INITIAL.npz','spike_count'),
               actor_theta_identical=array_sha(OUT/'dn_zero/ACTOR.npz','theta')==array_sha(OUT/'dn_on/ACTOR.npz','theta'),
               actor_weights_identical=array_sha(OUT/'dn_zero/ACTOR.npz','weights')==array_sha(OUT/'dn_on/ACTOR.npz','weights'),
               dn_vectors_differ=array_sha(OUT/'dn_zero/DN_INPUT.npz','current')!=array_sha(OUT/'dn_on/DN_INPUT.npz','current'),
               dn_zero_vector_all_zero=bool(np.all(load(OUT/'dn_zero/DN_INPUT.npz')['current']==0)),
               dn_on_values=load(OUT/'dn_on/DN_INPUT.npz')['current'][np.asarray(command.LOCAL_INDICES)].tolist(),
               selection_rule=SELECTION_RULE,selected_trial=int(best['trial']),selected_reward=float(best['reward']),
               conditions=summaries,totals=dict(native_steps=int(sum(s['native_steps'] for s in summaries)),
                                                neural_steps=int(sum(s['neural_steps'] for s in summaries)),
                                                forward_calls=int(sum(s['forward_calls'] for s in summaries))),
               note='matched engineering pair; not walking, not a biological claim; the parent runs and independently replays it')
    if not all(match[k] for k in ('initial_physics_state_identical','initial_actuator_target_identical','initial_neural_v_identical','initial_neural_spike_count_identical','actor_theta_identical','actor_weights_identical','dn_vectors_differ','dn_zero_vector_all_zero')):
        match['status']='MISMATCH_INVESTIGATE'
    dump(OUT/'DN_PAIR_MATCH.json',match)
    print(json.dumps(match,ensure_ascii=False),flush=True)
    return match


def material_slip(m,plant,d):
    loads=np.zeros(6);weighted=np.zeros(6)
    import mujoco as mj
    for contact_index,contact in enumerate(d.contact):
        ga,gb=int(contact.geom1),int(contact.geom2)
        if not(plant.planes[ga] or plant.planes[gb]):continue
        geom=gb if plant.planes[ga] else ga;leg=plant.feet[geom]
        if leg<0:continue
        force=np.zeros(6);mj.mj_contactForce(m,d,contact_index,force)
        fz=max(0.,float((contact.frame.reshape(3,3).T@force[:3])[2])*(1 if plant.planes[ga] else -1))
        jac=np.zeros((3,m.nv));mj.mj_jac(m,d,jac,np.zeros_like(jac),contact.pos,int(m.geom_bodyid[geom]))
        loads[leg]+=fz;weighted[leg]+=fz*np.linalg.norm((jac@d.qvel)[:2])
    return np.r_[d.time,loads,weighted]


def main():
    parser=argparse.ArgumentParser(description='Matched DN 0 vs 0.05 nA physical pair; plan-only unless --execute.')
    parser.add_argument('--execute',action='store_true',help='actually run the two 2 s physical releases and write Desktop outputs')
    args=parser.parse_args()
    command,command_digest=load_command_module()
    best,eligible=select_actor()
    plan=build_plan(eligible,best,command_digest)
    if not args.execute:
        print(json.dumps(plan,indent=2,ensure_ascii=False),flush=True)
        print('PLAN ONLY. No physical step was taken. Re-run with --execute to perform the pair.',flush=True)
        return 0
    match=run()
    return 0 if match['status']=='CHECKED' else 1


if __name__=='__main__':sys.exit(main())
