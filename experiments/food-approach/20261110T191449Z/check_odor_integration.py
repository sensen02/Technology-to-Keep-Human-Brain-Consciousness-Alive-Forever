"""Replay saved actual poses via mj_forward only; tests sensor clock and policy boundary."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco as mj
from conditional_motor import HERE,ConditionalMotor,OdorBehavior,load_baseline_actor,sha256
from odor_loop import FoodObservation,FoodNeuralPolicy,FoodSource
from physics import Core


def main():
    torch.set_num_threads(1)
    c=Core(adhesion=False)
    base=HERE.parent.parent/'banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500'
    with np.load(base/'trace_1ms.npz',allow_pickle=False) as z:
        qpos=z['qpos'].copy();qvel=z['qvel'].copy();times=z['time'].copy()
    with np.load(base/'neural_decisions.npz',allow_pickle=False) as z:
        x=z['X238'].copy();expected=z['policy_action'].copy()
    # Private pose replay is diagnostic only, never deploy or call tick/mj_step.
    index=5799
    c.d.qpos[:]=qpos[index];c.d.qvel[:]=qvel[index];c.d.time=times[index];mj.mj_forward(c.plant,c.d)
    head=c.plant.body('Head').id
    r=c.d.xmat[head].reshape(3,3)
    center=c.d.xpos[head,:2]+r[:2,0]*1.+r[:2,1]*.2
    source=FoodSource(tuple(center),.1,1.,.5,float(c.d.time),.2)
    sensing=FoodObservation(c,source)
    policy=FoodNeuralPolicy(ConditionalMotor(load_baseline_actor()),OdorBehavior())
    for obs in x[:960]:policy.prime(obs)
    responses=[];commands=[];outputs=[]
    for k in range(1920):
        i=5799+k*5
        c.d.qpos[:]=qpos[i];c.d.qvel[:]=qvel[i];c.d.time=times[i];mj.mj_forward(c.plant,c.d)
        receptors=sensing.receptors(c)
        output=policy.step(x[960+k],receptors)
        responses.append(receptors);commands.append(output['command']);outputs.append(output['action'])
    responses=np.asarray(responses);commands=np.asarray(commands);outputs=np.asarray(outputs)
    checks=dict(all1920_sensor_updates_contiguous=bool(len(responses)==1920),
                receptors_finite_bounded=bool(np.isfinite(responses).all() and responses.min()>=0 and responses.max()<=1),
                receptor_asymmetry_actual_geometry=bool(np.max(np.abs(responses[:,0]-responses[:,1]))>1e-4),
                causal_zero_onset=bool(np.array_equal(responses[:2],np.zeros((2,2)))),
                causal_response_after_elapsed_interval=bool(np.any(responses[2]>0)),
                response_used_matches_last_diagnostic=bool(np.array_equal(sensing.last_response,responses[-1])),
                pending_response_applies_only_next_interval=bool(not np.array_equal(sensing.pending_response,responses[-1])),
                zero_behavior_initialization_neutral=bool(np.array_equal(commands,np.zeros_like(commands))),
                full1920_neural_actions_baseline_bitexact=bool(np.array_equal(outputs,expected[960:])),
                no_physical_integration=bool(c.t==0))
    report=dict(all_passed=all(checks.values()),checks=checks,maximum_receptor_asymmetry=float(np.max(np.abs(responses[:,0]-responses[:,1]))),
                source_hashes={str(HERE/name):sha256(HERE/name) for name in ('conditional_motor.py','odor_loop.py','check_odor_integration.py','odor/food_field.py','odor/HEAD_GEOMETRY_AUDIT.json')},
                source_trajectory=str(base),body_writes='private saved-state replay for sensor test only, never physical deployment',
                physics_integration_steps=0,odor_approach='NOT_ESTABLISHED')
    (HERE/'ODOR_INTEGRATION_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['all_passed']:raise RuntimeError('odor integration check failed')

if __name__=='__main__':main()
