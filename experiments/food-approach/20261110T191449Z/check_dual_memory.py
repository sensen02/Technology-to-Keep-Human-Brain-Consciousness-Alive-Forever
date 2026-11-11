"""Offline check that behavior integration records/preserves BOTH motor memories."""
import json
import sys
import numpy as np
import torch
from conditional_motor import HERE,sha256
from odor_loop import FoodNeuralPolicy
from steering_behavior import SteeringBehavior
sys.path.insert(0,str(HERE/'residual_candidate'))
from residual_motor import ResidualMotor,load_baseline_actor


def main():
    torch.set_num_threads(1)
    path=HERE.parent.parent/'banc-walking-recovery/20261110T152754Z/NATIVE_ROLE_2500/neural_decisions.npz'
    with np.load(path,allow_pickle=False) as z:
        x=z['X238'].copy();expected=z['policy_action'].copy();hidden=z['hidden_after'].copy()
    policy=FoodNeuralPolicy(ResidualMotor(load_baseline_actor()),SteeringBehavior())
    for obs in x[:960]:policy.prime(obs)
    outputs=[];recorded=[];states=[]
    for obs in x[960:]:
        out=policy.step(obs,np.array([.3,.1]))
        if out['motor_state_dims']!=[128,64]:raise ValueError('dual-state dimension contract changed')
        outputs.append(out['action']);recorded.append(out['motor_hidden']);states.append(policy.motor_state[1][0].detach().numpy().copy())
    outputs=np.asarray(outputs);recorded=np.asarray(recorded);states=np.asarray(states)
    checks=dict(all1920_neutral_actions_bitexact=bool(np.array_equal(outputs,expected[960:])),
                both_memories_recorded_192_dims=bool(recorded.shape==(1920,192)),
                base_memory_exact=bool(np.array_equal(recorded[:,:128],hidden[960:])),
                residual_memory_exact=bool(np.array_equal(recorded[:,128:],states)),
                residual_memory_not_reset_at_withdrawal=bool(np.any(recorded[0,128:]!=0)))
    report=dict(all_passed=all(checks.values()),checks=checks,physics_steps=0,optimizer_updates=0,
                sources={str(HERE/n):sha256(HERE/n) for n in ('check_dual_memory.py','odor_loop.py','steering_behavior.py','residual_candidate/residual_motor.py')},
                interpretation='offline original saved-observation neutral command memory check; no food approach training or physical success')
    out=HERE/'DUAL_MEMORY_CHECKS.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
    if not report['all_passed']:raise ValueError('dual memory integration failed')

if __name__=='__main__':main()
