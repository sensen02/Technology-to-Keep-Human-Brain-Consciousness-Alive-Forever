"""Environment-side food transduction and strict actor input boundary.

No learning or simulation on import. This module does not choose steering actions.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
from conditional_motor import HERE,OdorBehavior
sys.path.insert(0,str(HERE/'odor'))
from food_field import FoodSource,BilateralOdorSensory,bilateral_positions


class FoodObservation:
    """Only receptors() crosses into policy; privileged diagnostics are separate."""
    def __init__(self,core,source,noise_std=0.,seed=20261110):
        audit=json.loads((HERE/'odor/HEAD_GEOMETRY_AUDIT.json').read_text())
        if audit['status']!='PASS':raise ValueError('head geometry audit required')
        self.head=core.plant.body('Head').id
        if self.head!=audit['head']['body_id']:raise ValueError('head identity mismatch')
        aabb=audit['head']['local_mesh_aabb']
        self.forward=aabb['max'][0]+.1*aabb['extent'][0]
        self.lateral=max(abs(aabb['min'][1]),abs(aabb['max'][1]))+.1*aabb['extent'][1]
        self.source=source
        self.sensory=BilateralOdorSensory(source,K=1.,tau_s=.05,noise_std=noise_std,seed=seed)
        # Separate accumulated sensory clock prevents tolerance mismatch from80 native steps/ms.
        self.clock=float(core.d.time)
        self.last_physics_time=None
        self.last_positions=None
        self.last_response=np.zeros(2)
        self.pending_response=np.zeros(2)

    def receptors(self,core):
        physical=float(core.d.time)
        if self.last_physics_time is not None and abs(physical-self.last_physics_time-.005)>1e-8:
            raise ValueError('food sensing requires exactly one update per actual5ms physical interval')
        if abs(physical-self.clock)>1e-8:raise ValueError('sensory and physical clock drift')
        pos=bilateral_positions(core.d.xpos[self.head],core.d.xmat[self.head].reshape(3,3),self.forward,self.lateral)
        # The action at t receives z(t), computed over the preceding actual interval.
        # Integrate current held input solely for the next decision at t+5ms.
        response=self.pending_response.copy()
        self.pending_response=self.sensory.sample(pos,self.clock,.005)
        self.clock+=.005
        self.last_physics_time=physical
        self.last_positions=pos.copy();self.last_response=response.copy()
        return response

    def diagnostics(self,core):
        """For environment reward/trace only, never append to neural observations."""
        pos=bilateral_positions(core.d.xpos[self.head],core.d.xmat[self.head].reshape(3,3),self.forward,self.lateral)
        mouth_proxy=pos.mean(axis=0)
        edge_distance=max(0.,float(np.linalg.norm(mouth_proxy-np.asarray(self.source.center_xy)))-self.source.patch_radius)
        return dict(physical_time_s=float(core.d.time),sensor_xy=pos.tolist(),receptors=self.last_response.tolist(),
                    food_source=asdict(self.source),mouth_proxy_xy=mouth_proxy.tolist(),
                    edge_distance=edge_distance,inside_patch=bool(edge_distance==0.),
                    ingestion=False,geometry='engineer head center sampling proxy; not mouth actuation',
                    actor_receives_only='receptors2')


class FoodNeuralPolicy:
    """Strict two-module inference, explicit recurrent states, zero hidden resets."""
    def __init__(self,motor,behavior):
        self.motor=motor;self.behavior=behavior
        self.motor_state=motor.initial_state(1)
        self.behavior_state=behavior.initial_state(1)

    @torch.no_grad()
    def prime(self,physical_obs):
        """Teacher priming is neutral; source inactive and behavior memory held initial."""
        x=torch.as_tensor(np.asarray(physical_obs)[None],dtype=torch.float64)
        mean,self.motor_state=self.motor.step(x,x.new_zeros((1,2)),self.motor_state)
        return mean[0].tanh().numpy()

    @torch.no_grad()
    def step(self,physical_obs,receptors):
        x=torch.as_tensor(np.asarray(physical_obs)[None],dtype=torch.float64)
        z=torch.as_tensor(np.asarray(receptors)[None],dtype=torch.float64)
        command,self.behavior_state=self.behavior.step(z,self.behavior_state)
        mean,self.motor_state=self.motor.step(x,command,self.motor_state)
        return dict(action=mean[0].tanh().numpy(),command=command[0].numpy(),
                    motor_hidden=(torch.cat(self.motor_state,dim=-1) if isinstance(self.motor_state,(tuple,list)) else self.motor_state)[0].numpy().copy(),
                    motor_state_dims=([s.shape[-1] for s in self.motor_state] if isinstance(self.motor_state,(tuple,list)) else [self.motor_state.shape[-1]]),
                    behavior_hidden=self.behavior_state[0].numpy().copy())
