"""Representation-only adapter; preserves residual neural calculations and state ordering."""
import torch
from torch import nn


class FlatResidualMotor(nn.Module):
    STATE_DIMS=(128,64)
    def __init__(self,motor):
        super().__init__();self.motor=motor

    def initial_state(self,batch_size):
        return torch.cat(self.motor.initial_state(batch_size),dim=-1)

    def step(self,obs,command,state):
        if state.ndim!=2 or state.shape!=(obs.shape[0],192):raise ValueError('flat state must be (B,192), base128 then residual64')
        mean,new=self.motor.step(obs,command,(state[:,:128],state[:,128:]))
        return mean,torch.cat(new,dim=-1)

    def sequence(self,obs,command,state=None):
        if obs.ndim!=3 or obs.shape[-1]!=238 or command.shape!=obs.shape[:2]+(2,) or len(obs)==0:raise ValueError('nonempty obs(T,B,238)/command(T,B,2) required')
        state=self.initial_state(obs.shape[1]) if state is None else state
        means,states=[],[]
        for x,u in zip(obs,command):
            mean,state=self.step(x,u,state);means.append(mean);states.append(state)
        return torch.stack(means),torch.stack(states)
