"""First food-stage neural head exposes only the motor's qualified turn channel.

Speed output is structurally absent until an independent deceleration motor is
validated. No source coordinates, direction, time or scripted steering enters.
"""
import math
import torch
from torch import nn


class SteeringBehavior(nn.Module):
    VERSION='receptors2_slow_gru32_turnonly_v1'
    ALPHA=-math.expm1(-.005/.1)
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(2,32,dtype=torch.float64)
        self.gru=nn.GRUCell(32,32,dtype=torch.float64)
        self.head=nn.Linear(32,1,dtype=torch.float64)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def initial_state(self,batch_size):
        return self.head.weight.new_zeros((batch_size,32))

    def mean_step(self,receptors,state):
        if receptors.ndim!=2 or receptors.shape[1]!=2 or state.shape!=(len(receptors),32):
            raise ValueError('requires actual bilateral receptors(B,2) and explicit memory(B,32)')
        candidate=self.gru(torch.tanh(self.encoder(receptors)),state)
        state=(1-self.ALPHA)*state+self.ALPHA*candidate
        return self.head(state),state

    def step(self,receptors,state):
        latent,state=self.mean_step(receptors,state)
        turn=latent.tanh()
        return torch.cat((torch.zeros_like(turn),turn),dim=-1),state

    def identity(self):
        return dict(version=self.VERSION,inputs='actual local receptor[left,right] only',
                    recurrent_state=32,alpha=self.ALPHA,output='[structural_zero_speed,tanh(neural_turn)]',
                    training='NOT_TRAINED_UNTIL_QUALIFIED_MOTOR_AND_PREREGISTERED_FOOD_LAYOUTS',
                    excludes=['food_position','target_distance','target_bearing','time','phase','teacher'])
