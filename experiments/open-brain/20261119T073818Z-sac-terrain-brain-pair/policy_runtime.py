"""Deterministic actor mean, shared by export validation and physical evaluation."""
import numpy as np


def actor_mean(saved,sense):
    def forward(y):
        for layer in (0,2,4):
            y=y@saved[f'w{layer}'].T+saved[f'b{layer}']
            if layer!=4:
                y=np.maximum(y,0.)
        return y[...,:2]

    command=forward(sense)
    if 'mirror' in saved:
        other=sense[...,saved['mirror']].copy()
        other[...,1]*=-1.
        partner=forward(other)
        partner[...,1]*=-1.
        command=(command+partner)*.5
    return np.tanh(command)


def decode_policy(saved,sense):
    output=actor_mean(saved,sense)
    translated=output
    if 'intent_grid' in saved:
        grid=saved['intent_grid']
        ij=np.rint((output+1.)*.5*(len(grid)-1)).astype(int).clip(0,len(grid)-1)
        translated=grid[tuple(ij)]
    return output,(.76*(translated[0]+1.)/2.,.22*translated[1])
