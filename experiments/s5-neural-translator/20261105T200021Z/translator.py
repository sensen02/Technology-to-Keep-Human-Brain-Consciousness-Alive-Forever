"""Bias-free neural-only residual translator and serializable causal spike filters."""
import numpy as np
import torch
from torch import nn

class SpikeHistory:
    def __init__(self,cells,dt=.0005):
        self.cells=np.asarray(cells,dtype=int);self.dt=dt;self.decay=np.exp(-dt/np.array([.005,.020,.050]))[:,None];self.reset()
    def reset(self):self.rates=np.zeros((3,len(self.cells)));self.samples=0
    def consume(self,spikes):
        self.rates=self.decay*self.rates+(1-self.decay)*np.asarray(spikes[self.cells],float)[None,:]/self.dt;self.samples+=1
    def features(self):return (self.rates/100).astype(np.float32).ravel().copy()
    def state_dict(self):return dict(cells=self.cells.copy(),rates=self.rates.copy(),samples=self.samples)
    def load_state_dict(self,state):
        assert np.array_equal(state['cells'],self.cells);self.rates=state['rates'].copy();self.samples=int(state['samples'])

class Translator(nn.Module):
    def __init__(self,kind='mlp'):
        super().__init__();self.kind=kind
        self.net=nn.Sequential(nn.Linear(612,64,bias=False),nn.Tanh(),nn.Linear(64,64,bias=False),nn.Tanh(),nn.Linear(64,42,bias=False)) if kind=='mlp' else nn.Sequential(nn.Linear(612,42,bias=False))
        nn.init.zeros_(self.net[-1].weight)
    def forward(self,x):return self.net(x)
    def export(self,path):
        matrices=[m.weight.detach().cpu().numpy() for m in self.net if isinstance(m,nn.Linear)]
        np.savez_compressed(path,kind=self.kind,**{f'W{i}':v for i,v in enumerate(matrices)})
    def restore_export(self,path):
        with np.load(path) as z:
            assert str(z['kind'])==self.kind
            with torch.no_grad():
                for i,m in enumerate(m for m in self.net if isinstance(m,nn.Linear)):m.weight.copy_(torch.from_numpy(z[f'W{i}']))

class NumpyTranslator:
    def __init__(self,path):
        with np.load(path) as z:self.kind=str(z['kind']);self.weights=[z[f'W{i}'].copy() for i in range(3 if self.kind=='mlp' else 1)]
    def __call__(self,x):
        a=np.asarray(x,np.float32)
        for i,w in enumerate(self.weights):
            a=a@w.T
            if i<len(self.weights)-1:a=np.tanh(a)
        return a
