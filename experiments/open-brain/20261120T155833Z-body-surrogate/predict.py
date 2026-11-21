"""Predict one or more observed-state transitions, without simulator dependencies."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model import FeedbackSampler
from skeleton import ObservedBody

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('input',type=Path,help='chronological observations in an NPZ')
    cli.add_argument('output',type=Path)
    cli.add_argument('--weights',type=Path,default=Path(__file__).with_name('weights.pt'))
    cli.add_argument('--seed',type=int,default=17)
    args=cli.parse_args()
    if args.output.exists():
        cli.error('choose a new output file')
    net=ObservedBody(args.weights).eval()
    with np.load(args.input) as z:
        fields=[torch.from_numpy(z[k].astype('f')) for k in ('previous_mm','current_mm','touch_uN','dn_counts')]
    n=len(fields[0])
    if n<1 or [tuple(v.shape) for v in fields]!=[(n,21,3),(n,21,3),(n,2,6),(n,4,17)]:
        cli.error('expected shapes [N,21,3], [N,21,3], [N,2,6], [N,4,17]')
    if not all(torch.isfinite(v).all() for v in fields) or any((v<0).any() for v in fields[2:]):
        cli.error('invalid observation')
    sampler=FeedbackSampler(1)
    with torch.no_grad():
        pos,logit=net(*fields)
        prob=logit.sigmoid()
        gen=torch.Generator().manual_seed(args.seed)
        spikes=torch.cat([sampler.sample(p[None],torch.rand((1,5,2),generator=gen)) for p in prob])
    with args.output.open('xb') as f:
        np.savez_compressed(f,next_mm=pos.numpy(),an_prob=prob.numpy(),an_body_source=spikes.numpy())
if __name__=='__main__':main()
