"""Repeat the prior fixed-parameter sensitivity suite with the retrained body."""
import argparse
from pathlib import Path

from context import HERE,first,study
import fixed_neck as kernel
from fixed_neck import FixedCircuit,write
from evaluate import archived


def run(variant,suite,name,checkpoint,out):
    factor=1.
    if variant=='slow':factor=.5
    elif variant=='fast':factor=2.
    elif variant=='kinetic08':
        kernel.RISE_MS*=.8
        kernel.DECAY_MS*=.8
    elif variant=='kinetic12':
        kernel.RISE_MS*=1.2
        kernel.DECAY_MS*=1.2
    elif variant=='leak08':kernel.LEAK_NS*=.8
    elif variant=='leak12':kernel.LEAK_NS*=1.2
    build=lambda catalog,batch=1:FixedCircuit(catalog,batch,velocity_multiplier=factor)
    first.Circuit=build
    study.Circuit=build
    archived(suite,name,checkpoint,out)
    write(out/'SENSITIVITY.json',dict(variant=variant,velocity_multiplier=factor,rise_ms=kernel.RISE_MS.tolist(),
                                   decay_ms=kernel.DECAY_MS.tolist(),leak_ns=kernel.LEAK_NS))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant',choices=('slow','fast','kinetic08','kinetic12','leak08','leak12'))
    parser.add_argument('suite',choices=('body','ground'))
    parser.add_argument('name')
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('out',type=Path)
    a=parser.parse_args()
    run(a.variant,a.suite,a.name,a.checkpoint,a.out)
