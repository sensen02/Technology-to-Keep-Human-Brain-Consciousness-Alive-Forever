"""Bound the effect of explicitly uncertain fixed parameters on ground behavior."""
import argparse
from pathlib import Path
import json

import adapters
import fixed_neck as kernel
from fixed_neck import FixedCircuit, HERE, write
import regress


def run(variant,suite,name,out):
    adapters.install(True)
    factor=1.
    if variant=='slow':
        factor=.5
    elif variant=='fast':
        factor=2.
    elif variant=='kinetic08':
        kernel.RISE_MS*=.8
        kernel.DECAY_MS*=.8
    elif variant=='kinetic12':
        kernel.RISE_MS*=1.2
        kernel.DECAY_MS*=1.2
    elif variant=='leak08':
        kernel.LEAK_NS*=.8
    elif variant=='leak12':
        kernel.LEAK_NS*=1.2
    make=lambda catalog,batch=1:FixedCircuit(catalog,batch,velocity_multiplier=factor)
    # This process alone uses the alternative fixed parameters.
    import body_run,expansion
    body_run.Circuit=make
    expansion.Circuit=make
    # regress.install runs first, so replace that install hook in this one process.
    regress.install=lambda mapped:None
    regress.run(suite,name,out,True)
    write(out/'SENSITIVITY.json',dict(variant=variant,velocity_multiplier=factor,
          rise_ms=kernel.RISE_MS.tolist(),decay_ms=kernel.DECAY_MS.tolist(),leak_ns=kernel.LEAK_NS,
          nominal_parameters_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant',choices=('slow','fast','kinetic08','kinetic12','leak08','leak12'))
    parser.add_argument('suite',choices=('body','ground'))
    parser.add_argument('name')
    parser.add_argument('out',type=Path)
    a=parser.parse_args()
    run(a.variant,a.suite,a.name,a.out)
