"""Same frozen location and slot with long north/south floor tiles."""
import argparse
from pathlib import Path

import anchored_course as actor

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("output",type=Path)
p.add_argument("--duration",type=int,default=actor.END)
p.add_argument("--native",action="store_true")
a=p.parse_args()
original=actor.q.stan.identity.capture
actor.q.SOURCE=Path(__file__).with_name("Y_SLOT.json")

def identity(core, info, *, extra_paths):
    return original(core,{**info,"floor_partition_axis_first":"y"},
                    extra_paths=[*extra_paths,Path(__file__).resolve()])

actor.q.stan.identity.capture=identity
try:
    actor.run(a.output.resolve(),a.duration,a.native)
finally:
    actor.q.stan.identity.capture=original
