"""Repeat the visual task with a later perceptual transition to the low speed."""
import argparse
from pathlib import Path
import approach
from approach_runtime import turn,sha
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("out",type=Path)
p.add_argument("--angle",type=float,default=20.)
p.add_argument("--near",type=float,required=True)
a=p.parse_args()
approach.NEAR=a.near
r=approach.trial(a.out.resolve(),a.angle,2.68,"intact",.12,18800)
turn.stand.put(a.out/"VARIANT.json",dict(slow_distance_native=a.near,wrapper_sha256=sha(Path(__file__))))
raise SystemExit(0 if r["passed"] else 1)
