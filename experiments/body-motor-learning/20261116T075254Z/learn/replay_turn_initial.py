"""Apply the fitted body actor to a physically qualified left or right teacher's course."""
import argparse
import importlib.util
from pathlib import Path
import sys

from physical import HERE, sha, write
from policy import load_policy

TEACHER=HERE.parent/"turn"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("direction",choices=("left","right"))
    p.add_argument("output",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--start",default="nominal")
    a=p.parse_args()
    actor=load_policy(a.checkpoint)
    sys.path.insert(0,str(TEACHER))
    spec=importlib.util.spec_from_file_location("continuous_turn_replay",TEACHER/"rollout.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint=a.checkpoint.resolve()
    library=checkpoint.parent/"LIBRARY.npz" if checkpoint.suffix==".pt" else checkpoint
    pins={str(f):sha(f) for f in (checkpoint,library,library.parent/"LIBRARY.json",
                Path(__file__),HERE/"policy.py",TEACHER/"rollout.py",TEACHER/"controller.py")}
    plan=TEACHER/f"{a.direction.upper()}_PLAN.json"
    module.run(a.output,a.direction,plan,label="fitted_library_residual" if checkpoint.suffix==".pt" else "fitted_library",
               subject=actor,start=a.start,subject_path=checkpoint)
    assert all(sha(path)==digest for path,digest in pins.items())
    write(a.output/"MODEL_PINS.json",dict(pins=pins,checkpoint=str(checkpoint),
         note="Observations and requested engineering speed/yaw only; private teacher does not choose subject actions."))


if __name__=="__main__":
    main()
