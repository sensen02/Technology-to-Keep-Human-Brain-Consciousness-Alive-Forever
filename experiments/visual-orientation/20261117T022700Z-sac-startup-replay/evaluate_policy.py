"""Evaluate a saved warm or SAC checkpoint through the physical closed loop."""
import argparse
from pathlib import Path

import torch

from course import rollout, sha
from model import SACBrain


class Policy:
    def __init__(self,model,deterministic):
        self.model=model
        self.deterministic=deterministic

    def choose(self,env,obs):
        return self.model.choose(env,obs,deterministic=self.deterministic)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("case")
    parser.add_argument("out",type=Path)
    parser.add_argument("--sample",type=int)
    parser.add_argument("--cut",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.sample if args.sample is not None else 3373)
    data=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    agent=SACBrain().eval()
    agent.load_state_dict(data["model"])
    policy=Policy(agent,args.sample is None)
    pin=dict(path=str(args.checkpoint.resolve()),sha256=sha(args.checkpoint),
             sac_updates=data.get("sac_updates",0),sample_seed=args.sample)
    rollout(args.out,args.case,policy,agent.brain,"sac" if data.get("sac_updates") else "warm",pin=pin,cut=args.cut)
    assert sha(args.checkpoint)==pin["sha256"]


if __name__=="__main__":
    main()
