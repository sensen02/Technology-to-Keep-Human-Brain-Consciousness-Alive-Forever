"""Collect verified narrow-angle initiation examples with the frozen visual proposal."""
import argparse
import json
from pathlib import Path

import torch

import course

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    course.SCENES=course.HERE/"FLAT_DEMOS.json"
    config=json.loads(course.SCENES.read_text())[args.case]
    core=course.new_brain()
    course.rollout(args.output,args.case,course.Teacher("visual",config),core,"visual")
