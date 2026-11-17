"""Immutable partial bank for three admitted movement teachers."""
import argparse
from pathlib import Path
from shutil import copyfile

import torch

import turning as trials
import verify_native


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--mode",choices=("pivot_left","pivot_right","arc_right"))
    parser.add_argument("--start",default="nominal")
    parser.add_argument("--native",action="store_true")
    parser.add_argument("--compare",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=trials.HERE
    bank=root/"BANK_PHASE_ONE"
    trials.HERE=bank
    if args.prepare:
        bank.mkdir()
        copyfile(root/"PREREG.zh-CN.md",bank/"PREREG.zh-CN.md")
        trials.build([root/p for p in ("CENTER_pivot_left","CENTER_pivot_right","TEACH_arc_right")])
        return
    args.entry=4800 if args.mode.startswith("arc") else 0
    args.exit_walk=4800 if args.mode.startswith("arc") else 0
    args.output=args.out.resolve()
    if args.native:
        verify_native.execute(args)
    else:
        trials.run(args.output,args.start,args.mode,"model",.13,.115,args.entry,args.exit_walk)


if __name__=="__main__":
    main()
