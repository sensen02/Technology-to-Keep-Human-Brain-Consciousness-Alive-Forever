"""Train the same SAC with a head-local obstacle observation."""
import argparse
from pathlib import Path

from proximity import AssistedProxy
import train


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--steps',type=int,default=170000)
    p.add_argument('--interval',type=int,default=10000)
    p.add_argument('--course',choices=('wall','detour'),default='wall')
    p.add_argument('--guided',action='store_true')
    train.Proxy=AssistedProxy
    train.main(p.parse_args())
