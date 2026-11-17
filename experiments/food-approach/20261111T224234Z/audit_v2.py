"""Exact inherited independent array/model audit with extra immutable input proof."""
import argparse
from bindings_v2 import verify_all
import audit
from runtime import IDS
audit.verify_frozen=verify_all
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trial',choices=IDS,required=True);args=p.parse_args();audit.verify(args.trial)
