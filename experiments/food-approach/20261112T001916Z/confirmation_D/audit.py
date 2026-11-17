"""Freshprocess exactmodels/filter/actorinterventions/highlevelreplay/donorbody proof."""
import argparse
from contracts import trials
from runtime import entry
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trial',choices=[r['id'] for r in trials()],required=True);a=p.parse_args();entry('audit.py','verify')(a.trial)
