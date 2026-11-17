"""Freshprocess288model/filter replay plus allactualsourcegeometry andarrival checks."""
import argparse
from contracts import ROOT,read
from runtime import entry
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trial',choices=[c['id'] for c in read(ROOT/'PREREG_PROPOSAL.json')['layouts']],required=True);a=p.parse_args();entry('audit.py','verify')(a.trial)
