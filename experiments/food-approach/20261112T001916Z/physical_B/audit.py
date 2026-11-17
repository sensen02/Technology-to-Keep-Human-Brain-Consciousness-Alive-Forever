"""Unchanged frozen AB exact288 audit with paired behavior loader/new proof globals."""
import argparse
from runtime import entry
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trial',choices=('B_noodor','B_equal','B_left_pulse','B_right_pulse','B_left_to_right','B_right_to_left'),required=True);a=p.parse_args();entry('audit.py','verify')(a.trial)
