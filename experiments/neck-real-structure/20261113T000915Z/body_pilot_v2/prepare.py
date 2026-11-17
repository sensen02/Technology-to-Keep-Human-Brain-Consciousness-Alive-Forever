"""Actual new final400 digest supplied by parent only after completed calibration proofs."""
import argparse,ast
from runtime import *
from adapter import namespace

def main():
 p=argparse.ArgumentParser();p.add_argument('--bind-completed-calibration',action='store_true');p.add_argument('--brain-sha256');a=p.parse_args()
 if not a.bind_completed_calibration:print('CODE_ONLY_UNBOUND no admission');return
 digest=a.brain_sha256
 if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Actualcompletedfinal400SHArequired')
 declared=[ast.literal_eval(node.value) for node in ast.parse((CAL/'common.py').read_text()).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FORMAT' for t in node.targets)]
 if declared!=['SOURCE_NECK_FEEDBACK_AGGREGATION_V2']:raise ValueError('Actual new strictloaderFORMAT required')
 ns=namespace('prepare.py');ns['BRIDGE_SHA']=digest;ns['main']()
if __name__=='__main__':main()
