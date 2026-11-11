import argparse,json,torch
from model_io import load_checked
from data import invariance
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sha256',required=True);a=p.parse_args()
m,payload=load_checked(a.checkpoint,a.sha256,require_final=False)
if payload['curve'][-1]['step'] not in (1000,2000):raise ValueError('step')
print(json.dumps(dict(step=payload['curve'][-1]['step'],weights_finite=True,config_kind=payload['config']['kind'])))
