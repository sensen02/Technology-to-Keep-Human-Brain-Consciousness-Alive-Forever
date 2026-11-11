"""Existing final checkpoint on original teacher history; pre-action stance error only."""
import json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from model import HERE,BASE,DATA,sha256
W=24.446354755878005
LEGS=['LF','LH','LM','RF','RH','RM']
XML=BASE.parent.parent/'banc-triad-pad/20261110T010000Z/MODEL_TRIAD_x200.xml'
def main():
 torch.set_num_threads(1)
 teacher=HERE.parent/'attempt02/right_yaw_m060/TRACE_1MS.npz';ck=HERE/'TRAIN/checkpoint_002000.pt'
 with np.load(DATA,allow_pickle=False) as z:y=z['Y42'].copy();times=z['actual_time'].copy()
 with np.load(teacher,allow_pickle=False) as z:loads=z['loads'].copy();clock=z['time'].copy()
 idx=999+5*np.arange(2880);assert np.max(np.abs(clock[idx]-times))<1e-8
 stance=loads[idx]>=.1*W;p=torch.load(ck,map_location='cpu',weights_only=False);mean=p['replay']['means'][:,0].numpy();error=(np.tanh(mean)-y)**2
 names=[n.attrib['joint'] for n in ET.parse(XML).getroot().find('actuator')];ids={leg:[i for i,n in enumerate(names) if n.startswith('joint_'+leg)] for leg in LEGS};mask=np.zeros((2880,42),bool)
 legs={};total=error[960:].sum()
 for j,leg in enumerate(LEGS):
  q=ids[leg];mask[:,q]=stance[:,j,None];e=error[960:,q];s=stance[960:,j];legs[leg]=dict(stance_row_fraction=float(s.mean()),total_action_MSE=float(e.mean()),stance_action_MSE=float(e[s].mean()),nonstance_action_MSE=float(e[~s].mean()),stance_error_fraction_within_leg=float(e[s].sum()/e.sum()),stance_error_fraction_all42=float(e[s].sum()/total),leg_error_fraction_all42=float(e.sum()/total))
 post=mask[960:];e=error[960:];weights=1.+3.*post;diag=json.loads((HERE/'SLIP_DIAGNOSTICS.json').read_text());ax=diag['actual']['xy_speed'];tx=diag['teacher']['xy_speed'];slip_factors={}
 for leg in LEGS:
  ar=diag['actual']['per_leg'][leg]['fullwindow_slip'];tr=diag['teacher']['per_leg'][leg]['fullwindow_slip'];slip_factors[leg]=dict(actual_slip=ar,teacher_slip=tr,weighted_foot_speed_actual_to_teacher=(ar*ax)/(tr*tx),slip_ratio_actual_to_teacher=ar/tr)
 r=dict(scope='read-only evidence for possible later fixed-hidden head-only weighted fit; this does not show that weighting improves actual physics and does not authorize/execute another model',preaction_load_index='trace row999+5*k; exact time matched all2880 actual teacher source times',threshold_W=.1,prime_rows_excluded=960,per_leg=legs,all42=dict(action_MSE=float(e.mean()),stance_action_MSE=float(e[post].mean()),nonstance_action_MSE=float(e[~post].mean()),stance_error_fraction=float(e[post].sum()/e.sum()),stance_joint_sample_fraction=float(post.mean()),proposed_weight4_normalized_MSE=float((e*weights).sum()/weights.sum())),denominator=dict(actual_xy_speed=ax,teacher_xy_speed=tx,actual_to_teacher=ax/tx),slip_decomposition=slip_factors,physics_steps=0,optimizer_updates=0,source_hashes={str(v):sha256(v) for v in (Path(__file__),XML,teacher,DATA,ck,HERE/'SLIP_DIAGNOSTICS.json')})
 (HERE/'STANCE_FIT_DIAGNOSTICS.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':main()
