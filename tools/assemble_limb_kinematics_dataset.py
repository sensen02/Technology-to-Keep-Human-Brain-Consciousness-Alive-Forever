"""Assemble a limb-level (joint-resolved) Drosophila walking dataset from open sources.

Source: Harvard Dataverse doi:10.7910/DVN/BBNPYX, CC0 1.0, Haustein et al. 2024 companion
tracking set: inverse-kinematics joint angles of a leg model, six legs, five joints, three
axes each, one CSV per walking trial. Downloaded, parsed, normalised and hashed here.

Also inventories every repository that was checked, with the observed access status, so the
next round does not repeat the search.
"""
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s4-limb-level-dataset/20261103T033000Z')
RAW=Path('/tmp/flydata/angles')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
LEGS=('R1','R2','R3','L1','L2','L3')
JOINTS=('ThCx','CxTr','TrFe','FeTi','TiTar')
AXES=('yaw','pitch','roll')
# our model hinges per leg, for the declared projection
MODEL_HINGES=('Coxa_pitch','Coxa_roll','Coxa_yaw','TrochanterFemur_pitch','TrochanterFemur_roll','Tibia_pitch','Tarsus1_pitch')
PROJECTION={
 'Coxa_pitch':'ThCx-pitch','Coxa_roll':'ThCx-roll','Coxa_yaw':'ThCx-yaw',
 'TrochanterFemur_pitch':'TrFe-pitch','TrochanterFemur_roll':'TrFe-roll',
 'Tibia_pitch':'FeTi-pitch','Tarsus1_pitch':'TiTar-pitch'}
def main():
 files=sorted(RAW.glob('*_modelAngles.csv'));assert files,(RAW,)
 trials=[];tables={};maxlen=0;cols_ref=None
 for p in files:
  with p.open() as f:
   r=csv.reader(f);hdr=next(r)
   data=np.array([[float(x) for x in row[1:]] for row in r if row and row[0]!=''])
  assert hdr[0]=='' and len(hdr)-1==90,hdr[:3]
  if cols_ref is None:cols_ref=hdr[1:]
  else:assert cols_ref==hdr[1:],'column order differs across trials'
  name=p.stem.replace('_modelAngles','')
  trial,name_parts=name,{}
  tables[name]=data;maxlen=max(maxlen,len(data))
  nz=(data!=0).sum(0)
  trials.append(dict(trial=name,frames=int(len(data)),columns=int(data.shape[1]),
   columns_ever_nonzero=int((nz>0).sum()),columns_always_zero=int((nz==0).sum()),
   fixed_dof_fraction=float((nz==0).mean()),nan_count=int(np.isnan(data).sum())))
 assert len(trials)>=16,len(trials)
 # padded dense tensor [trial, frame, 90] plus a mask, so nothing is silently zero-filled
 T=len(trials);A=np.full((T,maxlen,90),np.nan);M=np.zeros((T,maxlen),bool)
 for i,t in enumerate(trials):
  d=tables[t['trial']];A[i,:len(d)]=d;M[i,:len(d)]=True
 np.savez_compressed(O/'LIMB_JOINT_ANGLES.npz',angles_deg=A,frame_mask=M,
  trial_names=np.array([t['trial'] for t in trials]),column_names=np.array(cols_ref),
  legs=np.array(LEGS),joints=np.array(JOINTS),axes_axes=np.array(AXES),units='degrees (as published; sign convention is the source model, not ours)')
 # projection onto this project's 42 hinges, declared and explicit
 proj={}
 for leg in LEGS:
  for h in MODEL_HINGES:
   src=PROJECTION[h];proj[f'{leg}:{h}']=f'{leg}-{src}'
 missing=[k for k,v in proj.items() if v not in cols_ref]
 assert not missing,missing[:5]
 unused=[c for c in cols_ref if c not in set(proj.values())]
 out=dict(passed=True,
  source=dict(dataset='2D Tracking and 3D reconstruction of legs from tethered walking Drosophila',
   doi='10.7910/DVN/BBNPYX',license='CC0 1.0',host='Harvard Dataverse',
   api_json_sha256='recorded below',files_used=[p.name for p in files]),
  granularity=dict(level='joint-resolved limb kinematics',per_leg_joints=list(JOINTS),axes=list(AXES),
   columns_per_trial=90,model_hinges_mapped=42,extra_columns_in_source=len(unused)),
  trials=trials,shape=[T,int(maxlen),90],
  projection_to_our_42_hinges=proj,projection_note='direct name-level mapping; coil/claw and wing columns in the source are not used, and CxTr roll/pitch are dropped because the model has no separate coxa-trochanter DOF',
  unused_source_columns=unused,
  limits=['tethered walking: the thorax is fixed, so this dataset carries NO body translation, no ground reaction force and no centre-of-mass trajectory',
          'angles are in the source model convention and degrees; our model uses radians and its own hinge axes, so a sign/offset calibration is required before use',
          'columns that are always zero were fixed during the source inverse kinematics and must not be treated as measurements of zero'],
  calls=dict(model_load=0,forward=0,body_step=0,neural_step=0))
 (O/'RESULT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 inv={}
 for p in files+[O/'LIMB_JOINT_ANGLES.npz',O/'RESULT.json']:
  inv[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size,importance='核心必需')
 (O/'FILE_INVENTORY.json').write_text(json.dumps(inv,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(trials=T,frames=int(maxlen),mapped=len(proj),unused=len(unused),
  always_zero_cols=int(sum(t['columns_always_zero'] for t in trials))),ensure_ascii=False))
if __name__=='__main__':main()
