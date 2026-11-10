"""Read-only source/data audit: no physics, no neural stepping, no driver imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib,ast,types,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path('/run/media/sensen/Data2/cell_wound_prototype')
O=Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z/tactile')
assert O.resolve().is_relative_to(Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z').resolve())
O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R))
from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelQuantities,channel_index_map,LEGS,CHANNEL_SPEC
D=Path('/home/sensen/Desktop/engine修复验收')
ident=D/'s5-banc888-physical-handover/20261101T220018Z/IDENTITIES.npz'
lookup=D/'s3-banc888-runtime-admission/20261101T220018Z/SENSORY_LOOKUP_SNAPSHOT888.json'
xml=D/'s5-joint-torque-plant/20261105T060730Z/MODEL_JOINT_TORQUE.xml'
sources=[ident,lookup,xml,R/'engine/embodied/sensory_channels.py',R/'run_neural_muscle_closure.py',R/'run_integrated_repaired_closure.py',R/'experiments/s5-neural-translator/20261105T200021Z/runtime.py',R/'experiments/s5-neural-joint-torque/20261105T062800Z/run_trial.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in sources}
roots=np.load(ident)['root_ids'];groups,stats=channel_index_map(len(roots),roots,lookup_path=lookup);enc=ChannelSensoryEncoder(groups,len(roots))
counts=enc.coverage();totals={ch:sum(counts[l][ch] for l in LEGS) for ch in CHANNEL_SPEC}
root=ET.parse(xml).getroot();joints=[j for j in root.findall('.//worldbody//joint') if 'name' in j.attrib];q0=np.fromstring(root.find('.//key').get('qpos'),sep=' ')
qadr={j.get('name'):7+i for i,j in enumerate(joints)};vadr={n:q-1 for n,q in qadr.items()}
jmap={l:dict(q=[qadr[n] for n in qadr if n.lower().startswith('joint_'+l)],tib_q=next(qadr[n] for n in qadr if n.lower().startswith('joint_'+l) and 'tibia' in n.lower()),tib_v=next(vadr[n] for n in vadr if n.lower().startswith('joint_'+l) and 'tibia' in n.lower())) for l in LEGS}
geoms=[g.attrib for g in root.findall('.//worldbody//geom')]
foot={l:[g['name'] for g in geoms if g.get('name','').lower().startswith(l) and 'tars' in g.get('name','').lower()] for l in LEGS}
# Execute only the audited pure observation function, with a fake contactForce oracle.
node=next(n for n in ast.parse((R/'run_neural_muscle_closure.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='observe_quantities')
ns=dict(np=np,LEGS=LEGS,ChannelQuantities=ChannelQuantities)
exec(compile(ast.Module(body=[node],type_ignores=[]),'<source-extracted-observe_quantities>','exec'),ns)
fake=types.SimpleNamespace(mjtGeom=types.SimpleNamespace(mjGEOM_PLANE=0),mj_contactForce=lambda m,d,i,f:f.__setitem__(slice(None),d.forces[i]))
sys.modules['mujoco']=fake
model=types.SimpleNamespace(geom_type=[1,0,1,1]);gm={0:'lf'}
def obs(forces,pairs):
 d=types.SimpleNamespace(qpos=q0.copy(),qvel=np.zeros(48),ncon=len(pairs),contact=[types.SimpleNamespace(geom1=a,geom2=b) for a,b in pairs],forces=forces)
 return ns['observe_quantities'](model,d,q0,jmap,gm)
probes={}
for name,fs,ps in [('zero_force_geometric',[[0]*6],[(0,1)]),('normal_3_shear_4',[[3,4,0,0,0,0]],[(0,1)]),('nonplane_touch',[[3,4,0,0,0,0]],[(0,2)]),('nonfoot_plane',[[3,4,0,0,0,0]],[(3,1)])]:
 q=obs(fs,ps);cur,_=enc.encode(q);probes[name]={'lf_quantities':vars(q['lf']),'lf_bristle_current_nA':float(cur[groups['lf']['bristle_touch'][0]]),'lf_camp_current_nA':float(cur[groups['lf']['campaniform_strain'][0]])}
probes['linear_force']={str(f):float(enc.encode({'lf':ChannelQuantities(contact_force_norm=f)})[0][groups['lf']['campaniform_strain'][0]]) for f in [0,1,10,1000000]}
probes['missing_all_legs_nonzero_count']=int(np.count_nonzero(enc.encode({})[0]))
q=q0.copy();q[jmap['lf']['q'][0]]+=.3;q[jmap['lf']['q'][1]]-=.3
d=types.SimpleNamespace(qpos=q,qvel=np.zeros(48),ncon=0)
probes['opposed_joint_cancellation']=vars(ns['observe_quantities'](model,d,q0,jmap,gm)['lf'])
def dist(a):
 a=np.asarray(a);return dict(n=int(a.size),min=float(a.min()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()),negative=int((a<0).sum()),zero=int((a==0).sum()))
summary_path=D/'s5-neural-translator/20261106T003500Z/SUMMARY.json';s=json.loads(summary_path.read_text());base=next(c for c in s['candidates'] if c['name']=='zero');res=next(v for v in base['results'] if v['release_time_s']==1.3)
folders={'warm':D/'s5-neural-translator/20261105T200021Z/DATA/train_offset_0','baseline':Path(res['folder'])}
measure={}
for label,folder in folders.items():
 bp=folder/'BODY.npz';sources.append(bp);b=np.load(bp);dev=b['qpos']-q0
 per={}
 for l,j in jmap.items():
  per[l]={'tibia_position_current_nA':dist(dev[:,j['tib_q']]),'hook_velocity_current_nA':dist(.03*b['qvel'][:,j['tib_v']]),'hairplate_current_nA':dist(dev[:,j['q']].mean(1))}
 npath=folder/'NATIVE.npz';sources.append(npath);native=np.load(npath)['rows']
 measure[label]=dict(folder=str(folder),body_shapes={k:list(b[k].shape) for k in b.files},per_leg=per,recorded_vertical_force_NOT_sensory_norm=dist(native[:,1:7]),nonfoot_positive_record_count=int((native[:,7]>0).sum()),samples=int(len(native)))
 if (folder/'NEURAL.npz').exists():
  p=folder/'NEURAL.npz';sources.append(p);n=np.load(p);measure[label]['neural_shapes']={k:list(n[k].shape) for k in n.files}
  if 'sensory_current_norm' in n:measure[label]['saved_sensory_current_L2_nA']=dist(n['sensory_current_norm'])
result=dict(scope='S4/S5 read-only audit; 0 physics forward/step, 0 neural step; no heavy graph loaded',root_count=len(roots),unique_root_count=len(set(roots)),stats=stats,counts=counts,totals=totals,active_cells=sum(totals[ch] for ch in totals if CHANNEL_SPEC[ch][3]),intentionally_zero_cells=sum(totals[ch] for ch in totals if not CHANNEL_SPEC[ch][3]),foot_geom_names=foot,joint_addresses=jmap,probes=probes,measurements=measure,source_hashes={str(p):sha(p) for p in sources},original_sources_unchanged=all(sha(Path(p))==h for p,h in before.items()),limitations=['No saved contact identities/forces in current BODY or NEURAL; NATIVE vertical force cannot reconstruct sum contact-force norms or geometric touch bit.','Per-cell biological directional tuning not in this runtime mapping.'])
assert result['original_sources_unchanged'];assert len(roots)==12000
(O/'EVIDENCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['root_count','stats','counts','totals','active_cells','intentionally_zero_cells','probes']},ensure_ascii=False,indent=2))
