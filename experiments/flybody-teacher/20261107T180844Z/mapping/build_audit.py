"""Pure saved-array analysis: no MuJoCo import, no stepping/training."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
P=Path(__file__).resolve().parent
A=json.loads((P/'official_model.json').read_text());B=json.loads((P/'current_model.json').read_text())
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OFF=ROOT/'experiments/flybody-teacher/20261107T174640Z/official-flybody'
AXML=OFF/'flybody/fruitfly/assets/fruitfly.xml';BXML=Path(B['runtime']['model_source'])
def dump(name,obj):(P/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
def table(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (P/name).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def source_line(path,kind,name):
    text=path.read_text().splitlines()
    for i,line in enumerate(text,1):
        if ('<'+kind+' ') in line and ('name="'+name+'"') in line:return str(path)+':'+str(i)
    return str(path)
def ids(model,kind):return {x['name']:x for x in model[kind]}
aj=ids(A,'joints');bj=ids(B,'joints');ab=ids(A,'bodys');bb=ids(B,'bodys')
legs={'LF':'T1_left','LM':'T2_left','LH':'T3_left','RF':'T1_right','RM':'T2_right','RH':'T3_right'}
def acands(leg,group):
    suffix=legs[leg]
    if group=='Coxa':return ['walker/'+x+'_'+suffix for x in ('coxa_abduct','coxa_twist','coxa')]
    if group=='Trochanter':return ['walker/'+x+'_'+suffix for x in ('femur_twist','femur')]
    return ['walker/tibia_'+suffix]
current=[];pairaxes=[]
for actuator in B['actuators']:
    j=bj[actuator['target_name']];suffix=j['name'].removeprefix('joint_');leg=suffix[:2];group=suffix[2:].split('_')[0];candidates=acands(leg,group)
    r=dict(current_action_index=actuator['action_index'],current_actuator=actuator['name'],current_joint=j['name'],current_body=j['body_name'],candidate_official_joints=candidates,
        status='CANDIDATE_ANATOMICAL_GROUP_ONLY',verified_scalar_map=False,reason='serial thorax-coxa region, both three rotations' if group=='Coxa' else 'coxa-femur/trochanter region: official two rotations, current three; front has explicit trochanter body, middle/hind use femur body' if group=='Trochanter' else 'serial femur-tibia hinge; distinct local frames/rest geometry, sign and offset uncalibrated',
        axis_local=j['axis'],range_rad=j['range'],qpos0_rad=j['qpos0'],key0_rad=B['poses']['key0']['qpos'][j['qposadr']],springref_rad=j['qpos_spring'],
        torque_limits_native=actuator['ctrlrange'],rate_cap_rad_s=30,kp_native=actuator['ctrlrange'][1]/.2*128,kd_native=actuator['ctrlrange'][1]*.005*np.sqrt(128),source=source_line(BXML,'joint',j['name']))
    current.append(r)
    for c in candidates:
        o=aj[c]
        for pose in ('qpos0','key0'):
            u=np.array(A['poses'][pose]['joint_axes_thorax'][o['id']]);v=np.array(B['poses'][pose]['joint_axes_thorax'][j['id']])
            pa=np.array(A['poses'][pose]['joint_anchors_thorax'][o['id']])*10;pb=np.array(B['poses'][pose]['joint_anchors_thorax'][j['id']])
            pairaxes.append(dict(current_joint=j['name'],official_candidate=c,pose=pose,official_axis_thorax=u.tolist(),current_axis_thorax=v.tolist(),dot_same_numeric_thorax_frame=float(u@v),unsigned_axis_angle_deg=float(np.degrees(np.arccos(np.clip(abs(u@v),0,1)))),official_anchor_cm_times10=pa.tolist(),current_anchor_native=pb.tolist(),anchor_delta_current_minus_official_times10=(pb-pa).tolist(),raw_qpos_difference_rad=float(B['poses'][pose]['qpos'][j['qposadr']]-A['poses'][pose]['qpos'][o['qposadr']]),warning='not registered homologous poses; raw difference is NOT an angle conversion offset'))
table('current42_candidates.csv',current);table('candidate_axes_and_offsets.csv',pairaxes)
official=[]
for a in sorted(A['actuators'],key=lambda r:r['action_index']):
    short=a['name'].split('/')[-1];leg=next((k for k,v in legs.items() if short.endswith(v)),None)
    target_joints=[]
    if a['target_kind']=='joint':target_joints=[a['target_name']]
    elif a['target_kind']=='tendon':target_joints=[w['name'] for w in A['tendons'][a['trnid'][0]]['wraps']]
    if short.startswith('coxa'):
        candidate=[r['current_joint'] for r in current if r['current_joint'].startswith('joint_'+leg+'Coxa')];reason='3-to-3 regional orientation candidate, no scalar angle identity'
    elif short.startswith('femur'):
        candidate=[r['current_joint'] for r in current if r['current_joint'].startswith('joint_'+leg+'Trochanter')];reason='2-to-3 regional orientation candidate, current extra rotational freedom'
    elif short.startswith('tibia'):
        candidate=['joint_'+leg+'Tibia_pitch'];reason='hinge region candidate; frame/sign/zero still uncalibrated'
    else:
        candidate=[];reason='no current adhesion actuator' if short.startswith('adhere') else 'current tarsal chain fixed' if short.startswith('tarsus') else 'current head/abdomen chain fixed'
    official.append(dict(policy_action_index=a['action_index'],ctrl_index=a['id'],official_actuator=a['name'],transmission=a['target_kind'],target=a['target_name'],target_joints=target_joints,
        tendon_coefficients=[w['parameter'] for w in A['tendons'][a['trnid'][0]]['wraps']] if a['target_kind']=='tendon' else [],candidate_current_joints=candidate,status='CANDIDATE_GROUP_ONLY' if candidate else 'UNMATCHED_CURRENT_BODY',reason=reason,
        ctrlrange=a['ctrlrange'],control_units='dimensionless adhesion command' if short.startswith('adhere') else 'rad or weighted angular tendon coordinate',dyntype=a['dyntype'],filter_tau_s=a['dynprm'][0],gainprm=a['gainprm'],biasprm=a['biasprm'],forcerange=a['forcerange'],forcelimited=a['forcelimited'],source=source_line(AXML,'adhesion' if short.startswith('adhere') else 'general',short)))
table('official59_actions.csv',official)
for label,model,xml in [('official',A,AXML),('current',B,BXML)]:
    allj=[]
    for j in model['joints']:
        row=dict(j);row['source']=source_line(xml,'joint',j['name'].split('/')[-1]);row['key0']=model['poses']['key0']['qpos'][j['qposadr']];allj.append(row)
    table(label+'_joints.csv',allj);table(label+'_actuators.csv',model['actuators']);table(label+'_geometry.csv',model['geoms']);table(label+'_body_frames.csv',model['bodys']);table(label+'_tendons.csv',model['tendons'])
landmarks=[];lengths=[]
for leg,suffix in legs.items():
    pairs=[('coxa','Coxa'),('femur','Trochanter' if leg in ('LF','RF') else 'Femur'),('tibia','Tibia'),('tarsus','Tarsus1'),('tarsus2','Tarsus2'),('tarsus3','Tarsus3'),('tarsus4','Tarsus4'),('claw','Tarsus5')]
    for pose in ('qpos0','key0'):
        aps=[];bps=[]
        for a,b in pairs:
            ai=ab['walker/'+a+'_'+suffix]['id'];bi=bb[leg+b]['id'];pa=np.array(A['poses'][pose]['body_positions_thorax'][ai]);pb=np.array(B['poses'][pose]['body_positions_thorax'][bi]);ra=np.array(A['poses'][pose]['body_rotations_thorax'][ai]);rb=np.array(B['poses'][pose]['body_rotations_thorax'][bi]);aps.append(pa);bps.append(pb)
            landmarks.append(dict(leg=leg,pose=pose,official_body=ab['walker/'+a+'_'+suffix]['name'],current_body=bb[leg+b]['name'],official_pos_cm=pa.tolist(),current_pos_native=pb.tolist(),delta_current_minus_10official=(pb-10*pa).tolist(),relative_frame_R_currentT_R_official=(rb.T@ra).tolist(),frame_rotation_angle_rad=float(np.arccos(np.clip((np.trace(rb.T@ra)-1)/2,-1,1))),status='candidate anatomical landmark; two independent rest poses, not calibrated zero offset'))
        if pose=='qpos0':
            for k in range(len(pairs)-1):
                la=np.linalg.norm(aps[k+1]-aps[k]);lb=np.linalg.norm(bps[k+1]-bps[k]);lengths.append(dict(leg=leg,segment=pairs[k][0]+'->'+pairs[k+1][0],official_length_cm=float(la),current_length_native=float(lb),ratio_current_over_official=float(lb/la),ratio_after_nominal_cm_to_mm=float(lb/(10*la))))
table('landmark_frame_offsets.csv',landmarks);table('segment_lengths.csv',lengths)
# Distribution of fixed/active anatomical regions is measured from compiled body joint counts.
regions=[]
for label,model,prefixes in [('official',A,['head','rostrum','haustellum','labrum','antenna','wing','abdomen','haltere','coxa','femur','tibia','tarsus','claw']),('current',B,['Head','Rostrum','Haustellum','A1A2','A3','A4','A5','A6','LWing','RWing'])]:
    for prefix in prefixes:
        bodies=[b for b in model['bodys'] if b['name'].startswith('walker/'+prefix) ] if label=='official' else [b for b in model['bodys'] if b['name']==prefix]
        regions.append(dict(model=label,region=prefix,bodies=[b['name'] for b in bodies],joint_count=sum(b['jntnum'] for b in bodies)))
table('articulation_gaps.csv',regions)
summary=dict(official=A['counts'],current=B['counts'],official_options=A['options'],current_options=B['options'],
    official_action_group_counts=dict(adhesion=6,head=3,abdomen_tendons=2,coxa=18,femur=12,tibia=6,tarsus=6,tarsal_tendons=6),
    official_transmission_counts={k:sum(r['transmission']==k for r in official) for k in ['joint','tendon','body']},
    current42_candidate_rows=len(current),official59_rows=len(official),official_actions_with_regional_candidates=sum(bool(r['candidate_current_joints']) for r in official),unmatched_official_actions=sum(not r['candidate_current_joints'] for r in official),verified_scalar_joint_maps=0,
    segment_ratio_nominal_min=min(r['ratio_after_nominal_cm_to_mm'] for r in lengths),segment_ratio_nominal_max=max(r['ratio_after_nominal_cm_to_mm'] for r in lengths),
    retained_measurement_calls={k:v['measured_calls'] for k,v in [('official',A),('current',B)]},
    cumulative_all_extraction_calls={k:2*(A['measured_calls'][k]+B['measured_calls'][k]) for k in A['measured_calls']},
    execution_note='Initial successful extraction revealed missing tendon target-kind decoder; corrected extractor reran both models. Counts include both runs. Initial superseded model JSON replaced; no numerical FK differences.',
    fd_max_abs_error={'official':A['finite_difference']['max_abs_error'],'current':B['finite_difference']['max_abs_error']},no_environment_created=True,environment_steps=0,training_updates=0)
dump('SUMMARY.json',summary)
source_hashes={**A['source_hashes'],**B['source_hashes']}
for q in [ROOT/'AGENTS.md',ROOT/'GOAL_ROADMAP.zh-CN.md',ROOT/'GOAL_TRACKER.zh-CN.md',OFF/'flybody/fruitfly/build_fruitfly/make_fruitfly.py']:
    source_hashes[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
dump('SOURCE_HASHES.json',source_hashes)
# Show thorax-relative native keyframe skeletons at explicit nominal 10x length conversion.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,6),subplot_kw={'projection':'3d'})
colors=['#d62728','#2ca02c','#1f77b4','#ff7f0e','#9467bd','#17becf']
for ax,(label,model,lookup,scale) in zip(axes,[('Official key0 (cm x10)',A,ab,10.),('Current key0 (native)',B,bb,1.)]):
    for color,(leg,suffix) in zip(colors,legs.items()):
        names=['walker/'+k+'_'+suffix for k in ['coxa','femur','tibia','tarsus','tarsus2','tarsus3','tarsus4','claw']] if label.startswith('Official') else [leg+k for k in ['Coxa','Trochanter' if leg in ('LF','RF') else 'Femur','Tibia','Tarsus1','Tarsus2','Tarsus3','Tarsus4','Tarsus5']]
        pts=np.array([model['poses']['key0']['body_positions_thorax'][lookup[n]['id']] for n in names])*scale
        ax.plot(*pts.T,'o-',color=color,label=leg,markersize=3)
    ax.scatter([0],[0],[0],color='black');ax.set(xlabel='thorax x',ylabel='thorax y',zlabel='thorax z',title=label);ax.view_init(25,-60);ax.set_box_aspect((1,1,1));ax.legend(fontsize=8)
fig.suptitle('Independent native reference poses; anatomical anchors only\nNo pose registration, simulation, contact success, or angle mapping claimed')
fig.tight_layout();fig.savefig(P/'KINEMATIC_REFERENCE.png',dpi=170);plt.close(fig)
print(json.dumps(summary,indent=2))
