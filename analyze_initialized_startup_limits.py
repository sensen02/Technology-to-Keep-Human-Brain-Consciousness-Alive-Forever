"""Offline XML + saved-array joint localization; no physics backend calls."""
from pathlib import Path
import csv
import json
import hashlib
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OUT=Path('/home/sensen/Desktop/engine修复验收/s109-initialized-translation-training/20261024T045944Z')
MODEL=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
MANIFEST=Path('/run/media/sensen/Data2/engine修复验收/s4-endpoints/20261023T011559Z/MANIFEST.json')


def main():
    from engine.embodied.motor_translation import FiniteMotorTranslator
    xml=ET.parse(MODEL).getroot(); assert not xml.findall('.//include')
    nodes=list(xml.find('worldbody').iter('joint'))
    # Free root is a <joint type=free>, not an assumed six-leg display ordering.
    joints=[]; qadr=vadr=0
    for j in nodes:
        typ=j.get('type','hinge'); name=j.get('name')
        if typ=='free': qadr+=7;vadr+=6;continue
        assert typ=='hinge' and j.get('range') is not None
        joints.append(dict(name=name,qadr=qadr,vadr=vadr,range=list(map(float,j.get('range').split()))));qadr+=1;vadr+=1
    assert (qadr,vadr)==(49,48) and len(joints)==42
    parent={child:par for par in xml.iter() for child in par}
    sites={s.get('name'):s for s in xml.findall('.//site') if s.get('name')}
    def ancestor_joints(site):
        ans=set(); node=sites[site]
        while node in parent:
            node=parent[node]
            if node.tag=='body':ans.update(j.get('name') for j in node.findall('joint'))
        return ans
    tendons={}
    for t in xml.find('tendon'):
        assert t.tag=='spatial'
        anchors=[s.get('site') for s in t.findall('site')]
        chains=[ancestor_joints(s) for s in anchors]
        # A joint moves some but not all tendon anchors: structural crossing,
        # not a computed nonzero signed moment arm at this pose.
        union=set.union(*chains);common=set.intersection(*chains)
        tendons[t.get('name')]=dict(sites=anchors,crossing=sorted(union-common))
    acts=list(xml.find('actuator')); assert len(acts)==90
    manifest=json.loads(MANIFEST.read_text()); dec=FiniteMotorTranslator(manifest)
    mapping=[]
    for i,a in enumerate(acts):
        assert a.get('tendon') in tendons
        t=tendons[a.get('tendon')]
        mapping.append(dict(index=i,name=a.get('name'),tendon=a.get('tendon'),structural_crossing_joints=t['crossing'],
            pool=int(dec.actuator_pool[i]),pool_local_indices=list(dec.pools[dec.actuator_pool[i]]) if dec.actuator_pool[i]>=0 else [],
            note='anchor ancestry structural candidate; signed pose-specific moment arm NOT_SAVED'))
    def plain(v):
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    rows=[]; data={}; hashes={str(MODEL):hashlib.sha256(MODEL.read_bytes()).hexdigest(),str(MANIFEST):hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}
    for label in ('baseline','plus','minus','updated'):
        path=OUT/label/'TRAJECTORY.npz';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as z:data[label]={k:z[k][:101].copy() for k in ('time','qpos','qvel','foot_force','actuator_force','act','ctrl')}
        a=data[label]
        for j in joints:
            q=a['qpos'][:,j['qadr']];v=a['qvel'][:,j['vadr']];lo,hi=j['range']
            excess=np.maximum(np.maximum(lo-q,q-hi),0); failures=np.flatnonzero(excess>1e-6); imax=int(excess.argmax())
            candidates=[m for m in mapping if j['name'] in m['structural_crossing_joints']]
            row=dict(arm=label,joint=j['name'],side='left' if j['name'].lower().startswith('joint_l') else 'right',
                qpos_index=j['qadr'],qvel_index=j['vadr'],lower=lo,upper=hi,
                first_fail_ms=float(a['time'][failures[0]]*1000) if len(failures) else None,
                first_direction=('lower' if q[failures[0]]<lo else 'upper') if len(failures) else None,
                first_q=float(q[failures[0]]) if len(failures) else None,
                first_velocity=float(v[failures[0]]) if len(failures) else None,
                max_excess=float(excess[imax]),max_at_ms=float(a['time'][imax]*1000),velocity_at_max=float(v[imax]),
                structural_actuator_indices=[m['index'] for m in candidates],structural_actuator_names=[m['name'] for m in candidates])
            rows.append(row)
    failing=[r for r in rows if r['first_fail_ms'] is not None]
    assert failing
    target=min((r for r in failing if r['arm']=='baseline'),key=lambda r:r['first_fail_ms'])
    targetj=next(j for j in joints if j['name']==target['joint'])
    channels=[m for m in mapping if target['joint'] in m['structural_crossing_joints']]
    evidence={}
    for label,a in data.items():
        evidence[label]=dict(channels=[dict(index=m['index'],name=m['name'],pool=m['pool'],
            ctrl_at_0p1ms=float(a['ctrl'][1,m['index']]),activation_at_1p7ms=float(a['act'][17,m['index']]),
            force_at_1p7ms=float(a['actuator_force'][17,m['index']]),force_at_2p7ms=float(a['actuator_force'][27,m['index']])) for m in channels])
    record=dict(joint_order=joints,all_joint_rows=rows,violating_joints=failing,first_baseline=target,structural_mapping=mapping,
        target_channel_evidence=evidence,source_hashes=hashes,physics_calls=0,neural_ticks=0,
        missing=['per-pose actuator_moment','qfrc_actuator','per-contact Jacobians and wrench decomposition'],
        conclusion='Saved actuator_force is real muscle force, not ctrl. No signed moment arms or qfrc_actuator saved; no joint actuator torque computed. Structural crossing does not prove nonzero/sign of moment arm or causality of support loss.')
    (OUT/'EARLY_JOINT_LOCALIZATION.json').write_text(json.dumps(record,indent=2,default=plain)+'\n')
    with (OUT/'EARLY_JOINT_LIMITS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(12,12),sharex=True)
    for label,a in data.items():
        t=a['time']*1000; qi=targetj['qadr'];vi=targetj['vadr']
        axes[0].plot(t,a['qpos'][:,qi],label=label);axes[1].plot(t,a['qvel'][:,vi],label=label)
        axes[2].plot(t,a['foot_force'][:,:,2].sum(1)/24.446354755878005,label=label)
    axes[0].axhline(targetj['range'][1],color='black',ls=':');axes[0].axhline(targetj['range'][0],color='black',ls=':')
    for m in channels:
        axes[3].plot(data['baseline']['time']*1000,data['baseline']['actuator_force'][:,m['index']],label=m['name'])
    for ax,name in zip(axes,('Joint angle (rad)','Joint angular velocity (rad/s)','Foot Fz / native weight','Saved actuator force (native)')):
        ax.set_ylabel(name);ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
    axes[2].axhline(.8,color='black',ls=':');axes[-1].set_xlabel('Physical time after release (ms)')
    fig.suptitle('First violating joint: '+target['joint']+'\nStructural muscle candidates only; signed moment arms NOT SAVED; no torque/causal claim')
    fig.tight_layout();fig.savefig(OUT/'EARLY_JOINT_LOCALIZATION.png',dpi=150)
    print(json.dumps(dict(first=target,violations=failing,channels=channels),default=plain))


if __name__=='__main__':main()
