"""Independent saved-only audit; imports no simulator or production runner."""
from pathlib import Path
import sys,json,hashlib,xml.etree.ElementTree as E
import numpy as np
P=Path('/run/media/sensen/Data2/cell_wound_prototype')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def named(root,selector):return {e.get('name'):e for e in root.findall(selector) if e.get('name')}
def main():
    o=Path(sys.argv[1]);base=E.parse(P/'outputs/muscles_six_legs/fruitfly_six_leg_standing.xml').getroot();new=E.parse(o/'INTEGRATED_MODEL.xml').getroot()
    for body_a,body_b in zip(base.findall('.//body'),new.findall('.//body')):
        assert body_a.get('name')==body_b.get('name')
        assert [E.tostring(x) for x in body_a.findall('inertial')]==[E.tostring(x) for x in body_b.findall('inertial')]
    assert len(base.findall('.//body'))==len(new.findall('.//body'))
    assert [E.tostring(x) for x in base.findall('./tendon/*')]==[E.tostring(x) for x in new.findall('./tendon/*')]
    for selector in ['.//site','./tendon/*']:
        a=named(base,selector);b=named(new,selector);assert a.keys()==b.keys()
        for n in a:assert a[n].attrib==b[n].attrib,(selector,n)
    for body_a,body_b in zip(base.findall('.//body'),new.findall('.//body')):
        n=body_a.get('name');allowed={'pos','quat'} if n in ('LWing','RWing') else set()
        assert {k:v for k,v in body_a.attrib.items() if k not in allowed}=={k:v for k,v in body_b.attrib.items() if k not in allowed},n
    a=named(base,'./actuator/*');b=named(new,'./actuator/*');assert len(a)==len(b)==90
    for n in a:
        assert {k:v for k,v in a[n].attrib.items() if k!='lengthrange'}=={k:v for k,v in b[n].attrib.items() if k!='lengthrange'},n
    a=named(base,'.//geom');b=named(new,'.//geom');assert a.keys()==b.keys()
    for n in a:assert a[n].attrib==b[n].attrib,n
    states=np.load(o/'BODY.npz');initial=np.load(o/'INITIAL.npz');t=states['time'];q=states['qpos'];assert q.shape[1]==49 and np.array_equal(q[0],initial['qpos'])
    for k in states.files:assert np.isfinite(states[k]).all(),k
    assert np.all(states['external_force']==0);assert not states['warnings'].any();np.testing.assert_allclose(states['ground_force_foot_world'].sum(1)+states['ground_force_nonfoot_world'],states['ground_force_world'],rtol=1e-12,atol=1e-12)
    key=np.fromstring(new.find('./keyframe/key').get('qpos'),sep=' ');np.testing.assert_array_equal(key,initial['qpos']);assert float(new.find('option').get('timestep'))==.0000125
    rows=json.loads((o/'MAPPING.json').read_text())['rows'];u=states['command'];ties={};unresolved=[]
    for r in rows:
        indices=r['local_indices'];i=r['actuator_index']
        if not indices:unresolved.append(i);assert not u[:,i].any()
        else:ties.setdefault(tuple(sorted(indices)),[]).append(i)
    for group in ties.values():
        for i in group[1:]:assert np.array_equal(u[:,group[0]],u[:,i])
    assert len(unresolved)==16 and len(ties)==66
    ex=json.loads((o/'EXECUTION.json').read_text());assert ex['exit_code']==0 and ex['counts']['rollouts']==1 and ex['counts']['network_instances']==1
    assert ex['counts']['step']==80000 and ex['counts']['neural_step']==2000 and len(t)==80001
    np.testing.assert_allclose(t,np.arange(len(t))*.0000125,atol=2e-12,rtol=0)
    prereg=json.loads((o/'PREREGISTRATION.json').read_text());assert all(sha(Path(k))==v for k,v in prereg['source_hashes'].items())
    result=dict(passed=True,saved_states=len(t),physics_steps=0,neural_steps=0,checks=['original site/tendon/geom attributes unchanged','all90 actuator attributes except declared lengthrange unchanged','initial qpos equals persisted model keyframe and BODY0','finite full states, no external force/warnings','worldforce partition','66poolties and16unresolvedzero','1live network/1rollout counts80000body2000neural','input source hashes unchanged'],note='No new simulation; explicit inertial nodes compared by matched body order, including unnamed nodes.')
    print(json.dumps(result,indent=2));dest=o/'PARENT_SAVED_VERIFICATION.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
