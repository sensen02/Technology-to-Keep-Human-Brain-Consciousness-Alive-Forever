"""Independent source/current checker. Does not import command or probe runner.
Zero neural/physical steps. Retains existing full source graph audit unchanged.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

HERE=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=ROOT/'experiments/banc-physical-learning/20261107T195435Z'
ENC=ROOT/'experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def dump(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    ports=json.loads((HERE/'PORTS.json').read_text())
    source_hashes={x['path']:sha(x['path']) for x in ports['source_hashes'].values()}
    assert all(source_hashes[x['path']]==x['sha256'] for x in ports['source_hashes'].values())
    assert sha(HERE/'audit_ports.py')==ports['audit_script']['sha256']
    source=load(ports['source_hashes']['source_connectome']['path'])
    candidate=load(ports['source_hashes']['local_candidate']['path'])
    groups=load(ports['source_hashes']['groups']['path'])
    expected=[(1414,10567,720575941433155799,114),(13170,99854,720575941566493282,112)]
    strict=candidate['pure_verified_label_mask']
    pre=candidate['pre_local'][strict];post=candidate['post_local'][strict]
    for li,gi,rid,out in expected:
        assert int(candidate['cache_indices'][li])==gi and int(candidate['root_ids'][li])==rid
        assert int(source['root_ids'][gi])==rid
        assert source['ann_primary_type'][gi]=='DNp09' and source['ann_nt_verified'][gi]=='acetylcholine'
        assert source['ann_class'][gi]=='descending_neuron' and gi in groups['descending_boundary']
        assert int((pre==li).sum())==out
        rec=next(x for x in ports['candidates'] if x['identity']['local_index']==li)
        assert rec['identity']['root_id_string']==str(rid) and rec['outgoing']['selected_strict']['edge_pairs']==out
    rt=module('_round11_inputs_foundation_runtime',OLD/'foundation/runtime.py')
    c=rt.load_circuit()
    assert c.net.steps==2000 and c.net.t_ms==1000.
    assert np.array_equal(c.root_ids,candidate['root_ids'])
    graph=csr_matrix((np.ones(len(pre),dtype=np.uint8),(pre,post)),shape=(20000,20000))
    runtime_graph=(c.net.We+c.net.Wi).T.tocsr()
    runtime_graph.data[:]=1
    assert (graph!=runtime_graph).nnz==0
    local_mn=np.array([x['identity']['local_index'] for x in ports['exact_motor_target_records']],dtype=np.int64)
    assert len(local_mn)==391
    graph_checks=[]
    for direction,g in [('downstream',graph),('upstream',graph.T.tocsr())]:
        dist=shortest_path(g,method='D',directed=True,unweighted=True,indices=[1414,13170])
        for j,label in enumerate(['local1414','local13170']):
            actual=np.where(np.isfinite(dist[j,local_mn]),dist[j,local_mn],-1).astype(int)
            audit=np.array([x['local_strict_induced'][direction][label] for x in ports['exact_motor_target_records']])
            assert np.array_equal(actual,audit)
            graph_checks.append({'direction':direction,'DN':label,'exact_motor_distances':391})
    e=module('_round11_original_encoder',ENC).SensoryEncoder()
    policy_path=OLD/'learning/evaluation/zero/POLICY.npz';policy=load(policy_path)
    local=e.local_indices
    assert len(local)==556 and len(np.unique(local))==556 and not np.intersect1d(local,[1414,13170]).size
    assert np.isin(local,c.allowed_sensory_local).all() and np.isin([1414,13170],c.descending_local).all()
    assert not np.intersect1d(local,local_mn).size
    assert abs(float(policy['time_s'][0])-1.)<1e-8
    qvl=policy['local_physical_input'][:100].copy();saved=policy['sensory_current_nA'][:100].copy()
    # Independent feature equations, using source-qualified identities; no call to
    # encoder.encode in manual reconstruction. Keep original arithmetic order.
    names=list(e.joint_names);foot_order=['LF','LM','LH','RF','RM','RH']
    bounds=e.joint_ranges_rad;manual=[]
    for row in qvl:
        q,v,load6=row[:42],row[42:84],row[84:]
        normalized=(np.clip(q,bounds[:,0],bounds[:,1])-bounds[:,0])/(bounds[:,1]-bounds[:,0])
        current=np.zeros(20000)
        for leg in foot_order:
            knee=names.index('joint_'+leg+'Tibia_pitch')
            coxa=[names.index('joint_'+leg+'Coxa_'+axis) for axis in ['yaw','pitch','roll']]
            features={'claw_position':.01+.04*normalized[knee],
                      'hairplate_angle':.01+.04*normalized[coxa].mean(),
                      'hook_direction':.05*np.tanh(abs(v[knee])/10.),
                      'campaniform_strain':.05*np.tanh(max(load6[foot_order.index(leg)],0.)/(24.446354755878005/6.))}
            for r in e.qualified_records:
                if r['leg']==leg:current[r['local_index_0based']]=features[r['channel']]
        current=np.clip(current,0.,.06)
        assert np.array_equal(current,e.encode(q,v,load6))
        manual.append(current[local])
    assert np.array_equal(np.array(manual),saved)
    assert np.isfinite(saved).all() and np.all((saved>=0)&(saved<=.06))
    np.savez_compressed(HERE/'INPUTS.npz',sensory_local=local,sensory_recorded_nA=saved,
        sensory_held_nA=np.repeat(saved[:1],100,axis=0),source_q_v_load=qvl,source_time_s=policy['time_s'][:100],
        dn_local=np.array([1414,13170],dtype=np.int64),dn_root_ids=np.array([720575941433155799,720575941566493282],dtype=np.int64),
        all_mn_local=local_mn,all_mn_root_ids=c.root_ids[local_mn],mn323_local=c.mn_local,mn323_leg=c.mn_leg,
        old_zero_mn_start_hz=policy['mn_rates_hz'][:100],old_zero_mn_start_mV=policy['mn_voltages_mV'][:100],
        old_zero_mn_final_hz=policy['mn_rates_hz'][100],old_zero_mn_final_mV=policy['mn_voltages_mV'][100])
    # Per-MN labels are kept exactly from CSV; ambiguous 68 are never assigned a leg.
    with Path(ports['source_hashes']['motor_ports']['path']).open() as f:rows=list(csv.DictReader(f))
    dump(HERE/'MOTOR_IDENTITIES.json',rows)
    for p in [ENC,OLD/'foundation/runtime.py',OLD/'foundation/NEURAL_START.npz',policy_path,HERE/'PORTS.json',HERE/'PREREG.json',HERE/'command.py',HERE/'prepare_inputs.py']:
        source_hashes[str(p)]=sha(p)
    source_hashes.update({x['path']:sha(x['path']) for x in e.source_contract.values()})
    dump(HERE/'SOURCE_CHECK.json',{'status':'PASS','neural_steps':0,'physical_steps':0,'source_hashes':source_hashes,
        'source_identity_and_strict_counts_exact':True,'runtime_graph_equals_audited_graph':True,'graph_distance_checks':graph_checks,
        'encoder_manual_exact':True,'encoder_saved_exact':True,'sensory_cells':556,'DN_cells':2,'overlap':0,
        'all_leg_MN':391,'exported_sameleg_MN':323,'INPUTS_sha256':sha(HERE/'INPUTS.npz'),
        'canonical_steps':int(c.net.steps),'canonical_clock_ms':float(c.net.t_ms),
        'upstream_limit':'Local upstream graph reachability is not a demonstrated brain response; omitted brain cells remain absent.'})
    print('SOURCE_CHECK PASS: zero neural/physical steps, exact encoder and graph identities',flush=True)

if __name__=='__main__':main()
