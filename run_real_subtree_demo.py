#!/usr/bin/env python3
"""Explicit truncated positive-radius subtree; NOT repaired whole-cell morphology."""
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import run_real_morphology_demo as demo
from engine.cable import swc_to_morphology


def main():
    rejection = json.loads((demo.OUT/'real_morphology_metrics.json').read_text())
    candidate = next(a for a in rejection['selection']['attempts'] if a['reason'].startswith('invalid radii:'))
    with zipfile.ZipFile(demo.DATA/'fafb_skeletons_swc.zip') as z:
        with z.open(candidate['member']) as f:
            raw = f.read(demo.MAX_BYTES+1)
    assert hashlib.sha256(raw).hexdigest() == candidate['sha256']
    rows = [line.split('#',1)[0].split() for line in raw.decode('utf8').splitlines() if line.split('#',1)[0].strip()]
    assert 100 <= len(rows) <= 2500 and all(len(r)==7 for r in rows)
    by = {int(r[0]):r for r in rows}
    assert len(by)==len(rows)
    parents = {i:int(r[6]) for i,r in by.items()}
    assert all(p==-1 or p in by for p in parents.values())
    assert sum(p==-1 for p in parents.values())==1
    for start in by:
        seen=set(); i=start
        while i!=-1:
            assert i not in seen, 'source cycle'
            seen.add(i); i=parents[i]
    good={i for i,r in by.items() if np.isfinite(np.array(r[2:6],float)).all() and float(r[5])>0}
    adjacency={i:[] for i in good}
    for i in good:
        p=parents[i]
        if p in good:
            assert np.linalg.norm(np.array(by[i][2:5],float)-np.array(by[p][2:5],float))>0
            adjacency[i].append(p); adjacency[p].append(i)
    components=[]; visited=set()
    for start in sorted(good):
        if start in visited: continue
        component=[start]; visited.add(start)
        for i in component:
            for j in adjacency[i]:
                if j not in visited: visited.add(j); component.append(j)
        components.append(component)
    component=min(components,key=lambda c:(-len(c),min(c)))
    assert len(component)>=100, f'largest valid component only {len(component)} nodes'
    keep=set(component)
    boundary=[{'retained_id':i,'excluded_id':p,'direction':'excluded_parent'} for i,p in parents.items() if i in keep and p!=-1 and p not in keep]
    boundary += [{'retained_id':p,'excluded_id':i,'direction':'excluded_child'} for i,p in parents.items() if i not in keep and p in keep]
    subtree_rows=[]
    for r in rows:
        i=int(r[0])
        if i not in keep: continue
        row=r.copy()
        if parents[i] not in keep: row[6]='-1'
        subtree_rows.append(' '.join(row))
    source_path=demo.OUT/'real_morphology_subtree_source_rejected.swc'
    source_path.write_bytes(raw)
    sample=demo.OUT/'real_morphology_subtree_sample.swc'
    sample.write_text('# TRUNCATED VERIFIED-POSITIVE-RADIUS SUBTREE, not a whole neuron\n# Artificial sealed boundaries; no radius or coordinate repair\n# units: nm\n'+'\n'.join(subtree_rows)+'\n')
    m,ids=swc_to_morphology(sample,units='nm')
    for i in ids:
        row=next(r.split() for r in subtree_rows if int(r.split()[0])==i)
        assert row[:6]==by[i][:6]
        assert int(row[6])==parents[i] or (int(row[6])==-1 and parents[i] not in keep)
    result={'status':'running','implementation':'manually implemented/tested truncated-subtree prototype, not calibrated biology',
            'geometry_label':'Truncated real FAFB subtree',
            'figure_caption':'TRUNCATED VERIFIED-POSITIVE-RADIUS SUBTREE, not whole neuron or exact neck anatomy; artificial sealed boundaries.\nDNp01/DNp03 passive parameters, not whole-fly calibration. No real synapse positions; retained coordinates/radii/edges unchanged.',
            'caveats':demo.CAVEATS+['Explicit induced-subtree truncation; all edges to excluded nodes become artificial sealed boundaries. Not repair or validation of the rejected whole skeleton.'],
            'source':{**candidate,'source_id':candidate['source_id'],'sample':str(sample),'original_rejected_source':str(source_path),
                      'original_rejection_reason':candidate['reason'],'original_node_count':len(rows),'node_count':m.n,
                      'sample_sha256':hashlib.sha256(sample.read_bytes()).hexdigest(),'source_units':'nm','engine_units':'um',
                      'coordinate_and_radius_scale':.001,'archive':str(demo.DATA/'fafb_skeletons_swc.zip')},
            'truncation':{'selection':'first radius-rejected <=2500-node candidate, largest connected induced all-positive-radius component; tie by lowest ID',
                          'component_sizes':sorted(map(len,components),reverse=True),'excluded_source_ids':sorted(set(by)-keep),
                          'source_nonpositive_or_nonfinite_ids':sorted(set(by)-good),'artificial_sealed_boundary_edges':boundary,
                          'row_to_original_swc_id':ids,'retained_geometry_unchanged':True,
                          'boundary_assumptions':'Zero axial coupling to excluded source nodes; new component root uses engine 1e-6 um virtual-root membrane. Retained-child segment to excluded parent is omitted: not whole-neuron equivalent.'}}
    demo.PREFIX=demo.OUT/'real_morphology_subtree'
    demo.simulate(m,ids,result)
    result['status']='passed_truncated_subtree_only'
    target=demo.OUT/'real_morphology_subtree_metrics.json'
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'source_id':candidate['source_id'],'source_nodes':len(rows),'subtree_nodes':m.n,'boundary_edges':len(boundary),'metrics':result['simulation']['metrics'],'output':str(target)},indent=2))

if __name__=='__main__': main()
