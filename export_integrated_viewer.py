"""Bounded offline spatial export; separate coordinate frames, not registration."""
from pathlib import Path
import json, os, resource
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'viewer'

def rounded(x,dec=5): return np.round(np.asarray(x),dec).tolist()
def samples(n,limit=100): return np.unique(np.linspace(0,n-1,min(limit,n),dtype=int))
def body_row_mapping(be):
    """Resolve FlyGym's documented segment order, never infer it from row count.

    MuJoCo may fuse fixed bodies: FlyGym currently exposes -1 for c_head,
    inadvertently returning the last body's pose. Such rows are NOT head poses.
    Export compiled model order (world included) and omit those invalid rows.
    """
    import mujoco
    m=be.model
    segments=list(be.fly.get_bodysegs_order())
    ids=np.asarray(be.sim._internal_bodyids_by_fly[be.fly.name],dtype=int)
    if len(ids)!=len(segments): raise RuntimeError('FlyGym segment/ID order mismatch')
    mapping={}; dropped=[]; records=[]
    for row,(seg,bid) in enumerate(zip(segments,ids)):
        source=be.fly.bodyseg_to_mjcfbody[seg]
        resolved=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,source.name)
        if resolved!=bid: raise RuntimeError('FlyGym internal body ID disagrees with source name')
        records.append({'recorded_row':row,'segment':seg.name,'model_body_id':int(bid)})
        if bid<0:
            # Only a fixed source body with an identifiable compiled parent and
            # retained geom can be safely omitted; reject every ambiguous case.
            parent=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,source.parent.name)
            geom=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,source.name)
            if bid!=-1 or list(source.joints) or parent<=0 or geom<0 or int(m.geom_bodyid[geom])!=parent:
                raise RuntimeError(f'ambiguous missing source body {source.name}')
            dropped.append({'recorded_row':row,'name':source.name,'reason':'fixed body fused into parent; FlyGym -1 aliases last body','compiled_parent_id':parent})
        elif bid==0 or bid in mapping:
            raise RuntimeError('ambiguous duplicated/world FlyGym body row')
        else: mapping[int(bid)]=row
    if set(mapping)!=set(range(1,m.nbody)):
        raise RuntimeError('recorded FlyGym rows do not cover compiled bodies exactly')
    return mapping,{'method':'get_bodysegs_order + _internal_bodyids_by_fly + source-name validation',
                    'output_order':'MuJoCo compiled model order; world identity synthesized',
                    'recorded_order':records,'dropped_invalid_rows':dropped}


def canonical_body_poses(be,positions,rotations):
    mapping,audit=body_row_mapping(be)
    p=np.asarray(positions);q=np.asarray(rotations)
    n=len(audit['recorded_order'])
    if p.ndim!=3 or p.shape[1:]!=(n,3) or q.shape!=(len(p),n,4):
        raise RuntimeError('recorded pose dimensions disagree with documented segment order')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q,axis=-1),1,atol=1e-5):
        raise RuntimeError('invalid recorded body pose values')
    for item in audit['dropped_invalid_rows']:
        row=item['recorded_row']; last=mapping[be.model.nbody-1]
        if not np.array_equal(p[:,row],p[:,last]) or not np.array_equal(q[:,row],q[:,last]):
            raise RuntimeError('missing-body row is not the verified FlyGym negative-index alias; refusing ambiguous recording')
    outp=np.zeros((len(p),be.model.nbody,3));outq=np.zeros((len(p),be.model.nbody,4));outq[:,:,0]=1
    for bid,row in mapping.items(): outp[:,bid]=p[:,row];outq[:,bid]=q[:,row]
    return outp,outq,audit


def export_body(be,ep):
    import mujoco
    m=be.model
    node_ids=list(range(m.nbody));node_map={i:i for i in node_ids}
    # Validate the complete recording, not just its sampled subset.
    pos,rot,audit=canonical_body_poses(be,ep['truth/body_positions_mm'],ep['truth/body_rotations_wxyz'])
    if not np.allclose(pos[:,m.body(be.fly.name+'/c_thorax').id],ep['truth/thorax_mm'],atol=1e-10):
        raise RuntimeError('recorded thorax channel disagrees with resolved row mapping')
    ix=samples(len(ep['truth/time_s']));pos=pos[ix];rot=rot[ix]
    names=[m.body(i).name for i in node_ids]
    body={'nodes':[{'name':n,'position':p.tolist(),'radius_mm':.025} for n,p in zip(names,pos[0])],
          'links':[[int(m.body_parentid[i]),i] for i in node_ids if i!=0],
          'frames':rounded(pos), 'rotations_wxyz':rounded(rot,7),'times_s':rounded(ep['truth/time_s'][ix]),
          'contact_flags':ep['truth/contact_present'][ix].tolist(),
          'descending_rate_hz':rounded(ep['observed/descending_rate_hz'][ix]),
          'provenance':'SIMULATED recorded FlyGym body; demo CPG + illustrative neural modulation, NOT measured animal',
          'coordinate_frame':'FlyGym world mm; no BANC anatomical registration','meshes':[],
          'row_mapping_audit':audit}
    vertices_total=0
    for gi in range(m.ngeom):
        bid=int(m.geom_bodyid[gi]);mid=int(m.geom_dataid[gi])
        if int(m.geom_type[gi])!=int(mujoco.mjtGeom.mjGEOM_MESH):continue
        start=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);fs=int(m.mesh_faceadr[mid]);nf=int(m.mesh_facenum[mid])
        if vertices_total+nv>250000:raise RuntimeError('mesh export vertex cap')
        v=np.asarray(m.mesh_vert[start:start+nv]);mat=np.empty(9)
        mujoco.mju_quat2Mat(mat,np.asarray(m.geom_quat[gi]));local=v@mat.reshape(3,3).T+np.asarray(m.geom_pos[gi])
        body['meshes'].append({'name':m.geom(gi).name,'body_index':node_map[bid],'local_vertices_mm':rounded(local,6),
                               'indices':m.mesh_face[fs:fs+nf].reshape(-1).tolist()})
        vertices_total+=nv
    return body,vertices_total


def main():
    OUT.mkdir(exist_ok=True)
    # Body reconstruction from actual FlyGym model, no replacement physics.
    from engine.embodied.body_backend import BodyBackend,BodyConfig
    import mujoco
    be=BodyBackend(BodyConfig(),gl_backend=None); m=be.model
    with np.load(ROOT/'outputs/embodied_body/loop_episode_neural_modulated_seed0.npz',allow_pickle=False) as ep:
        body,vertices_total=export_body(be,ep)
    be.close()
    # Soma table only; no huge connectome build.
    import pyarrow.parquet as pq
    table=pq.read_table(ROOT/'data/banc/somas_v1.parquet',columns=['pt_root_id','pt_position'])
    ids=np.asarray(table['pt_root_id']); xyz=np.asarray(table['pt_position'].to_pylist(),dtype=float)*[.004,.004,.045]
    old=json.loads((ROOT/'outputs/embodied_body/access_map.json').read_text())
    old_centers=np.asarray(old['primary_build']['array']['channel_centres_um']);center=old_centers.mean(axis=0)
    x,y=np.meshgrid((np.arange(16)-7.5)*20,(np.arange(16)-7.5)*20)
    electrodes=np.c_[x.ravel()+center[0],y.ravel()+center[1],np.full(256,center[2])]
    # Select bounded nearby somas, then distributed context; root IDs string for JS precision.
    dist=np.linalg.norm(xyz-center,axis=1); order=np.argsort(dist)[:6000]; xyz=xyz[order];ids=ids[order]
    from engine.neural_cond import ConductanceNetwork
    net=ConductanceNetwork(len(xyz),dt_ms=.1)
    volts=[];activity=[]; count=np.zeros(len(xyz));nearest=dist[order]
    for k in range(10000):
        phase=2*np.pi*(k*.0001*3+nearest/200)
        drive=.024+.012*np.sin(phase)
        count+=net.step(current_nA=drive)
        if (k+1)%100==0:
            activity.append(rounded(count/ .01,2));volts.append(rounded(net.v,3));count[:]=0
    neural={'positions_um':rounded(xyz,3),'ids':[str(v) for v in ids], 'activity':activity,'membrane_mV':volts,
            'times_s':rounded(np.arange(1,101)*.01),'units':'Hz per10ms window',
            'provenance':'ILLUSTRATIVE independent ConductanceNetwork cells; real soma coordinates, NO connectome or measured neural activity',
            'coordinate_frame':'BANC 4x4x45nm voxel conversion; not registered to FlyGym body',
            'activity_assumptions':{'drive_nA':[.012,.036],'parameters':net.p.provenance(),'connectivity':'none'},
            'attribution':old['attribution']}
    ca=np.load(ROOT/'outputs/control.npz',allow_pickle=False);ci=samples(len(ca['times']))
    calcium={'positions_um':rounded(np.c_[ca['positions'],np.zeros(len(ca['positions']))],3),
             'values':rounded(ca['c'][ci],5),'er_values':rounded(ca['er'][ci],5),'ip3_values':rounded(ca['ip3'][ci],5),
             'times_s':rounded(ca['times'][ci]),'units':'µM cytosolic Ca; ER µM per ER volume; IP3 µM',
             'ablated':ca['ablated'].tolist(),'damage':rounded(ca['damage'],4),
             'provenance':'SIMULATED epithelial wound model, NOT brain calcium; ablated cells imposed1000µM reservoirs, NOT viable cells',
             'coordinate_frame':'separate_epithelial_model','metadata':json.loads(ca['metadata_json'].item())}
    payload={'meta':{'status':'hand-built research viewer, not experimental measurements',
             'registration':'Three independent coordinate frames juxtaposed; NO anatomical registration or causal time synchronization',
             'memory_limits':{'neurons':6000,'frames':100,'body_vertices':vertices_total},'source_root':str(ROOT)},
             'body':body,'neural':neural,'electrodes':{'positions_um':rounded(electrodes,3),'diameter_um':7,'pitch_um':20,
             'capture_radius_um':50,'provenance':'Fixed user geometry7/20; centered on previous placement; inserted shaft direction/depth only schematic',
             'reference_um':rounded(center+[0,0,500]),'return_um':rounded(center+[0,0,-500])},'calcium':calcium,
             'bench':(json.loads((ROOT/'outputs/bench/status.json').read_text())
                      if (ROOT/'outputs/bench/status.json').exists() else
                      {'status':'NOT_RUN','note':'No measured hardware data yet'})}
    payload['bench']['chain']=[
        {'node':'electrode','model':'fixed7µm/20µm geometric disc approximation','measured':'NOT_RUN','gap':'shaft vs exposed active area needs evidence'},
        {'node':'interface','model':'Randles assumedCdl20µF/cm² rhoct385Ωcm² sigma0.3S/m','measured':'NOT_RUN'},
        {'node':'amplifier','model':'assumed voltage10nV/√Hz current10fA/√Hz;1/f corner100Hz','measured':'NOT_RUN'},
        {'node':'analog','model':'ButterworthHP2+LP4 before sampling; named recording bands','measured':'NOT_RUN'},
        {'node':'ADC','model':'ideal30kHzsampling only','gap':'clipping/quantization/jitter not implemented','measured':'NOT_RUN'},
        {'node':'digital','model':'causalSOSbandpass; not anti-alias','measured':'NOT_RUN'}]
    text='window.VIEWER_DATA='+json.dumps(payload,separators=(',',':'),allow_nan=False)+';\n'
    if len(text)>35*1024**2:raise MemoryError('viewer payload cap35MiB')
    (OUT/'data.js').write_text(text)
    report={'payload_bytes':len(text),'body_vertices':vertices_total,'neurons':len(xyz),'frames':100,
            'calcium_cells':len(ca['positions']),'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'all_coordinates_finite':bool(np.isfinite(xyz).all()),'fixed_geometry':{'diameter_um':7,'pitch_um':20},
            'limitations':payload['meta']['registration']}
    (OUT/'export_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
