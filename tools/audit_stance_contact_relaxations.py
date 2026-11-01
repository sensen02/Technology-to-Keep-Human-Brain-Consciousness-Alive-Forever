"""S2/S5 fixed-pose contact relaxations. Offline only; no integration.
Run: venv_body/bin/python -B tools/audit_stance_contact_relaxations.py
Independent array-only verification: same command with --verify (no MuJoCo import).
"""
from pathlib import Path
import hashlib,json,sys,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
O=Path('/home/sensen/Desktop/engine修复验收/s5-stance-contact-relaxations/20261101T205953Z')
S=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
POSES=[Path('/home/sensen/Desktop/engine修复验收/s5-stance-lp-feasibility/20261101T205257Z/BEST_LP.npz'),Path('/home/sensen/Desktop/engine修复验收/s5-equilibrium-wider-stance/20261101T205257Z/SEED_0.9_FINAL_STATE.npz')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def certificate(a):
    x,C,r,c,lo,hi=[a[k] for k in ('solution','C','rhs','objective','lower','upper')]
    z,l,u=[a[k] for k in ('ineqlin_marginals','lower_marginals','upper_marginals')]
    fl=np.isfinite(lo);fu=np.isfinite(hi)
    primal=float(c@x);dual=float(r@z+lo[fl]@l[fl]+hi[fu]@u[fu])
    return dict(primal=primal,dual=dual,gap=abs(primal-dual),primal_violation=float(max(0,np.max(C@x-r),np.max(lo[fl]-x[fl],initial=0),np.max(x[fu]-hi[fu],initial=0))),stationarity=float(abs(c-C.T@z-l-u).max()),dual_sign_violation=float(max(0,z.max(initial=0),-l.min(initial=0),u.max(initial=0))),complementarity=float(max(abs(z*(r-C@x)).max(initial=0),abs(l[fl]*(x[fl]-lo[fl])).max(initial=0),abs(u[fu]*(hi[fu]-x[fu])).max(initial=0))))
def verify():
    result=json.loads((O/'RESULT.json').read_text());checks=[]
    for row in result['solutions']:
        a=np.load(O/row['file']);cert=certificate(a)
        assert max(cert[k] for k in ('gap','primal_violation','stationarity','dual_sign_violation','complementarity'))<1e-7,(row,cert)
        n=int(a['nforce']);nv=a['B'].shape[1];x=a['solution'];W=float(a['W']);G=a['G']
        assert np.max(abs(a['B']@x[:nv]-a['y']-a['residual']))<1e-12
        assert abs(np.sum(abs(a['residual']))-cert['primal'])<1e-7
        assert np.max(abs(a['B'][:,:66]-.9999*a['A']@G/W))<1e-12
        assert np.max(abs(a['y']-(a['bias']-a['joint_passive']-a['baseline'])/W))<1e-12
        assert np.max(abs(a['B'][:,66:66+3*n]-a['J'].transpose(2,0,1).reshape(48,3*n)/W),initial=0)<1e-12
        if row['mode']=='limits':assert np.max(abs(a['B'][:,66+3*n:]-a['limit_J'].T/W),initial=0)<1e-12
        f=x[66:66+3*n].reshape(-1,3);mu=a['mu'];assert f[:,2].min(initial=0)>-1e-8
        tang=np.sum(abs(f[:,:2]),axis=1) if row['friction']=='diamond' else np.max(abs(f[:,:2]),axis=1)
        assert np.max(tang-mu*f[:,2],initial=0)<1e-7
        assert np.max(abs(a['activation']-(.0001+.9999*G@x[:66])))<1e-12
        if row['mode']!='signed':assert a['activation'].min()>=.0001-1e-9 and a['activation'].max()<=1+1e-9
        checks.append(dict(file=row['file'],certificate=cert))
    for pose in range(2):
        p=np.load(O/f'POSE_{pose}.npz');src=np.load(POSES[pose]);assert np.array_equal(p['qpos'],src['qpos'])
        assert np.array_equal(p['qvel'],np.zeros(48));assert np.sum(p['G'].sum(axis=1)==0)==16
        assert np.max(abs(p['A']-p['moment'].T*(p['f1']-p['f0'])))<1e-12
        assert np.max(abs(p['baseline']-(p['moment'].T@p['f0']+p['A']@np.full(90,.0001))))<1e-12
        meta=json.loads((O/f'POINTS_{pose}.json').read_text());sets=meta['sets']
        assert set(sets['a'])<=set(sets['c'])<=set(sets['d']);assert set(sets['b'])<=set(sets['d'])
        for i in sets['d']:assert meta['points'][i]['mesh_min_gap']<=.001+1e-12
        vertices=np.load(O/f'VERTICES_{pose}.npz')
        for mesh in meta['meshes']:
            g=mesh['geom'];ids=np.flatnonzero(vertices['geom']==g);gaps=vertices['world'][ids,2]-float(p['floor_z'])
            assert np.allclose(gaps,vertices['gap'][ids],atol=1e-15)
            for idx in mesh['retained_global_vertex_rows']:assert vertices['gap'][idx]<=.001+1e-12
        rows=[r for r in result['solutions'] if r['pose']==pose];lookup={(r['set'],r['friction'],r['mode']):r['l1'] for r in rows}
        for fr in ('diamond','square'):
            for small,large in [('a','c'),('a','d'),('b','d'),('c','d')]:assert lookup[(large,fr,'physiological')]<=lookup[(small,fr,'physiological')]+1e-7
            assert lookup[('d',fr,'signed')]<=lookup[('d',fr,'physiological')]+1e-7
            if ('d',fr,'limits') in lookup:assert lookup[('d',fr,'limits')]<=lookup[('d',fr,'physiological')]+1e-7
        for name in 'abcd':assert lookup[(name,'square','physiological')]<=lookup[(name,'diamond','physiological')]+1e-7
    for path,h in result['source_sha256'].items():assert sha(Path(path))==h
    for a in result['model_array_identity'].values():assert a['before']==a['after']
    dump(O/'VERIFICATION.json',dict(passed=True,counts=dict(model_load=0,forward=0,LP=0),checks=checks,nesting='all declared L1 objective inclusions pass; infinity residual is not monotonic requirement'))
    print(json.dumps(dict(verified=len(checks),passed=True)))
def main():
    import mujoco
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    from flygym.compose.fly.musculoskeletal import _load_mjcf
    assert (O/'PREREGISTRATION.json').exists() and not (O/'RESULT.json').exists()
    sources=[Path(__file__),S/'MODEL.xml',S/'MAPPING.json',*POSES,O/'PREREGISTRATION.json']
    hashes={str(p):sha(p) for p in sources};assert sha(S/'MODEL.xml')=='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6'
    (O/'EXECUTED_audit_stance_contact_relaxations.py').write_bytes(Path(__file__).read_bytes())
    m=_load_mjcf(str(S/'MODEL.xml')).compile();d=mujoco.MjData(m)
    def modelhash():return {k:hashlib.sha256(getattr(m,k).tobytes()).hexdigest() for k in dir(m) if isinstance(getattr(m,k),np.ndarray)}
    before=modelhash();counts=dict(model_load=1,forward=0,LP=0,jac=0,body_step=0,neural_step=0,pose_optimization=0)
    groups={}
    for row in json.loads((S/'MAPPING.json').read_text())['rows']:
        if row['local_indices']:groups.setdefault(tuple(row['local_indices']),[]).append(row['actuator_index'])
    G=np.zeros((90,66));assert len(groups)==66
    for k,ix in enumerate(groups.values()):G[ix,k]=1
    assert np.sum(G.sum(axis=1)==0)==16 and m.nv==48
    floor=m.geom('floor').id;feet=[g for g in range(m.ngeom) if 'Tars' in (m.geom(g).name or '')];assert len(feet)==30
    W=float(m.body_mass.sum()*abs(m.opt.gravity[2]));results=[];posemeta=[]
    for pi,source in enumerate(POSES):
        q=np.load(source)['qpos'].copy()
        def forward(act):
            mujoco.mj_resetData(m,d);d.qpos[:]=q;d.qvel[:]=0;d.act[:]=act;d.ctrl[:]=act
            mujoco.mj_forward(m,d);counts['forward']+=1
        forward(np.zeros(90));f0=d.actuator_force.copy();bias=d.qfrc_bias.copy();jp=d.qfrc_passive.copy();moment=np.zeros((90,48))
        if np.asarray(d.actuator_moment).shape==(90,48):moment[:]=d.actuator_moment
        else:
            for i in range(90):
                adr=int(d.moment_rowadr[i]);n=int(d.moment_rownnz[i]);moment[i,d.moment_colind[adr:adr+n]]=d.actuator_moment[adr:adr+n]
        forward(np.ones(90));f1=d.actuator_force.copy();A=moment.T*(f1-f0);base=moment.T@f0+A@np.full(90,.0001)
        probe=np.linspace(.01,.99,90);forward(probe);error=float(abs(moment.T@f0+A@probe-d.qfrc_actuator).max());assert error<1e-8
        floorz=float(d.geom_xpos[floor,2]);assert np.allclose(d.geom_xmat[floor].reshape(3,3),np.eye(3))
        # Capture actual active limit rows. MuJoCo convention: generalized force J.T*lambda, lambda>=0.
        if mujoco.mj_isSparse(m):
            efc=np.zeros((d.nefc,48))
            for r in range(d.nefc):
                adr=int(d.efc_J_rowadr[r]);n=int(d.efc_J_rownnz[r]);efc[r,d.efc_J_colind[adr:adr+n]]=d.efc_J[adr:adr+n]
        else:efc=np.asarray(d.efc_J).reshape(d.nefc,48).copy()
        limrows=np.flatnonzero(d.efc_type==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT));L=efc[limrows]
        vertices=[];vgeom=[];vgap=[];meshes=[];points=[];sets={k:[] for k in 'abcd'}
        def add(g,pos,gap,origin,mingap,extra=None):
            # Exact duplicate point+body identity gives a proper set union; memberships retained.
            key=(g,*map(float,pos));existing=next((i for i,p in enumerate(points) if tuple(p['key'])==key),None)
            if existing is None:
                priority=m.geom_priority[[floor,g]];fr=m.geom_friction[[floor,g],0]
                mu=float(max(fr) if priority[0]==priority[1] else fr[int(priority[1]>priority[0])])
                body=int(m.geom_bodyid[g]);j=np.zeros((3,48));jr=np.zeros_like(j);mujoco.mj_jac(m,d,j,jr,np.asarray(pos),body);counts['jac']+=1
                points.append(dict(key=key,geom=g,geom_name=m.geom(g).name,body=body,body_name=m.body(body).name,pos=list(pos),gap=float(gap),mesh_min_gap=float(mingap),mu=mu,priority=priority.tolist(),geom_friction=m.geom_friction[[floor,g]].tolist(),normal=[0,0,1],origins=[],J=j.tolist()));existing=len(points)-1
            points[existing]['origins'].append(dict(origin=origin,gap=float(gap),extra=extra))
            return existing
        min_gaps={}
        for g in feet:
            mesh=int(m.geom_dataid[g]);adr=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh]);world=m.mesh_vert[adr:adr+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];gaps=world[:,2]-floorz;imin=int(np.argmin(gaps));min_gaps[g]=float(gaps[imin]);start=len(vertices)
            vertices.extend(world);vgeom.extend([g]*n);vgap.extend(gaps)
            valid=np.flatnonzero(gaps<=.001);selected=[]
            if len(valid):
                xy=world[valid,:2];unique,uid=np.unique(xy,axis=0,return_index=True);ids=valid[uid]
                if len(unique)>=3 and np.linalg.matrix_rank(unique-unique[0])==2:selected=ids[ConvexHull(unique).vertices].tolist()
                elif len(unique)>1:
                    axis=int(np.argmax(np.ptp(unique,axis=0)));selected=ids[[int(np.argmin(unique[:,axis])),int(np.argmax(unique[:,axis]))]].tolist()
                else:selected=ids.tolist()
                selected=sorted(set(selected+[imin]));p=world[imin].copy();p[2]=floorz;sets['a'].append(add(g,p,gaps[imin],'a_minimum_vertex',gaps[imin],dict(vertex=imin)))
                for vi in selected:
                    p=world[vi].copy();p[2]=floorz;sets['c'].append(add(g,p,gaps[vi],'c_hull_or_mandatory_a',gaps[imin],dict(vertex=vi)))
            meshes.append(dict(geom=g,name=m.geom(g).name,mesh=mesh,compiled_vertex_address=adr,vertex_count=n,minimum_gap=float(gaps[imin]),minimum_vertex=imin,eligible_vertices=valid.tolist(),retained_local_vertices=selected,retained_global_vertex_rows=[start+i for i in selected],rule='gap<=.001, unique xy convex hull extremes + mandatory a; collinear endpoints; outside-gap vertices rejected; other eligible interior/duplicate vertices omitted'))
        contact_records=[]
        for ci in range(d.ncon):
            co=d.contact[ci];g1,g2=int(co.geom1),int(co.geom2)
            record=dict(index=ci,geom1=g1,geom2=g2,pos=co.pos.copy(),dist=float(co.dist),friction=co.friction.copy(),frame=co.frame.copy(),dim=int(co.dim),included=False)
            if floor in (g1,g2):
                g=g2 if g1==floor else g1
                if g in feet and min_gaps[g]<=.001:
                    sets['b'].append(add(g,co.pos.copy(),co.pos[2]-floorz,'b_actual_contact',min_gaps[g],dict(contact_index=ci,contact_dist=float(co.dist),actual_friction=co.friction.tolist())));record['included']=True
            contact_records.append(record)
        for key in 'abc':sets[key]=sorted(set(sets[key]))
        sets['d']=sorted(set(sets['a']+sets['b']+sets['c']))
        dump(O/f'POINTS_{pi}.json',dict(points=points,sets=sets,meshes=meshes,actual_contacts=contact_records))
        np.savez_compressed(O/f'VERTICES_{pi}.npz',world=np.asarray(vertices),geom=vgeom,gap=vgap)
        np.savez_compressed(O/f'POSE_{pi}.npz',qpos=q,qvel=d.qvel,A=A,moment=moment,f0=f0,f1=f1,baseline=base,bias=bias,joint_passive=jp,G=G,W=W,floor_z=floorz,efc_J=efc,efc_type=d.efc_type,efc_id=d.efc_id,efc_force=d.efc_force,efc_pos=d.efc_pos,limit_rows=limrows,limit_J=L,qfrc_constraint=d.qfrc_constraint,probe=probe,probe_qfrc_actuator=d.qfrc_actuator)
        modes=[(s,'physiological') for s in 'abcd']+[('d','signed')]+([('d','limits')] if len(limrows) else [])
        posemeta.append(dict(pose=pi,source=str(source),reconstruction_error=error,points={s:len(sets[s]) for s in 'abcd'},active_limit_rows=limrows,active_limit_joint_ids=d.efc_id[limrows],limit_duplicate_skipped=not len(limrows)))
        for name,mode in modes:
            chosen=[points[i] for i in sets[name]];n=len(chosen);J=np.asarray([p['J'] for p in chosen]).reshape(n,3,48);mu=np.asarray([p['mu'] for p in chosen]);extra=L.T if mode=='limits' else np.zeros((48,0));B=np.column_stack([.9999*A@G,J.transpose(2,0,1).reshape(48,3*n),extra])/W;y=(bias-jp-base)/W;nv=B.shape[1]
            for fric in ('diamond','square'):
                C=np.vstack([np.c_[B,-np.eye(48)],np.c_[-B,-np.eye(48)]]);rhs=np.r_[y,-y];F=np.zeros((4*n,nv+48))
                for k,mm in enumerate(mu):
                    signs=[(1,1),(1,-1),(-1,1),(-1,-1)] if fric=='diamond' else [(1,0),(-1,0),(0,1),(0,-1)]
                    for t,(sx,sy) in enumerate(signs):F[4*k+t,66+3*k:69+3*k]=[sx,sy,-mm]
                C=np.vstack([C,F]);rhs=np.r_[rhs,np.zeros(4*n)];lo=np.r_[np.full(66,-np.inf if mode=='signed' else 0.),np.tile([-np.inf,-np.inf,0.],n),np.zeros(extra.shape[1]+48)];hi=np.r_[np.full(66,np.inf if mode=='signed' else 1.),np.full(nv-66+48,np.inf)];obj=np.r_[np.zeros(nv),np.ones(48)]
                sol=linprog(obj,A_ub=C,b_ub=rhs,bounds=list(zip(lo,hi)),method='highs');counts['LP']+=1;assert sol.success,sol.message
                f=sol.x[66:66+3*n].reshape(n,3);res=B@sol.x[:nv]-y;den=mu*f[:,2];tangent=np.linalg.norm(f[:,:2],axis=1);ratio=np.divide(tangent,den,out=np.zeros_like(den),where=den>1e-12);undefined=(den<=1e-12)&(tangent>1e-9)
                ar=dict(A=A,G=G,W=W,B=B,y=y,C=C,rhs=rhs,objective=obj,lower=lo,upper=hi,solution=sol.x,residual=res,activation=.0001+.9999*G@sol.x[:66],baseline=base,bias=bias,joint_passive=jp,J=J,mu=mu,points=np.asarray([p['pos'] for p in chosen]),point_ids=sets[name],gaps=np.asarray([p['gap'] for p in chosen]),bodies=np.asarray([p['body'] for p in chosen]),contact_force=f,circular_ratio=ratio,circular_ratio_undefined_violation=undefined,nforce=n,limit_J=L if mode=='limits' else np.zeros((0,48)),limit_reaction=sol.x[66+3*n:nv],generalized_contact_force=J.transpose(2,0,1).reshape(48,3*n)@f.ravel(),generalized_limit_force=extra@sol.x[66+3*n:nv])
                for part in ('ineqlin','eqlin','lower','upper'):
                    for field in ('marginals','residual'):ar[part+'_'+field]=getattr(getattr(sol,part),field)
                cert=certificate(ar);assert max(cert[k] for k in ('gap','primal_violation','stationarity','dual_sign_violation','complementarity'))<1e-7
                file=f'LP_{pi}_{name}_{fric}_{mode}.npz';np.savez_compressed(O/file,**ar)
                row=dict(pose=pi,set=name,mode=mode,friction=fric,file=file,l1=float(sol.fun),inf=float(abs(res).max()),root_inf=float(abs(res[:6]).max()),joint_inf=float(abs(res[6:]).max()),gate=bool(abs(res).max()<.001),circular_verified=bool(not undefined.any() and np.all(tangent<=den+1e-8)),max_circular_ratio=float(ratio.max(initial=0)),sumFz_over_W=float(f[:,2].sum()/W),certificate=cert,highs_status=sol.status,highs_message=sol.message,nit=sol.nit)
                results.append(row);print(json.dumps(row),flush=True)
    after=modelhash();assert before==after and all(sha(Path(p))==h for p,h in hashes.items())
    dump(O/'RESULT.json',dict(counts=counts,poses=posemeta,solutions=results,source_sha256=hashes,model_array_identity={k:dict(before=h,after=after[k]) for k,h in before.items()},physiological_conservative_candidates=[r for r in results if r['mode']!='signed' and r['friction']=='diamond' and r['gate']],native_units='NOT_ESTABLISHED',interpretation='Fixed-pose rigid-contact LP relaxation only. Signed muscle solutions nonphysiological; square may violate circular friction. No realized soft-contact or standing certification.'))
if __name__=='__main__':
    if '--verify' in sys.argv:verify()
    else:main()
