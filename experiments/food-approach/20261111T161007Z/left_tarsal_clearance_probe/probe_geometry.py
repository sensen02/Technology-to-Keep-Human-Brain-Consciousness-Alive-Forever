"""Deferred static-only geometry engine. Imported only after explicit runtime flag.

All poses live in private MjData. No integration/force-query API is used here.
"""


class StaticGeometry:
    def __init__(self,core,initial_saved_qpos):
        import numpy as np
        import mujoco as mj
        self.np,self.mj,self.c,self.m=np,mj,core,core.plant
        self.calls=dict(kinematics=0,comPos=0,jac=0,collision=0,geomDistance=0,integration=0)
        self.qa=core.a_qpos.copy();self.va=core.a_dof.copy()
        names=[self.m.joint(int(j)).name for j in core.actuator.joint_ids]
        self.rh=np.array([i for i,n in enumerate(names) if n.startswith('joint_RH')],dtype=int)
        expected=['joint_RHCoxa_yaw','joint_RHCoxa_pitch','joint_RHCoxa_roll','joint_RHTrochanter_yaw','joint_RHTrochanter_pitch','joint_RHTrochanter_roll','joint_RHTibia_pitch']
        if [names[i] for i in self.rh]!=expected:raise RuntimeError('Unexpected RH actuator mapping')
        self.names=[names[i] for i in self.rh];self.rqa=self.qa[self.rh];self.rva=self.va[self.rh]
        self.bounds=self.m.jnt_range[core.actuator.joint_ids[self.rh]].copy()
        self.geom=int(self.m.geom('RHTarsus4_geom').id);self.floor=int(self.m.geom('floor').id)
        self.body=int(self.m.geom_bodyid[self.geom]);self.pads=np.asarray(core.pads['RH'],dtype=int)
        if self.geom!=72 or self.body!=61 or len(self.pads)!=3 or int(core.geom_leg[self.geom])!=-1:
            raise RuntimeError('Executed geometry/body/pad classification mismatch')
        if int(self.m.geom_type[self.floor])!=int(mj.mjtGeom.mjGEOM_PLANE):raise RuntimeError('Floor is not plane')
        root=int(self.m.body('RHCoxa').id);desc={root}
        for body in range(self.m.nbody):
            if int(self.m.body_parentid[body]) in desc:desc.add(body)
        self.other=[g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) in desc and g!=self.geom and int(core.geom_leg[g])<0 and (int(self.m.geom_contype[g]) or int(self.m.geom_conaffinity[g]))]
        self.d=mj.MjData(self.m);self.d.qpos[:]=initial_saved_qpos;self.fk()
        self.R0=self.d.xmat[core.th].reshape(3,3).copy();self.o0=self.d.xpos[core.th].copy()
        self.plane_n=self.d.geom_xmat[self.floor].reshape(3,3)[:,2].copy();self.plane_p=self.d.geom_xpos[self.floor].copy()
        self.mesh_cache={}
        for g in [self.geom]+self.other:
            if int(self.m.geom_type[g])!=int(mj.mjtGeom.mjGEOM_MESH):raise RuntimeError('Unexpected RH collider shape')
            mid=int(self.m.geom_dataid[g]);adr=int(self.m.mesh_vertadr[mid]);n=int(self.m.mesh_vertnum[mid])
            self.mesh_cache[g]=self.m.mesh_vert[adr:adr+n].astype(float).copy()
        self.base_q=None

    def fk(self):
        self.mj.mj_kinematics(self.m,self.d);self.calls['kinematics']+=1
        self.mj.mj_comPos(self.m,self.d);self.calls['comPos']+=1

    def jac(self,point,body):
        j=self.np.zeros((3,self.m.nv));self.mj.mj_jac(self.m,self.d,j,None,point,int(body));self.calls['jac']+=1
        return j[:,self.rva]

    def vertices(self,g):
        return self.mesh_cache[g]@self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g]

    def heights(self):
        out={}
        for g in [self.geom]+self.other:
            h=(self.vertices(g)-self.plane_p)@self.plane_n
            out[self.m.geom(g).name]=float(h.min())
        return out

    def pose(self,q):
        self.base_q=self.np.asarray(q,dtype=float).copy()
        self.d.qpos[:]=self.base_q;self.d.qvel[:]=0.;self.fk()
        self.Ra=self.d.xmat[self.c.th].reshape(3,3).copy();self.oa=self.d.xpos[self.c.th].copy()
        self.B=self.R0@self.Ra.T
        return self.evaluate(self.base_q[self.rqa],grad=True,collision=True,others=True)

    def evaluate(self,x,grad=False,collision=False,others=False,attached_vertex=None):
        np=self.np
        self.d.qpos[:]=self.base_q;self.d.qpos[self.rqa]=x;self.d.qvel[:]=0.;self.fk()
        if not np.array_equal(self.d.qpos[:7],self.base_q[:7]):raise RuntimeError('Private saved root changed')
        mask=np.ones(self.m.nq,dtype=bool);mask[self.rqa]=False
        if not np.array_equal(self.d.qpos[mask],self.base_q[mask]):raise RuntimeError('Non-RH angles changed')
        centers=self.d.geom_xpos[self.pads].copy();radii=self.m.geom_size[self.pads,0]
        in_initial=(centers-self.oa)@self.B.T+self.o0
        bottoms=in_initial[:,2]-radii;low=int(np.argmin(bottoms))
        xy=(self.Ra.T@(centers.mean(0)-self.oa))[:2]
        pad=np.r_[xy,bottoms[low]]
        vertices=self.vertices(self.geom);height=(vertices-self.plane_p)@self.plane_n;vi=int(np.argmin(height))
        result=dict(h=float(height[vi]),vertex=vi,point=vertices[vi].copy(),pad=pad,
            pad_low=low,pad_bottoms_actual_world=centers[:,2]-radii,pad_bottoms_initial_frame=bottoms,
            pad_active_gap=float(np.sort(bottoms)[1]-np.sort(bottoms)[0]),
            mesh_active_gap=float(np.sort(height)[1]-np.sort(height)[0]))
        if attached_vertex is not None:result['attached_h']=float(height[int(attached_vertex)])
        if grad:
            jj=[self.jac(self.d.geom_xpos[g],self.m.geom_bodyid[g]) for g in self.pads]
            result['Jpad']=np.vstack([(self.Ra.T@np.mean(jj,axis=0))[:2],(self.B@jj[low])[2]])
            result['g']=self.plane_n@self.jac(vertices[vi],self.body)
        if others:result['RH_nonfoot_minimum_signed_plane_heights']=self.heights()
        if collision:
            fromto=np.zeros(6)
            distance=self.mj.mj_geomDistance(self.m,self.d,self.geom,self.floor,1000.,fromto);self.calls['geomDistance']+=1
            self.mj.mj_collision(self.m,self.d);self.calls['collision']+=1
            contacts=[]
            for i in range(self.d.ncon):
                ct=self.d.contact[i]
                if {int(ct.geom1),int(ct.geom2)}=={self.geom,self.floor}:
                    contacts.append(dict(distance=float(ct.dist),includemargin=float(ct.includemargin),position=ct.pos.copy().tolist(),exclude=int(ct.exclude)))
            result['collision']=dict(geomDistance=float(distance),fromto=fromto.tolist(),support_distance_error=float(abs(distance-result['h'])),contacts=contacts,
                note='Geometric collision only; no constraint solve or force prediction')
        return result

    def nullspace(self,J):
        np=self.np;u,s,vt=np.linalg.svd(J,full_matrices=True)
        tol=max(1e-12,float(s[0])*1e-9);rank=int(np.sum(s>tol))
        return vt[rank:].T,s,rank,tol

    def fd(self,x,baseline):
        np=self.np;eps=1e-6;J=np.empty((3,7));g=np.empty(7);gmin=np.empty(7)
        for j in range(7):
            plus=x.copy();minus=x.copy();plus[j]+=eps;minus[j]-=eps
            a=self.evaluate(plus,attached_vertex=baseline['vertex']);b=self.evaluate(minus,attached_vertex=baseline['vertex'])
            J[:,j]=(a['pad']-b['pad'])/(2*eps);g[j]=(a['attached_h']-b['attached_h'])/(2*eps);gmin[j]=(a['h']-b['h'])/(2*eps)
        return dict(epsilon_rad=eps,attachedpoint_gradient_fd=g.tolist(),attachedpoint_jacobian_max_abs_error=float(np.max(np.abs(g-baseline['g']))),
            min_support_gradient_fd=gmin.tolist(),min_support_gradient_max_abs_error=float(np.max(np.abs(gmin-baseline['g']))),
            pad_jacobian_fd=J.tolist(),pad_jacobian_max_abs_error=float(np.max(np.abs(J-baseline['Jpad']))),
            pad_nondifferentiability_possible=bool(baseline['pad_active_gap']<1e-8),mesh_nondifferentiability_possible=bool(baseline['mesh_active_gap']<1e-8))

    def bounded_secondary(self,x0,baseline,safe_height,config,primary_target=None,cumulative_anchor=None):
        np=self.np
        from scipy.optimize import minimize,LinearConstraint
        trust=config['per_decision_trust_rad'];tol=config['pad_error_norm_max'];improve=config['clearance_improvement_min']
        lo=np.maximum(self.bounds[:,0],x0-trust);hi=np.minimum(self.bounds[:,1],x0+trust)
        target=baseline['pad'].copy() if primary_target is None else np.asarray(primary_target).copy()
        if cumulative_anchor is not None:
            lo=np.maximum(lo,cumulative_anchor-config['cumulative_trust_rad']);hi=np.minimum(hi,cumulative_anchor+config['cumulative_trust_rad'])
        if np.any(x0<self.bounds[:,0]-1e-12) or np.any(x0>self.bounds[:,1]+1e-12):
            return dict(status='BASELINE_OUTSIDE_ORIGINAL_JOINT_BOUNDS_NO_SILENT_CLIP',local_improvement=False,safe_margin_reached=False,attempts=[],q_initial=x0.tolist(),q_final=x0.tolist())
        x=x0.copy();current=baseline;attempts=[]
        for iteration in range(config['local_updates_max']):
            if current['h']>=safe_height:break
            Z,s,rank,svtol=self.nullspace(current['Jpad']);gZ=current['g']@Z
            if float(np.linalg.norm(gZ))<1e-10:
                attempts.append(dict(iteration=iteration,status='CLEARANCE_GRADIENT_NULLSPACE_DEGENERATE',gZ_norm=float(np.linalg.norm(gZ))));break
            # Variables are dimensionless; all7 joint/trust bounds remain explicit.
            A=trust*Z;linear=LinearConstraint(A,lo-x,hi-x)
            fit=minimize(lambda z:float(-gZ@z+5e-9*(z@z)),np.zeros(Z.shape[1]),jac=lambda z:-gZ+1e-8*z,
                constraints=[linear],method='SLSQP',options=dict(maxiter=40,ftol=1e-12,disp=False))
            entry=dict(iteration=iteration,optimizer_success=bool(fit.success),optimizer_message=str(fit.message),gZ_norm=float(np.linalg.norm(gZ)),backtracks=[])
            delta=A@fit.x;accepted=False
            for back in range(config['backtracks_max']):
                trial=x+delta*(.5**back);reason=None
                if np.any(trial<lo-1e-12) or np.any(trial>hi+1e-12):reason='JOINT_OR_TRUST_BOUNDS'
                candidate=None
                if reason is None:
                    candidate=self.evaluate(trial,grad=True)
                    for correction in range(config['primary_reprojection_iterations_max']):
                        error=candidate['pad']-target
                        if np.linalg.norm(error)<=tol*.25:break
                        d=-np.linalg.pinv(candidate['Jpad'],rcond=1e-9)@error
                        corrected=trial+d
                        if np.any(corrected<lo-1e-12) or np.any(corrected>hi+1e-12):reason='PRIMARY_REPROJECTION_BOUND';break
                        trial=corrected;candidate=self.evaluate(trial,grad=True)
                    paderr=float(np.linalg.norm(candidate['pad']-target))
                    if reason is None and paderr>tol:reason='NONLINEAR_PAD_RESIDUAL'
                    if reason is None and candidate['h']<=current['h']+1e-10:reason='NO_CLEARANCE_ASCENT'
                    if reason is None:
                        hs=self.heights()
                        for name,hnew in hs.items():
                            if name==self.m.geom(self.geom).name:continue
                            hold=baseline['RH_nonfoot_minimum_signed_plane_heights'][name]
                            if hnew<min(hold,0.)-1e-8:reason='OTHER_RH_NEW_OR_WORSE_FLOOR_PENETRATION';break
                bt=dict(backtrack=back,accepted=reason is None,reason=reason,
                    q_trial=trial.tolist(),pad_error_norm=None if candidate is None else float(np.linalg.norm(candidate['pad']-target)),
                    h=None if candidate is None else candidate['h'])
                entry['backtracks'].append(bt)
                if reason is None:x=trial;current=candidate;accepted=True;break
            attempts.append(entry)
            if not accepted:break
        final=self.evaluate(x,grad=True,collision=True,others=True)
        Z,s,rank,svtol=self.nullspace(final['Jpad'])
        err=float(np.linalg.norm(final['pad']-target));gain=final['h']-baseline['h']
        return dict(status='STATIC_LOCAL_RESULT',local_improvement=bool(gain>=improve and err<=tol),safe_margin_reached=bool(final['h']>=safe_height and err<=tol),
            q_initial=x0.tolist(),q_final=x.tolist(),delta_q=(x-x0).tolist(),max_abs_delta_rad=float(np.max(np.abs(x-x0))),
            pad_error_norm=err,clearance_gain=float(gain),safe_height=float(safe_height),initial_height=baseline['h'],final_height=final['h'],
            root_unchanged=True,other_angles_unchanged=True,original_joint_bounds_respected=bool(np.all(x>=self.bounds[:,0]-1e-12) and np.all(x<=self.bounds[:,1]+1e-12)),
            minimum_joint_bound_margin_rad=float(np.minimum(x-self.bounds[:,0],self.bounds[:,1]-x).min()),
            hypothetical_delta_per5ms_rad_s=((x-x0)/config['decision_s']).tolist(),actual_teacher_rate_feasibility='NOT_ESTABLISHED: these are saved snapshots, not q_des chain',
            pad_bottoms_actual_world_before=baseline['pad_bottoms_actual_world'].tolist(),pad_bottoms_actual_world_after=final['pad_bottoms_actual_world'].tolist(),
            other_RH_heights_before=baseline['RH_nonfoot_minimum_signed_plane_heights'],other_RH_heights_after=final['RH_nonfoot_minimum_signed_plane_heights'],
            fixed_primary_target=target.tolist(),final_collision=final['collision'],attempts=attempts)

    def kinematic_sequence(self,x0,baseline,safe_height,config):
        np=self.np;x=x0.copy();steps=[];current=baseline
        for step in range(config['kinematic_sequence_steps_max']):
            if current['h']>=safe_height:break
            outcome=self.bounded_secondary(x,current,safe_height,config,primary_target=baseline['pad'],cumulative_anchor=x0)
            steps.append(outcome)
            if not outcome.get('local_improvement',False):break
            x=np.asarray(outcome['q_final']);current=self.evaluate(x,grad=True,collision=True,others=True)
        final=self.evaluate(x,grad=True,collision=True,others=True)
        delta=x-x0;paderr=float(np.linalg.norm(final['pad']-baseline['pad']))
        return dict(status='BOUNDED_PRIVATE_KINEMATIC_SEQUENCE',not_physical_integration=True,steps=steps,
            step_count=len(steps),accepted_step_count=sum(bool(s.get('local_improvement')) for s in steps),
            safe_margin_reached=bool(final['h']>=safe_height and paderr<=config['pad_error_norm_max']),
            local_improvement=bool(final['h']-baseline['h']>=config['clearance_improvement_min'] and paderr<=config['pad_error_norm_max']),
            initial_height=baseline['h'],final_height=final['h'],safe_height=float(safe_height),clearance_gain=float(final['h']-baseline['h']),
            fixed_primary_target=baseline['pad'].tolist(),pad_error_norm=paderr,q_initial=x0.tolist(),q_final=x.tolist(),cumulative_delta_q=delta.tolist(),
            cumulative_max_abs_rad=float(np.max(np.abs(delta))),rate2rad_s_time_lower_bound_s=float(np.max(np.abs(delta))/2.),
            nominal_number_of5ms_kinematic_steps=len(steps),dynamic_tracking_feasibility='NOT_ESTABLISHED; neither force prediction nor physical execution',
            other_RH_heights_before=baseline['RH_nonfoot_minimum_signed_plane_heights'],other_RH_heights_after=final['RH_nonfoot_minimum_signed_plane_heights'],
            final_collision=final['collision'],root_unchanged=True,other_angles_unchanged=True)
