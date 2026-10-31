"""Layered solve-state Newton, Euler damping update, and discrete COM remainder."""
from pathlib import Path
import json,sys
import numpy as np,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_solver_step_observation import O,BASE,c,sha,dump

def main():
 packed=np.load(O/'SOLVER_STEPS.npz');s={k:packed[k] for k in packed.files};packed.close();z=np.load(O/'BODY.npz');contacts=json.loads((O/'SOLVER_CONTACTS.json').read_text());m=c.cp.load_model();mass=m.body_mass.sum();up=-m.opt.gravity/np.linalg.norm(m.opt.gravity);h=m.opt.timestep=.0000125;weight=mass*np.linalg.norm(m.opt.gravity);n=len(s['qacc']);legforce=np.zeros((n,7,3));force=np.zeros((n,3));newton=[];balance=[];euler=[];explicit=[]
 assert not m.dof_dampingpoly.any() and np.all(m.jnt_actuatorid==-1),'nonlinear/joint actuator damping needs exact derivative'
 for i in range(n):
  for co in contacts[i]:
   a,b=co['geom1'],co['geom2'];pa=m.geom_type[a]==mujoco.mjtGeom.mjGEOM_PLANE;pb=m.geom_type[b]==mujoco.mjtGeom.mjGEOM_PLANE
   if not(pa or pb):continue
   g=b if pa else a;gn=(m.geom(g).name or '').lower();bn=(m.body(int(m.geom_bodyid[g])).name or '').lower();leg=next((j for j,l in enumerate(['lf','lm','lh','rf','rm','rh']) if (gn.startswith(l) or bn.startswith(l)) and 'tars' in gn+bn),6);legforce[i,leg]+=np.array(co['frame']).reshape(3,3).T@np.array(co['local_force'])[:3]*(1 if pa else -1)
  force[i]=legforce[i].sum(0)+mass*m.opt.gravity+s['xfrc_applied'][i,:,:3].sum(0)
  newton.append(mass*s['solve_com_acc'][i]-force[i])
  # qM packed inertia was computed at solve position; no forward or state restore.
  M=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,M,s['qM'][i])
  rhs=s['qfrc_actuator'][i]+s['qfrc_passive'][i]-s['qfrc_bias'][i]+s['qfrc_constraint'][i]+s['qfrc_applied'][i]
  balance.append(M@s['qacc'][i]-rhs)
  dv=s['qvel'][i]-s['pre_qvel'][i];explicit.append(dv-h*s['qacc'][i]);euler.append((M+h*np.diag(m.dof_damping))@(dv/h)-rhs)
 newton=np.array(newton);balance=np.array(balance);euler=np.array(euler);explicit=np.array(explicit);t=z['time'];momentum=mass*((z['com_velocity']-z['com_velocity'][0])@up);net=force@up;impulse=np.r_[0,np.cumsum(net*h)];remainder=momentum-impulse;solveacc=s['solve_com_acc']@up;foot=legforce[:,:6].sum(1)@up;nonfoot=legforce[:,6]@up
 np.savez_compressed(O/'LAYERED_BALANCES.npz',time_start=s['time_start'],time_end=s['time_return'],perleg_world_force=legforce,net_vertical_force=net,solve_com_vertical_acc=solveacc,newton_world_force_residual=newton,generalized_dynamics_residual=balance,explicit_dv_minus_hqacc=explicit,euler_implicit_damping_equation_residual=euler,body_time=t,actual_momentum_change=momentum,actual_solver_force_impulse=impulse,discrete_COM_momentum_remainder=remainder)
 summary=dict(exit_code=0,states=n,mass=float(mass),weight=float(weight),max_newton_world_residual=float(abs(newton).max()),max_generalized_solve_residual=float(abs(balance).max()),max_explicit_dv_hqacc_residual=float(abs(explicit).max()),max_Euler_damping_equation_residual=float(abs(euler).max()),max_discrete_COM_momentum_remainder=float(abs(remainder).max()),first10ms=dict(gravity_impulse=float(-weight*800*h),foot_impulse=float(foot[:800].sum()*h),nonfoot_impulse=float(nonfoot[:800].sum()*h),net_impulse=float(impulse[800]),actual_delta_momentum=float(momentum[800]),kinematic_remainder=float(remainder[800])),interpretation='Euler implicit joint damping acceleration is stack-local, not d.qacc; verify (M+hD)*dv/h=rhs separately; solve Newton uses unmodified solve qacc. Residual COM change includes damping integrator and nonlinear velocity mapping, not pure endpoint quadrature.',model_load=1,mj_forward=0,mj_step=0,neural_step=0)
 dump(O/'LAYERED_SUMMARY.json',summary)
 fig,axs=plt.subplots(4,1,figsize=(16,13));tm=s['time_start']*1000;axs[0].plot(tm,foot/weight,label='solve foot vertical /weight');axs[0].plot(tm,net/weight,label='solve net vertical /weight');axs[0].legend();axs[1].plot(tm,solveacc,label='solve full COM acceleration');axs[1].plot(tm,net/mass,'--',label='solve external force /mass');axs[1].legend();axs[2].plot(t*1000,momentum,label='actual mass*delta COM velocity');axs[2].plot(t*1000,impulse,'--',label='actual solver force impulse');axs[2].legend();axs[3].plot(t*1000,remainder,label='discrete COM remainder');axs[3].legend()
 for ax in axs:ax.grid(alpha=.2);ax.set_xlabel('physical ms')
 fig.suptitle('Exact historical40ms reproduction / solve-stage force vs new integration state separated');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(O/'SOLVER_MOMENTUM_COMPARISON.png',dpi=130);plt.close(fig);Path('/home/sensen/Desktop/S2_SOLVER_MOMENTUM.png').write_bytes((O/'SOLVER_MOMENTUM_COMPARISON.png').read_bytes());(O/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps(summary))
if __name__=='__main__':main()
