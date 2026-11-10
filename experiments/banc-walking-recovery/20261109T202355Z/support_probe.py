"""Declared whole-body reference controller for diagnosis/teacher only; NOT neural walking.
No qpos/qvel writes after reset, all actuation through unchanged rate/torque path.
"""
from pathlib import Path
import sys,json
import numpy as np
import mujoco as mj
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import frame_audit as F
rc,T=F.rc,F.T

def run(height_gain,velocity_gain,rate_scale):
    c=rc.Core();c.reset();leg=rc.LEG_ORDER.index('RH')
    pads=lambda l:np.mean([c.d.geom_xpos[g] for g in c.pads[rc.LEG_ORDER[l]]],axis=0)
    anchors=[pads(i).copy() for i in range(rc.NL)]
    start=c.measure()['foot'][leg].copy();height0=c.measure()['com']
    target_height=1.30
    state='settle'; phase=0; held=0; rows=[]; swing_x0=None
    st=dict(swing=False,start_x=0.,max_lift=0.,best_adv=0.)
    for k in range(900):
        m=c.measure();pz=m['padz'][leg]
        if k==150:state='raise'
        if state=='raise' and pz>=.035:
            state='swing';phase=0;swing_x0=m['foot'][leg][0]
        if state=='swing' and (m['foot'][leg][0]-swing_x0>=.06 or phase>=160):state='place'
        if state=='place' and pz<=T.THRESH['place_padz'] and m['loads'][leg]>=T.SWING_END:
            state='hold';anchors[leg]=pads(leg).copy()
        a=np.zeros(rc.NJ)
        for l in range(rc.NL):
            rel,J=F.jacobians(c,l)
            ids=T.leg_joint_action_index(c,l);ds=c.a_dof[ids]
            if l==leg and state in ('raise','swing','place'):
                # mixed coordinates: thorax-relative fore-aft/lateral, world vertical.
                A=np.vstack([rel[0,ds],rel[1,ds],J[2,ds]])
                xgoal=start[0]+(.09 if state in ('swing','place') else 0.)
                zgoal=.045 if state in ('raise','swing') else -.008
                desired=np.array([velocity_gain*(xgoal-m['foot'][l][0]),velocity_gain*(start[1]-m['foot'][l][1]),velocity_gain*(zgoal-m['padz'][l])])
                full=np.vstack([rel[0],rel[1],J[2]])
            else:
                A=J[:,ds];full=J
                desired=velocity_gain*(anchors[l]-pads(l))
                # downward stance extension to recover height: declared teacher feedback only.
                desired[2]-=height_gain*max(0.,target_height-m['com'])
            root_v=c.d.qvel.copy();root_v[c.a_dof]=0
            rhs=desired-full@root_v
            dq=A.T@np.linalg.solve(A@A.T+.02*np.eye(3),rhs)
            a[ids]=np.clip(dq/(rc.MAX_RATE*rate_scale),-1,1)*rate_scale
        out=c.tick(np.clip(a,-1,1));load=out['loads'][leg];x=out['foot'][leg][0];z=out['padz'][leg]
        if load<=T.SWING_START and not st['swing']:st.update(swing=True,start_x=x,max_lift=0.,best_adv=0.)
        if st['swing']:
            st['max_lift']=max(st['max_lift'],float(z));st['best_adv']=max(st['best_adv'],float(x-st['start_x']))
            if load>=T.SWING_END and z<=T.THRESH['place_padz']:st['swing']=False
        rows.append(dict(k=k,state=state,support=float(out['loads'].sum()/rc.WEIGHT),com=float(out['com']),nonfoot=float(out['nonfoot']),padz=float(z),load=float(load/rc.WEIGHT),x=float(x),root_x=float(c.d.qpos[0])))
        phase+=1
    # Standard single-leg numbers plus a longer declared pre-settle/hold protocol; not same task.
    summary=dict(height_gain=height_gain,velocity_gain=velocity_gain,rate_scale=rate_scale,state=state,max_lift=st['max_lift'],advance=st['best_adv'],support_min=min(r['support'] for r in rows),com_min=min(r['com'] for r in rows),nonfoot_max=max(r['nonfoot'] for r in rows),final_load=rows[-1]['load'],final_padz=rows[-1]['padz'])
    summary['support_safe']=bool(summary['support_min']>=.9 and summary['com_min']>=1 and summary['nonfoot_max']<=T.THRESH['nonfoot'])
    return summary,rows

def main():
    results=[]
    for h,v,s in [(0,10,1),(3,10,1),(10,10,1),(3,30,1),(10,30,1),(3,10,.5)]:
        summary,rows=run(h,v,s);results.append(dict(summary=summary,trace=rows));print(json.dumps(summary),flush=True)
        (HERE/'SUPPORT_PROBE.json').write_text(json.dumps(dict(label='all-leg active compensation diagnostic; 900 ticks with 150 tick settle; NOT walking acceptance',runs=results),indent=2)+'\n')
if __name__=='__main__':main()
