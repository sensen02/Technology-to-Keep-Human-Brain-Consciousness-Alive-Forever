"""Parent oneCore0integration state-restore/teacherquery benchmark prerequisite."""
import argparse
from contextlib import ExitStack
import time
from metadata import ROOT,OLD,read,save,verify,qualify
from common import cpu1,stack,load_arrays,snapshot,snapshot_equal,make_teacher,jacobian_probe,query_one,resources


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-parent-query-preflight',action='store_true');a=p.parse_args()
    if not a.execute_parent_query_preflight:p.error('Parentofflinequerypreflightflag required')
    if (ROOT/'PREFLIGHT.json').exists():raise FileExistsError(ROOT/'PREFLIGHT.json')
    start=time.monotonic();cpu1();meta=qualify();verify(meta['source_hashes'])
    import numpy as np
    import restore
    physics,identity,mj=stack();entry=next(e for e in meta['teachers'] if e['condition']=='left_pivot');folder=entry['folder'];trace=load_arrays(entry['trace']);samples=load_arrays(entry['samples']);initial=read(folder+'/INITIAL_STATE.json');source_ident=read(folder+'/IDENTITY.json');hand=read(folder+'/HANDOVER.json');checks={}
    with restore.install_step_traps(mj) as counters, ExitStack() as guards:
        c=physics.Core(adhesion=False);core_trap=guards.enter_context(restore.block_core_ticks(c));restore.trap_neural_calls_if_loaded(counters)
        identity.verify_runtime(source_ident,c)
        info=restore.restore(c,trace,initial,5800,mujoco_module=mj);handover=restore.handover_check(c,hand)
        checks['actual_restore_exact']=bool(info['exact_state']);checks['handover7statehashes_exact']=bool(handover['all_exact'])
        teacher=make_teacher(c,'left_pivot');before=snapshot(c);jac=jacobian_probe(c,teacher);checks['privateFKactualgeometryXYJac']=max(jac)<=1e-4;checks['privateFK_no_restoredstatewrite']=snapshot_equal(before,snapshot(c))
        timer=time.monotonic();diffs=[];restores=[]
        for j in range(20):
            i=960+j;ms=int(samples['physical_ms'][i]);elapsed=float(samples['elapsed_teacher_s'][i]);row=query_one(c,teacher,trace,initial,ms,elapsed,restore,mj)
            diffs.append(float(np.max(np.abs(row['action']-samples['Y42'][i]))));restores.append(row['restore']['exact_state'])
        benchmark=time.monotonic()-timer;checks['first20qualified_teacher_action_reproduction']=max(diffs)<=1e-6;checks['all20preactionrestores_exact']=all(restores)
        # Same offlineCore, after benchmark only: validate floorforce orientation
        # on an actually saved student row with nonfoot contact, never byforceinjection.
        source_student=next(e for e in meta['students'] if e['id']=='v1_left_pivot')
        st=load_arrays(source_student['trace']);si=read(source_student['folder']+'/INITIAL_STATE.json');nf_index=int(np.argmax(st['nonfoot']))
        nf_restore=restore.restore(c,st,si,nf_index+1,mujoco_module=mj);measured=c.measure();signed=[];nonfoot_sum=0.
        for ci in range(c.d.ncon):
            contact=c.d.contact[ci];ga,gb=int(contact.geom1),int(contact.geom2)
            if not (c.planes[ga] or c.planes[gb]):continue
            geom=gb if c.planes[ga] else ga;force=np.zeros(6);mj.mj_contactForce(c.plant,c.d,ci,force)
            vertical=float((contact.frame.reshape(3,3).T@force[:3])[2])*(1. if c.planes[ga] else -1.)
            signed.append(vertical)
            if c.geom_leg[geom]<0:nonfoot_sum+=max(0.,vertical)
        force_probe=dict(source='saved v1left max1msnonfoot row after20benchmark',physical_ms=nf_index+1,saved_NF_1ms=float(st['nonfoot'][nf_index]),recomputed_NF=float(measured['nonfoot']),independent_signed_NF_sum=nonfoot_sum,min_floor_signed_vertical=None if not signed else min(signed),max_floor_signed_vertical=None if not signed else max(signed),recompute_vs_saved_notbitexact=True,restore=nf_restore)
        checks['nonfoot_force_frame_orientation']=bool(st['nonfoot'][nf_index]>0 and signed and min(signed)>=-1e-8 and abs(nonfoot_sum-float(measured['nonfoot']))<=1e-8)
        checks['zero_integration_traps']=counters['integration_calls']==0 and core_trap['core_tick_or_reset_attempts']==0
        checks['zero_NN_forward_or_weightload']=counters.get('NN_forward_calls',0)==0 and counters.get('torch_load_calls',0)==0
        identity.verify_runtime(source_ident,c)
    elapsed=time.monotonic()-start;estimate=benchmark/20.*meta['total_teacher_queries']*1.5+elapsed*2.;timing=estimate<=300;rss=resources(start);verify(meta['source_hashes'])
    save(ROOT/'PREFLIGHT.json',dict(all_passed=bool(all(checks.values()) and timing),all_correctness_checks_passed=bool(all(checks.values())),checks=checks,Core_constructions=1,physical_native_steps=counters['integration_calls'],NN_forwards=counters['NN_forward_calls'],benchmark20query_s=benchmark,total_query_rows=meta['total_teacher_queries'],predicted_main_wall_s_with_margin=estimate,timing_status='READY_QUERY_300S' if timing else 'BLOCKED_BEFORE_MAIN_QUERY_TIMING',peak_RSS_MiB=rss,jacobian_errors=jac,teacher_action_errors_first20=diffs,first_restore=info,handover=handover,nonfoot_force_orientation_probe=force_probe,traps=counters,source_hashes=meta['source_hashes'],status='LABEL_QUERY_ONLY_NOT_PHYSICAL_SUCCESS'))
    if not all(checks.values()):raise ValueError('Offlinequerycorrectnesspreflightfailed')
    if not timing:raise ValueError('Mainquerytimingblocked beforeexecution; noautomaticextension')
    print('OFFLINE_TEACHER_QUERY_PREFLIGHT_PASS_0INTEGRATION',flush=True)


if __name__=='__main__':main()
