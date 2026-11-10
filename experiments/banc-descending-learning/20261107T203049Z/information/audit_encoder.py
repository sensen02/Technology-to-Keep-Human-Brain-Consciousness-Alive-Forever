"""Exact frozen-encoder audit + saved-data extraction; zero physical/neural steps.

Diagnostic only. Independent re-implementation of the four declared channel
formulas, analytic Jacobian vs central finite differences, engineered alias
fixtures, and verification of the actual encoder on every saved bin.

Scope note: a zero input column is an uncalibrated engineering interface gap in
this proxy encoder. It is NOT evidence that the source sensory biology lacks
that information, and no cell tuning direction is inferred anywhere here.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
OLD=ROOT/'experiments/banc-local-learning/20261107T185910Z'
TRIALS=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/learning/training')
FOUNDATION=ROOT/'experiments/banc-physical-learning/20261107T195435Z/foundation'
NEWSOURCE=ROOT/'experiments/banc-local-learning/20261107T192931Z/dynamics/SOURCE_TRACE.npz'
OLDSOURCE=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz')
sys.path.insert(0,str(OLD/'encoder'))
from encoder import SensoryEncoder
CHANNELS=('claw_position','hook_direction','hairplate_angle','campaniform_strain')
WEIGHT=24.446354755878005
EPS=1e-6


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def formula(physical,enc):
    """Independent explicit leg/name formula; never calls encoder helper features.

    claw_position   = 0.01+0.04*(clip(q_knee,lo,hi)-lo)/(hi-lo)
    hook_direction  = 0.05*tanh(abs(v_knee)/10)
    hairplate_angle = 0.01+0.04*mean_{axis in yaw,pitch,roll}((clip(q,lo,hi)-lo)/(hi-lo))
    campaniform     = 0.05*tanh(max(load_leg,0)/(W/6))
    """
    q,velocity,force=physical[:42],physical[42:84],physical[84:]
    bounds=enc.joint_ranges_rad;out=np.zeros((6,4))
    for legindex,leg in enumerate(enc.foot_order):
        knee=enc.joint_names.index('joint_'+leg+'Tibia_pitch')
        lo,hi=bounds[knee]
        out[legindex,0]=.01+.04*((min(max(q[knee],lo),hi)-lo)/(hi-lo))
        out[legindex,1]=.05*np.tanh(abs(velocity[knee])/10.)
        angle=[]
        for axis in ('yaw','pitch','roll'):
            j=enc.joint_names.index('joint_'+leg+'Coxa_'+axis);lo,hi=bounds[j]
            angle.append((min(max(q[j],lo),hi)-lo)/(hi-lo))
        out[legindex,2]=.01+.04*sum(angle)/3.
        out[legindex,3]=.05*np.tanh(max(force[legindex],0.)/(WEIGHT/6.))
    return out


def analytic_jacobian(enc,reference):
    """24x90 channel Jacobian at the reference input; rows leg-major then channel."""
    names=list(enc.joint_names);bounds=enc.joint_ranges_rad;q,v,load=reference[:42],reference[42:84],reference[84:]
    jac=np.zeros((6*4,90))
    for fi,leg in enumerate(enc.foot_order):
        knee=names.index('joint_'+leg+'Tibia_pitch')
        row=4*fi
        jac[row+0,knee]=.04/(bounds[knee,1]-bounds[knee,0])
        jac[row+1,42+knee]=.005*(1-np.tanh(abs(v[knee])/10.)**2)
        for axis in ('yaw','pitch','roll'):
            j=names.index('joint_'+leg+'Coxa_'+axis);jac[row+2,j]=.04/3./(bounds[j,1]-bounds[j,0])
        if load[fi]>0:jac[row+3,84+fi]=.05/(WEIGHT/6.)*(1-np.tanh(max(load[fi],0)/(WEIGHT/6.))**2)
    return jac


def main():
    if (HERE/'ENCODER_AUDIT.json').exists():raise FileExistsError('Preserve prior diagnostic: '+str(HERE/'ENCODER_AUDIT.json'))
    enc=SensoryEncoder();base=load(FOUNDATION/'BASE.npz')
    provenance={str(p):sha(p) for p in (HERE/'PREREG.json',Path(__file__),OLD/'encoder/encoder.py',OLD/'encoder/CONTRACT.json',FOUNDATION/'BASE.npz')}
    dump(HERE/'COARSE_FEATURE_CONTRACT.json',dict(feature_contract=enc.feature_contract,source_contract=enc.source_contract,qualified_records=enc.qualified_records,exclusions=enc.exclusions,qualified_count=enc.qualified_count,mn_root_ids=[str(int(x)) for x in base['mn_root_ids']],mn_leg=base['mn_leg'].tolist(),interpretation='Annotation identity is source evidence; formulas are deliberately uncalibrated proxies. No new cells or tuning directions assigned.'))

    # ---- reference input and analytic/analytic-numeric Jacobian ----
    names=list(enc.joint_names);bounds=enc.joint_ranges_rad;mid=bounds.mean(axis=1)
    reference=np.r_[mid,np.full(42,2.),np.full(6,WEIGHT/6)]
    local_to_compact={int(li):i for i,li in enumerate(enc.local_indices)}
    channel_jac=analytic_jacobian(enc,reference);cell_jac=np.zeros((556,90))
    for fi,leg in enumerate(enc.foot_order):
        for ci,ch in enumerate(CHANNELS):
            rows=[local_to_compact[int(li)] for li in enc.channel_local_indices[leg][ch]]
            cell_jac[rows]=channel_jac[4*fi+ci]
    numeric=np.empty_like(channel_jac)
    for col in range(90):
        plus=reference.copy();minus=reference.copy();plus[col]+=EPS;minus[col]-=EPS
        fa=formula(plus,enc);fb=formula(minus,enc)
        numeric[:,col]=(fa-fb).reshape(-1)/(2*EPS)
    maxerror=float(np.max(np.abs(numeric-channel_jac)))
    cell_numeric=np.empty_like(cell_jac)
    for col in range(90):
        plus=reference.copy();minus=reference.copy();plus[col]+=EPS;minus[col]-=EPS
        ca=enc.encode(plus[:42],plus[42:84],plus[84:])
        cb=enc.encode(minus[:42],minus[42:84],minus[84:])
        cell_numeric[:,col]=(ca[enc.local_indices]-cb[enc.local_indices])/(2*EPS)
    cell_maxerror=float(np.max(np.abs(cell_numeric-cell_jac)))
    assert maxerror<1e-9 and cell_maxerror<1e-9
    clipped_outputs=float(np.max(reference_jac_output_guard(enc,reference)))

    # ---- columns and influence port records ----
    columns=[]
    for col in range(90):
        kind='q' if col<42 else 'qvel' if col<84 else 'foot_load'
        name=names[col] if col<42 else names[col-42] if col<84 else enc.foot_order[col-84]
        influenced=np.flatnonzero(cell_jac[:,col]!=0)
        rows=np.flatnonzero(channel_jac[:,col]!=0)
        columns.append(dict(column=col,kind=kind,name=name,units='rad' if kind=='q' else 'rad/s' if kind=='qvel' else 'nativeforce',
                            zero=bool(len(rows)==0),channel_rows=rows.tolist(),
                            influenced=(None if len(rows)==0 else [dict(leg=enc.foot_order[r//4],channel=CHANNELS[r%4]) for r in rows]),
                            local_current_indices=([] if len(influenced)==0 else [int(x) for x in enc.local_indices[influenced]]),
                            qualified_cell_count=int(len(influenced))))
    zero_columns=[c['column'] for c in columns if c['zero']];nonzero_columns=[c['column'] for c in columns if not c['zero']]

    def rank(x):
        if x.size==0 or not np.any(x):return 0
        s=np.linalg.svd(x,compute_uv=False)
        return int(np.sum(s>s[0]*max(x.shape)*np.finfo(float).eps*10))
    ranks=dict(channel_map_24x90=rank(channel_jac),qualified_cell_map_556x90=rank(cell_jac),
               q_block_24x42=rank(channel_jac[:,:42]),qvel_block_24x42=rank(channel_jac[:,42:84]),
               foot_load_block_24x6=rank(channel_jac[:,84:]),
               nonzero_q_columns=sum(1 for c in columns[:42] if not c['zero']),
               nonzero_qvel_columns=sum(1 for c in columns[42:84] if not c['zero']),
               nonzero_foot_load_columns=sum(1 for c in columns[84:] if not c['zero']),
               zero_q_columns=sum(1 for c in columns[:42] if c['zero']),
               zero_qvel_columns=sum(1 for c in columns[42:84] if c['zero']),
               zero_foot_load_columns=sum(1 for c in columns[84:] if c['zero']))
    # Aliasing inside the q block: each leg's three coxa columns feed ONE hairplate output row,
    # so they are 3 distinct columns of rank 1 (12 independent q directions from 24 nonzero columns).
    coxa_per_leg_rank=[rank(channel_jac[:,list(enc.coxa_indices[i])]) for i in range(6)]
    coxa_rank_one=all(x==1 for x in coxa_per_leg_rank)
    knee_per_leg_rank=[rank(channel_jac[:,int(k):int(k)+1]) for k in enc.knee_indices]

    np.savez_compressed(HERE/'SENSITIVITY.npz',channel_analytic=channel_jac,channel_finite_difference=numeric,
                        cell_analytic=cell_jac,cell_finite_difference=cell_numeric,reference_input=reference,
                        qualified_local_indices=enc.local_indices,epsilon=EPS,
                        channel_rows=np.array([f'{leg}:{ch}' for leg in enc.foot_order for ch in CHANNELS]),
                        column_names=np.array([f'{c["kind"]}:{c["name"]}' for c in columns]))

    # ---- engineered alias fixtures (not real body trajectories) ----
    fixtures=[];fixture_arrays={}
    for leg in enc.foot_order:
        knee=names.index('joint_'+leg+'Tibia_pitch');kn=bounds[knee]
        coxa=[names.index('joint_'+leg+'Coxa_'+a) for a in ('yaw','pitch','roll')]
        troch=names.index('joint_'+leg+'Trochanter_yaw');tr=bounds[troch]
        rotter=names.index('joint_'+leg+'Trochanter_roll')
        # coxa yaw/pitch swapped around their midpoints: identical 3-axis mean, distinct q
        a=reference.copy();b=reference.copy()
        dy=.25*(bounds[coxa[0],1]-bounds[coxa[0],0]);dp=.25*(bounds[coxa[1],1]-bounds[coxa[1],0])
        a[coxa[0]]=mid[coxa[0]]+dy;a[coxa[1]]=mid[coxa[1]]-dp
        b[coxa[0]]=mid[coxa[0]]-dy;b[coxa[1]]=mid[coxa[1]]+dp
        # trochanter angle change: no declared input port
        a2=reference.copy();b2=reference.copy()
        b2[troch]=tr[0]+.1*(tr[1]-tr[0]);b2[rotter]=bounds[rotter,0]+.1*(bounds[rotter,1]-bounds[rotter,0])
        # opposite knee speed: exact abs() alias
        a3=reference.copy();b3=reference.copy();a3[42+knee]=2.;b3[42+knee]=-2.
        for mode,pa,pb in (('coxa_yaw_pitch_swap_midpoint',a,b),('trochanter_angle_no_port',a2,b2),('opposite_knee_speed_abs_alias',a3,b3)):
            ia=enc.encode(pa[:42],pa[42:84],pa[84:]);ib=enc.encode(pb[:42],pb[42:84],pb[84:])
            assert np.array_equal(ia,ib),(leg,mode,float(np.max(np.abs(ia-ib))))
            assert np.array_equal(formula(pa,enc),formula(pb,enc)),(leg,mode)
            assert np.all(pa[:42]>=bounds[:,0]) and np.all(pa[:42]<=bounds[:,1]) and np.all(pb[:42]>=bounds[:,0]) and np.all(pb[:42]<=bounds[:,1])
            changed_inputs=[int(x) for x in np.flatnonzero(pa!=pb)]
            assert changed_inputs,(leg,mode)
            key=leg+'_'+mode;fixture_arrays[key+'_a']=pa;fixture_arrays[key+'_b']=pb
            fixtures.append(dict(leg=leg,type=mode,exact_equal_encoder_currents=True,exact_equal_independent_formula=True,
                                 changed_input_columns=changed_inputs,changed_input_names=[f'{"q" if c<42 else "qvel" if c<84 else "foot_load"}:{names[c] if c<42 else names[c-42] if c<84 else enc.foot_order[c-84]}' for c in changed_inputs],
                                 joint_bounds_valid=True,distinct_q=bool(np.any(pa[:42]!=pb[:42])),distinct_qvel=bool(np.any(pa[42:84]!=pb[42:84])),
                                 physical_trajectory=False,collision_validity_not_tested=True))
    # non-preregistered sensitivity control: a real claw-direction change must be visible
    controls=[]
    for leg in enc.foot_order:
        knee=names.index('joint_'+leg+'Tibia_pitch');span=bounds[knee,1]-bounds[knee,0]
        pa=reference.copy();pb=reference.copy();pb[knee]=mid[knee]+.1*span
        ia=enc.encode(pa[:42],pa[42:84],pa[84:]);ib=enc.encode(pb[:42],pb[42:84],pb[84:])
        delta=float(np.max(np.abs((ia-ib)[enc.local_indices])))
        assert delta>0
        controls.append(dict(leg=leg,type='knee_angle_plus_10pct_span_visibility_control',preregistered=False,
                             max_abs_current_change_nA=delta,encoder_distinguishes=True))
    np.savez_compressed(HERE/'ALIAS_FIXTURES.npz',**fixture_arrays)

    # ---- frozen encoder verified on every saved bin ----
    p_all=[];mn_all=[];f_all=[];episode_ids=[];generations=[];time_all=[];current_error=0.;formula_error=0.;reads=0
    trial_rows=[]
    for gen in range(4):
        for candidate in range(12):
            trial=1+13*gen+candidate;folder=TRIALS/f'trial_{trial:03d}'
            paths=[folder/name for name in ('POLICY.npz','BODY.npz','NATIVE.npz','INITIAL.npz','ACTOR.npz','RESULT.json')]
            for path in paths:provenance[str(path)]=sha(path)
            pol,body,native,initial,actor=[load(path) for path in paths[:-1]];result=json.loads(paths[-1].read_text());reads+=6
            assert result['valid'] and np.any(actor['theta']),trial
            for key in ('mn_root_ids','mn_local','mn_leg','joint_names'):assert np.array_equal(actor[key],base[key]),trial
            physical=pol['local_physical_input'];indices=np.arange(200)*5
            assert np.array_equal(physical[:,:42],body['qpos'][indices][:,initial['qa']]),trial
            assert np.array_equal(physical[:,42:84],body['qvel'][indices][:,initial['va']]),trial
            assert np.array_equal(physical[1:,84:],native['rows'][indices[1:]*80-1,1:7]),trial
            features=[]
            for i,row in enumerate(physical):
                actual=enc.encode(row[:42],row[42:84],row[84:])
                current_error=max(current_error,float(np.max(np.abs(actual[enc.local_indices]-pol['sensory_current_nA'][i]))))
                f=formula(row,enc)
                observed=np.array([[actual[enc.channel_local_indices[leg][ch][0]] for ch in CHANNELS] for leg in enc.foot_order])
                formula_error=max(formula_error,float(np.max(np.abs(observed-f))))
                # every qualified cell of a leg/channel carries the identical declared current
                for li,leg in enumerate(enc.foot_order):
                    for ci,ch in enumerate(CHANNELS):
                        port=enc.channel_local_indices[leg][ch]
                        assert np.all(actual[port]==observed[li,ci]),(trial,leg,ch)
                assert np.array_equal(actual[enc.local_indices],pol['sensory_current_nA'][i]),(trial,i)
                features.append(observed)
            p_all.append(physical);mn_all.append(pol['mn_rates_hz']);f_all.append(features);episode_ids.extend([trial]*200);generations.extend([gen]*200);time_all.extend(pol['time_s'].tolist())
            trial_rows.append(dict(trial=trial,generation=gen,valid=True,feasible=bool(result['feasible']),reward=result['reward'],
                                   support_gate_passed=bool((result.get('physical_gate') or {}).get('passed')),
                                   theta_max_abs=float(np.abs(actor['theta']).max()),
                                   mn_rates_hz_min=float(pol['mn_rates_hz'].min()),mn_rates_hz_max=float(pol['mn_rates_hz'].max())))
    old=load(OLDSOURCE);new=load(NEWSOURCE);provenance[str(OLDSOURCE)]=sha(OLDSOURCE);provenance[str(NEWSOURCE)]=sha(NEWSOURCE);reads+=2
    assert np.array_equal(old['time_s'],new['time_s'])
    oldfeatures=[]
    for i,row in enumerate(old['physical_local_input']):
        actual=enc.encode(row[:42],row[42:84],row[84:]);current_error=max(current_error,float(np.max(np.abs(actual[enc.local_indices]-old['sensory_current_nA'][i]))))
        observed=np.array([[actual[enc.channel_local_indices[leg][ch][0]] for ch in CHANNELS] for leg in enc.foot_order]);formula_error=max(formula_error,float(np.max(np.abs(observed-formula(row,enc)))))
        oldfeatures.append(observed)
    assert current_error==0 and formula_error<1e-16
    np.savez_compressed(HERE/'SAVED_DATA.npz',physical=np.concatenate(p_all),mn_rates=np.concatenate(mn_all),encoded=np.concatenate(f_all),
                        episode_id=episode_ids,generation=generations,time_s=time_all,foot_order=enc.foot_order,joint_names=enc.joint_names,
                        mn_leg=base['mn_leg'],mn_root_ids=base['mn_root_ids'],mn_local=base['mn_local'],
                        oldphysical=old['physical_local_input'],oldmn=new['mn_rates_hz'],oldencoded=np.asarray(oldfeatures),oldtime=old['time_s'],
                        old_sensory_current_nA=old['sensory_current_nA'],old_sensory_local_indices=old['sensory_local_indices'])
    summary=dict(status='PASS_EXACT_INTERFACE_AUDIT',source_hashes=provenance,
                 encoder_calls=90*3+18+6+9600+len(old['time_s']),data_artifact_reads=reads,
                 physics_steps=0,neural_steps=0,training_updates=0,
    reference_input=dict(q='all42jointmidpoints_rad',qvel=2.0,foot_load_native=WEIGHT/6.,note='interior, no clip active at reference'),
                 max_abs_channel_value_before_output_clip_nA=clipped_outputs,output_clip_active_at_reference=False,
                 columns=columns,zero_columns=zero_columns,nonzero_columns=nonzero_columns,ranks=ranks,
                 coxa_axis_rank_per_leg=coxa_per_leg_rank,coxa_three_axes_collapse_to_one_rank_per_leg=coxa_rank_one,
                 knee_angle_rank_per_leg=knee_per_leg_rank,
                 q_independent_directions_vs_nonzero_q_columns=f'{ranks["q_block_24x42"]} independent of {ranks["nonzero_q_columns"]} nonzero q columns',
                 jacobian_max_error_vs_central_difference=maxerror,cell_jacobian_max_error_vs_central_difference=cell_maxerror,
                 finite_difference_epsilon=EPS,alias_fixtures=fixtures,sensitivity_controls=controls,
                 saved_trials=trial_rows,actual_trial_bins=int(sum(len(x) for x in p_all)),actual_trial_episodes=len(p_all),
                 actual_trial_generations=[0,1,2,3],excluded_trials=[0,13,26,39,52],
                 old_failed_source_bins=int(len(old['time_s'])),old_failed_source_time_s=[float(old['time_s'][0]),float(old['time_s'][-1])],
                 recorded_current_max_error_nA=current_error,independent_formula_max_error_nA=formula_error,
                 interface_gap_statement='A zero input column is an uncalibrated interface gap in this proxy encoder, not evidence that source sensory biology lacks that information. No cell tuning direction is inferred.',
                 limits='Local Jacobian at an interior, positive-speed, positive-load reference. Clamped angles at/outside joint bounds have zero derivative beyond the boundary; abs() has a nondifferentiable kink at zero speed; max(load,0) has a kink at zero load. Joint-bounded fixtures are synthetic inputs, not real body trajectories or collision-valid poses. Channels are deliberate uncalibrated proxies, not measured fly physiology.')
    dump(HERE/'ENCODER_AUDIT.json',summary)
    report={k:summary[k] for k in ('status','ranks','zero_columns','coxa_three_axes_collapse_to_one_rank_per_leg','jacobian_max_error_vs_central_difference','recorded_current_max_error_nA','independent_formula_max_error_nA','actual_trial_bins','old_failed_source_bins','data_artifact_reads','encoder_calls')}
    print(json.dumps(report),flush=True)


def reference_jac_output_guard(enc,reference):
    """Max pre-clip declared channel value at the reference (clip must be inactive)."""
    qn=np.clip(reference[:42],enc.joint_ranges_rad[:,0],enc.joint_ranges_rad[:,1])
    qn=(qn-enc.joint_ranges_rad[:,0])/(enc.joint_ranges_rad[:,1]-enc.joint_ranges_rad[:,0])
    v=reference[42:84];load=reference[84:]
    values=[.01+.04*qn[enc.knee_indices],.05*np.tanh(np.abs(v[enc.knee_indices])/10.),
            .01+.04*qn[enc.coxa_indices].mean(axis=1),.05*np.tanh(np.maximum(load,0)/(WEIGHT/6.))]
    return [float(np.max(x)) for x in values]


if __name__=='__main__':main()
