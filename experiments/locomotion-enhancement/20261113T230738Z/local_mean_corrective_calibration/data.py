"""Parent-only actual12 cache. Invalid assisted rows keep recurrent history.
Imports numerical libraries only inside explicit parent-called functions.
"""
from sources import ROOT,read,verify


def prepare(m):
    import numpy as np
    import torch
    binding=read(ROOT/'INPUT_BINDINGS.json');verify(binding['source_hashes']);verify(binding['data_hashes'])
    if len(binding['entries'])!=12:raise ValueError('Exactly12 actual training sources')
    xs=[];us=[];ys=[];features=[];oldmeans=[];valids=[];primed_states=[];raw=[];proofs=[]
    with torch.no_grad():
        for entry in binding['entries']:
            with np.load(entry['path'],allow_pickle=False) as archive:
                arrays={key:archive[key].copy() for key in entry['headers']}
            collection=bool(entry.get('collection',False));teacher=bool(entry['teacher']);start=int(entry['start']);warm_start=int(entry['warm_start'])
            x=torch.tensor(arrays['X238'][start:],dtype=torch.float64);warm=torch.tensor(arrays['X238'][warm_start:start],dtype=torch.float64)
            if x.shape!=(1920,238) or warm.shape!=(960,238) or not bool(torch.isfinite(x).all() and torch.isfinite(warm).all()):raise ValueError('Fullactual960warm+1920suffix required')
            if collection:
                if not np.allclose(arrays['time'],np.arange(3080)*.005,atol=1e-8,rtol=0) or not np.array_equal(arrays['physical_step_ms'],np.arange(3080)*5):raise ValueError('Collection actual clocks')
                if not np.array_equal(arrays['command'][:200],np.zeros((200,2))) or not np.array_equal(arrays['command'][200:start],np.repeat([[1.,0.]],960,axis=0)):raise ValueError('Collection settle/warm task path')
                expected=np.repeat(np.asarray(entry['command'],dtype=float)[None],1920,axis=0)
                if not np.array_equal(arrays['command'][start:],expected):raise ValueError('Collection actual fixed command')
                u=torch.tensor(arrays['command'][start:],dtype=torch.float64);y=torch.tensor(arrays['supervision_target'][start:],dtype=torch.float64)
                if teacher:
                    if not bool(arrays['teacher_available'][start:].all()) or not np.array_equal(arrays['supervision_target'][start:],arrays['teacher_action'][start:]):raise ValueError('Assisted targets must be available unmixed teacher_action')
                    if not np.array_equal(arrays['actually_applied_action'][start:],.75*arrays['teacher_action'][start:]+.25*arrays['student_action'][start:]):raise ValueError('Actual assisted action provenance differs')
                    mask=np.asarray(arrays['label_qualified'][start:],dtype=bool)
                    if not np.array_equal(mask,np.asarray(arrays['label_mask'][start:],dtype=bool).all(1)):raise ValueError('Actual assisted mask conjunction differs')
                    if not np.all(arrays['control_source'][start:]==7):raise ValueError('Assisted actual source classification')
                else:
                    if bool(arrays['teacher_available'][start:].any()) or not np.all(arrays['control_source'][start:]==8):raise ValueError('Old control is unassisted directold execution')
                    mask=np.ones(1920,dtype=bool)
            elif teacher:
                if not np.array_equal(arrays['candidate_phase'],np.r_[np.zeros(960,dtype=bool),np.ones(1920,dtype=bool)]) or not np.allclose(arrays['time'],1.+np.arange(2880)*.005,atol=1e-8,rtol=0):raise ValueError('Original actual teacher full clocks')
                u=x.new_tensor([[0.,entry['turn']]]).repeat(1920,1);y=torch.tensor(arrays['Y42'][start:],dtype=torch.float64);mask=np.ones(1920,dtype=bool)
            else:
                if not np.allclose(arrays['time'],np.arange(3080)*.005,atol=1e-8,rtol=0) or not np.array_equal(arrays['actually_applied_action'][start:],arrays['policy_action'][start:]):raise ValueError('Original actual NN target/clock provenance')
                if not np.array_equal(arrays['command'][:,1],arrays['raw_turn']) or not np.all(arrays['command'][:start]==0):raise ValueError('Original NN signedturn/warm0 contract')
                u=torch.tensor(np.column_stack([np.ones(1920),arrays['raw_turn'][start:]]),dtype=torch.float64);y=torch.tensor(arrays['actually_applied_action'][start:],dtype=torch.float64);mask=np.ones(1920,dtype=bool)
            if mask.shape!=(1920,) or int(mask.sum())!=int(entry['expected_valid_count']) or not mask.any():raise ValueError('Exact nonempty qualified labels required, no allinvalid fallback')
            if y.shape!=(1920,42) or not bool(torch.isfinite(y).all()) or bool((y.abs()>1.+1e-12).any()):raise ValueError('Real bounded target42 remains fullhistory')
            old=m.old.initial_state(1);zero=x.new_zeros((1,2));recorded_old=collection or not teacher
            for k,row in enumerate(warm):
                idx=warm_start+k
                if recorded_old and not torch.equal(torch.cat(old,-1)[0],torch.tensor(arrays['hidden_before'][idx,:256])):raise ValueError('Archived old256 warm BEFORE mismatch')
                mean,old=m.old.step(row[None],zero,old)
                if recorded_old:
                    if not torch.equal(torch.cat(old,-1)[0],torch.tensor(arrays['hidden_after'][idx,:256])):raise ValueError('Archived old256 warm AFTER mismatch')
                    recorded=arrays['oldmean_direct'][idx] if collection else arrays['mean'][idx]
                    if not torch.equal(mean[0],torch.tensor(recorded)):raise ValueError('Archived directold warm mean mismatch')
                    if not collection and not torch.equal(mean[0].tanh(),torch.tensor(arrays['policy_action'][idx])):raise ValueError('Original NN warm action parity')
            primed=tuple(value.clone() for value in old);via=m.initial_state(1)
            for row in warm:_,via=m.prime_old(row[None],zero,via)
            if not all(torch.equal(a,b) for a,b in zip(primed,via[:3])) or bool(via[3].abs().max()):raise ValueError('Independent old warm/public prime parity or new96 notzero')
            f=[];means=[];compat_diff=0.
            for k,(row,task) in enumerate(zip(x,u)):
                index=start+k
                if recorded_old and not torch.equal(torch.cat(old,-1)[0],torch.tensor(arrays['hidden_before'][index,:256])):raise ValueError('Archived old256 suffix BEFORE mismatch')
                turn=task[1:2].reshape(1,1);legacy=torch.cat((turn.relu()/3.,turn),-1);mean,old=m.old.step(row[None],legacy,old)
                if recorded_old:
                    if not torch.equal(torch.cat(old,-1)[0],torch.tensor(arrays['hidden_after'][index,:256])):raise ValueError('Archived old256 suffix AFTER mismatch')
                    recorded=arrays['oldmean_direct'][index] if collection else arrays['mean'][index]
                    if not torch.equal(mean[0],torch.tensor(recorded)):raise ValueError('Archived directold suffix mean mismatch')
                if collection:
                    compat_diff=max(compat_diff,float(np.max(np.abs(mean[0].numpy()-arrays['oldmean_compat'][index]))))
                    if not teacher and (not torch.equal(mean[0].tanh(),y[k]) or not np.array_equal(arrays['supervision_target'][index],arrays['actually_applied_action'][index])):raise ValueError('Oldcontrol target is directold tanh, never shadow policy')
                elif not teacher and not torch.equal(mean[0].tanh(),y[k]):raise ValueError('Original actual NN suffix action parity')
                # DIRECT frozenoldmean is appended, never mean-residual compatibility.
                f.append(torch.cat((row[None,m.old.base.physical_mask.bool()],*old,task[None],mean),-1)[0]);means.append(mean[0])
            valid=torch.tensor(mask.copy(),dtype=torch.float64)
            xs.append(x);us.append(u);ys.append(y);features.append(torch.stack(f));oldmeans.append(torch.stack(means));valids.append(valid);primed_states.append(primed)
            raw.append(dict(entry=entry,warm=warm,x=x,u=u,y=y,valid=torch.tensor(mask.copy(),dtype=torch.bool),labels_available=True,training_input=True))
            proofs.append(dict(id=entry['id'],warm_start=warm_start,warm_rows=960,suffix_rows=1920,qualified_target_rows=int(mask.sum()),all_rows_preserved=True,invalid_loss_only_zero=True,archived_old256_and_directmean_parity=recorded_old,teacher_without_archived_NN_state=teacher and not collection,prime_dualpath_exact=True,features_oldmean_source='direct frozen m.old.step output',oldmean_compat_max_abs_diff_descriptive=compat_diff if collection else None,targets='unmixed teacher_action' if collection and teacher else 'directoldmean.tanh' if collection else 'original immutable actualtarget',neural_acceptance=False))
    v1=torch.tensor([float(bool(entry['v1_prior'])) for entry in binding['entries']],dtype=torch.float64)
    result=dict(x=torch.stack(xs,1),u=torch.stack(us,1),y=torch.stack(ys,1),features=torch.stack(features,1),oldmean=torch.stack(oldmeans,1),valid=torch.stack(valids,1),v1=v1,initial_old=tuple(torch.cat([state[i] for state in primed_states],0) for i in range(3)),raw=raw,proofs=proofs,binding=binding)
    if result['features'].shape!=(1920,12,412) or result['valid'].shape!=(1920,12) or int(v1.sum())!=6:raise ValueError('Exact12fullhistory412/6prior schema')
    verify(binding['source_hashes']);verify(binding['data_hashes']);return result


def production_sequence(m,raw):
    import torch
    with torch.no_grad():
        state=m.initial_state(1)
        for row in raw['warm']:_,state=m.prime_old(row[None],row.new_zeros((1,2)),state)
        means=[];states=[]
        for x,u in zip(raw['x'],raw['u']):
            mean,state=m.step(x[None],u[None],state);means.append(mean[0]);states.append(torch.cat(state,-1)[0])
        return torch.stack(means),torch.stack(states)


def dev_sources(binding):
    import numpy as np
    import torch
    rows=[]
    for entry in binding['dev_entries']:
        with np.load(entry['path'],allow_pickle=False) as archive:x=archive['X238'].copy();u=archive['command'].copy()
        n=entry['start']+entry['length']
        if x.shape!=(n,238) or u.shape!=(n,2) or not np.isfinite(x).all() or not np.isfinite(u).all():raise ValueError('Actual variable-length unlabeled DEV schema')
        if entry['training_input'] is not False or entry['labels_available'] is not False:raise ValueError('Failedstudent DEV has no valid teacherY')
        rows.append(dict(entry=entry,warm=torch.tensor(x[200:1160],dtype=torch.float64),x=torch.tensor(x[1160:],dtype=torch.float64),u=torch.tensor(u[1160:],dtype=torch.float64),labels_available=False,training_input=False))
    return rows
