"""Parent-only actual array load and frozen oldmotor cache, never physics."""
from sources import ROOT,read,verify


def prepare(m):
    import numpy as np
    import torch
    binding=read(ROOT/'INPUT_BINDINGS.json');verify(binding['source_hashes']);verify(binding['data_hashes'])
    X=[];U=[];Y=[];F=[];M=[];states=[];raw=[];proofs=[]
    with torch.no_grad():
        for e in binding['entries']:
            with np.load(e['path'],allow_pickle=False) as archive:
                a={key:archive[key].copy() for key in e['headers']}
            start=e['start'];warm_start=e['warm_start'];x=torch.tensor(a['X238'][start:],dtype=torch.float64)
            if tuple(x.shape)!=(1920,238) or not np.isfinite(a['X238']).all():raise ValueError('Full actual1920 suffix finite')
            if e['teacher']:
                if not np.array_equal(a['candidate_phase'],np.r_[np.zeros(960,dtype=bool),np.ones(1920,dtype=bool)]):raise ValueError('Exact teacherwarm/candidatephase')
                u=torch.tensor([[0.,e['turn']]]*1920,dtype=torch.float64);y=torch.tensor(a['Y42'][start:],dtype=torch.float64)
                if not np.allclose(a['time'],1.+np.arange(2880)*.005,atol=1e-8,rtol=0):raise ValueError('Teacher clocks')
            else:
                u=torch.tensor(np.column_stack([np.ones(1920),a['raw_turn'][start:]]),dtype=torch.float64);y=torch.tensor(a['actually_applied_action'][start:],dtype=torch.float64)
                if not np.array_equal(a['actually_applied_action'][start:],a['policy_action'][start:]):raise ValueError('ActualNNappliedactions required')
                if not np.array_equal(a['command'][:,1],a['raw_turn']) or not np.all(a['command'][:start]==0):raise ValueError('Actualstored neuralcommand/warm0')
                if not np.allclose(a['time'],np.arange(3080)*.005,atol=1e-8,rtol=0):raise ValueError('NNclocks')
            warm=torch.tensor(a['X238'][warm_start:start],dtype=torch.float64)
            old=m.old.initial_state(1);zero=x.new_zeros((1,2));prime_exact=True
            for k,row in enumerate(warm):
                idx=warm_start+k
                if not e['teacher'] and not torch.equal(torch.cat(old,-1)[0],torch.tensor(a['hidden_before'][idx])):raise ValueError('ActualNNwarm hiddenbefore differs')
                mean,old=m.old.step(row[None],zero,old)
                if not e['teacher']:
                    if not torch.equal(torch.cat(old,-1)[0],torch.tensor(a['hidden_after'][idx])) or not torch.equal(mean[0],torch.tensor(a['mean'][idx])) or not torch.equal(mean[0].tanh(),torch.tensor(a['policy_action'][idx])):raise ValueError('ActualNNwarm oldproduction parity')
            primed=tuple(v.clone() for v in old)
            # Independent producer-loop vs public warm API: no invented teacherhidden.
            via_api=m.initial_state(1)
            for row in warm:
                _,via_api=m.prime_old(row[None],zero,via_api)
            if not all(torch.equal(a,b) for a,b in zip(primed,via_api[:3])) or bool(via_api[3].abs().max()):raise ValueError('Dualpriming/publicoldstate exact/new96zero')
            features=[];means=[]
            for k,(row,task) in enumerate(zip(x,u)):
                if not e['teacher'] and not torch.equal(torch.cat(old,-1)[0],torch.tensor(a['hidden_before'][start+k])):raise ValueError('ActualNNsuffix hiddenbefore differs')
                turn=task[1:2].reshape(1,1);oldcmd=torch.cat((turn.relu()/3.,turn),-1)
                mean,old=m.old.step(row[None],oldcmd,old)
                if not e['teacher']:
                    if not torch.equal(torch.cat(old,-1)[0],torch.tensor(a['hidden_after'][start+k])) or not torch.equal(mean[0],torch.tensor(a['mean'][start+k])) or not torch.equal(mean[0].tanh(),y[k]):raise ValueError('ActualNNsuffix oldmean/hidden/action parity')
                features.append(torch.cat((row[None,m.old.base.physical_mask.bool()],*old,task[None]),-1)[0]);means.append(mean[0])
            if not bool(torch.isfinite(y).all()) or bool((y.abs()>1.+1e-12).any()):raise ValueError('Actualbounded42 targets required')
            X.append(x);U.append(u);Y.append(y);F.append(torch.stack(features));M.append(torch.stack(means));states.append(primed);raw.append(dict(entry=e,warm=warm,x=x,u=u,y=y))
            proofs.append(dict(id=e['id'],teacher=e['teacher'],warm_rows=960,start=start,suffix_rows=1920,NN_archived_hidden_parity=not e['teacher'],teacher_prime_proof='dual implementation exact, not archivedhidden',suffix_direct_task_no_shift=True))
    if len(X)!=8:raise ValueError('Exactly8actualtrajectories')
    result=dict(x=torch.stack(X,1),u=torch.stack(U,1),y=torch.stack(Y,1),features=torch.stack(F,1),oldmean=torch.stack(M,1),initial_old=tuple(torch.cat([s[i] for s in states],0) for i in range(3)),v1=torch.tensor([0. if e['teacher'] else 1. for e in binding['entries']],dtype=torch.float64),raw=raw,proofs=proofs,binding=binding)
    if result['features'].shape!=(1920,8,370):raise ValueError('physical112/oldafter256/command2features')
    verify(binding['source_hashes']);verify(binding['data_hashes']);return result


def production_sequence(m,raw):
    import torch
    with torch.no_grad():
        state=m.initial_state(1)
        for row in raw['warm']:
            _,state=m.prime_old(row[None],row.new_zeros((1,2)),state)
        means=[];hidden=[]
        for x,u in zip(raw['x'],raw['u']):
            mean,state=m.step(x[None],u[None],state);means.append(mean[0]);hidden.append(torch.cat(state,-1)[0])
        return torch.stack(means),torch.stack(hidden)
