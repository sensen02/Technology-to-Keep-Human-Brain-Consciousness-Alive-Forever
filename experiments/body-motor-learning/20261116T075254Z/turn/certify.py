"""Seal native qualified arc labels and independently replay frozen actions."""
import argparse
from pathlib import Path
import numpy as np

from audit_pair import bound
from rollout import HERE,UNITS,read,stan,run


def seal(path,config):
    r=bound(path)
    if r['kind']!='teacher' or not r['passed'] or not all(r['checks'].values()):
        raise ValueError('The entire native directional teacher must pass before admission')
    assert r['native_integrations']==r['actual_ms']*80
    assert r['command_units']==UNITS
    ident=read(path/'RUN_IDENTITY.json')
    assert ident['config']['plan']==read(config)
    with np.load(path/'DECISIONS.npz',allow_pickle=False) as file:
        assert file['x'].shape==(r['actual_ms']//5,213)
        assert file['command'].shape==(r['actual_ms']//5,2)
        assert file['action'].shape==(r['actual_ms']//5,42)
        assert np.max(np.abs(file['action']))<=1.
        np.testing.assert_array_equal(file['ms'],np.arange(r['actual_ms']//5)*5)
    with np.load(path/'NATIVE.npz',allow_pickle=False) as z:
        assert len(z['forces'])==r['native_integrations']
        assert not z['pose'][:,7].any()
    n=dict(native_sha256=r['native_sha256'],native_integrations=r['native_integrations'],zero_extra_force=True,
           native_force_cross_checks=r['native_force_cross_checks'],native_joint_pass=r['passed'],
           legacy_exit=r['legacy_exit_window'],steady_straight=r['straight_window'],full_exit=r['full_exit_window'],
           terminal=r['original_arc']['tail'],full_D1_D8=False)
    native=path/'NATIVE_AUDIT.json'
    stan.put(native,n)
    cert=dict(passed=True,role='teacher',scope='whole arc + physically verified native exit, contiguous finite body',
              direction=r['direction'],run=str(path.resolve()),config_path=str(config.resolve()),
              config_sha256=stan.sha(config),result_sha256=stan.sha(path/'RESULT.json'),trace_sha256=r['trace_sha256'],
              decisions_sha256=r['decisions_sha256'],decision_sha256=r['decisions_sha256'],
              native_sha256=r['native_sha256'],native_audit_sha256=stan.sha(native),
              native_integrations=r['native_integrations'],identity_sha256=stan.sha(path/'RUN_IDENTITY.json'),
              x_dims=213,action_dims=42,command_units=UNITS,first_active_phase_ms=r['bounds'][0],
              checks=r['checks'],full_D1_D8=False)
    stan.put(path/'TEACHER_ACCEPTANCE.json',cert)
    print(cert,flush=True)


class FrozenDecision:
    def __init__(self, source,entry):
        with np.load(source,allow_pickle=False) as f:
            self.obs,self.command,self.action=(f[k].copy() for k in ('x','command','action'))
        self.i=entry//5

    def __call__(self,x,command):
        i=self.i
        np.testing.assert_array_equal(x,self.obs[i],err_msg=f'Saved physical x at row {i}')
        np.testing.assert_array_equal(command,self.command[i],err_msg=f'Saved command at row {i}')
        self.i+=1
        return self.action[i].copy()


def replay(src,dest,config):
    a=bound(src)
    c=read(src/'TEACHER_ACCEPTANCE.json')
    assert c['config_sha256']==stan.sha(config)
    assert c['decision_sha256']==a['decisions_sha256']
    assert c['passed']
    obs=FrozenDecision(src/'DECISIONS.npz',a['bounds'][0])
    b=run(dest,a['direction'],config,label='independent_frozen_teacher_decision_reintegration',
          subject=obs,subject_path=Path(__file__).resolve())
    assert obs.i==len(obs.action)
    matched={}
    for file in ('TRACE.npz','DECISIONS.npz','NATIVE.npz'):
        matched[file]=stan.sha(src/file)==stan.sha(dest/file)
    assert all(matched.values()),matched
    assert b['native_integrations']==a['native_integrations'] and a['passed']==b['passed']
    stan.put(dest/'REPLAY_AUDIT.json',dict(source=str(src.resolve()),replay=str(dest.resolve()),passed=True,
        native_integrations=b['native_integrations'],same_physics_sealed_directions=True,
        all_fields_bitwise=matched,processed_decisions=obs.i,full_D1_D8=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('config',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.out is None:
        seal(a.source.resolve(),a.config.resolve())
    else:
        replay(a.source.resolve(),a.out.resolve(),a.config.resolve())
