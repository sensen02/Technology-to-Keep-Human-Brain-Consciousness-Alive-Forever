"""Parent numeric helpers for independent fullproduction targetfeedback check."""
from pathlib import Path
import ast,hashlib,importlib.util,sys
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/locomotion-enhancement/20261113T161918Z/independent_motor_calibration')
g={'__name__':'_targetfeedback_exact_comparators','hashlib':hashlib};tree=ast.parse((OLD/'preflight.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('max_error','tensor_digest')],type_ignores=[]),str(OLD/'preflight.py')+'::comparators','exec'),g)
max_error=g['max_error'];tensor_digest=g['tensor_digest']

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def torch_integrate(actions,initial_target,bounds,max_rate):
    import torch
    target=initial_target;rows=[]
    for action in actions:
        rate=torch.clamp(action*2.,-2.,2.);rate=torch.minimum(torch.maximum(rate,-max_rate),max_rate)
        for _ in range(5):target=torch.minimum(torch.maximum(target+.001*rate,bounds[:,0]),bounds[:,1]);rows.append(target)
    return torch.stack(rows)


def production_feedback(m,raw,data,index,*,forced=False):
    """No physicalsteps: actualfixedX withtargetfields replaced, fullold.step path.
    Return sixscanoutputs andold256states; onlyparentcalls underno_gradvalidation.
    """
    import torch
    state=m.initial_state(1)
    for x in raw['warm']:_,state=m.prime_old(x[None],x.new_zeros((1,2)),state)
    target=data['initial_target'][index:index+1].clone();rows=[[],[],[],[],[],[]];old=[]
    for k,(x,u) in enumerate(zip(raw['x'],raw['u'])):
        if forced:target=data['actual_pre_targets'][k,index:index+1]
        before=target.clone();xx=x[None].clone();qactual=data['actual_q'][k,index:index+1]
        xx[:,42:84]=((target-qactual)/data['halfspan']).clamp(-10.,10.).float().double()
        xx[:,126:168]=((target-data['q0'])/data['halfspan']).clamp(-10.,10.).float().double()
        mean,state,res=m.step_details(xx,u[None],state);action=mean.tanh();rate=(action*2.).clamp(-2.,2.);rate=torch.minimum(torch.maximum(rate,-data['max_rate']),data['max_rate'])
        for _ in range(5):target=torch.minimum(torch.maximum(target+.001*rate,data['bounds'][:,0]),data['bounds'][:,1])
        for row,value in zip(rows,(mean,res,state[3],before,target,action)):row.append(value[0])
        old.append(torch.cat(state[:3],-1)[0])
    return tuple(torch.stack(row) for row in rows),torch.stack(old)


def output_errors(a,b):
    return [float((x-y).abs().max()) for x,y in zip(a,b)]
