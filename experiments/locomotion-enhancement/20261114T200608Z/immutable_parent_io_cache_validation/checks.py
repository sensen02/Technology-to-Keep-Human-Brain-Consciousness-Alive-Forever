"""Parent-only lazy numerical checks; import uses stdlib, never loads/forwards NN.
Deepcopy equality is NOT storage isolation: assert_isolated must also pass.
Fingerprints preserve container/key types, dtype/shape/device and raw byte content.
"""
import hashlib,inspect,json,os,platform,random,struct,sys,types
from pathlib import Path


def fingerprint(value):
    """Exact canonical nested content, including RNG ndarray; no pickle/repr fallback."""
    import torch,numpy as np
    def pack(tag,*parts):
        return tag+b''.join(struct.pack('>Q',len(p))+p for p in parts)
    active=set()
    def encode(x):
        if x is None:return b'N'
        if x is Ellipsis:return b'Zellipsis'
        if x is NotImplemented:return b'Znotimplemented'
        if type(x) is bool:return b'B1' if x else b'B0'
        if type(x) is int:return pack(b'I',str(x).encode())
        if type(x) is float:return pack(b'F',struct.pack('>d',x))
        if type(x) is complex:return pack(b'C',struct.pack('>dd',x.real,x.imag))
        if type(x) is str:return pack(b'S',x.encode('utf8'))
        if type(x) is bytes:return pack(b'Y',x)
        if type(x) is bytearray:return pack(b'J',bytes(x))
        if isinstance(x,(set,frozenset)):return pack(b'Q',type(x).__qualname__.encode(),*sorted(encode(v) for v in x))
        if isinstance(x,Path):return pack(b'P',type(x).__module__.encode(),type(x).__qualname__.encode(),str(x).encode())
        if torch.is_tensor(x):
            if x.layout!=torch.strided:raise TypeError('Only dense strided checkpoint tensors supported')
            raw=x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return pack(b'T',str(x.dtype).encode(),encode(tuple(x.shape)),str(x.device).encode(),raw)
        if isinstance(x,np.ndarray):
            if x.dtype.hasobject:raise TypeError('Object ndarray forbidden')
            return pack(b'A',encode(x.dtype.descr if x.dtype.fields else x.dtype.str),encode(tuple(x.shape)),np.ascontiguousarray(x).tobytes())
        if isinstance(x,np.generic):return pack(b'G',x.dtype.str.encode(),x.tobytes())
        if isinstance(x,(dict,list,tuple)):
            if id(x) in active:raise TypeError('Cyclic payload forbidden')
            active.add(id(x))
            try:
                typename=pack(b'K',type(x).__module__.encode(),type(x).__qualname__.encode())
                if isinstance(x,dict):
                    entries=sorted((encode(k),encode(v)) for k,v in x.items())
                    return pack(b'D',typename,*(pack(b'E',k,v) for k,v in entries))
                return pack(b'L' if isinstance(x,list) else b'U',typename,*(encode(v) for v in x))
            finally:active.remove(id(x))
        raise TypeError('Unsupported exact fingerprint type: '+type(x).__module__+'.'+type(x).__qualname__)
    return hashlib.sha256(encode(value)).hexdigest()


def model_payload_equal(a,b):
    """Both pairs are (model,payload); covers weights, identity, config/anchor/Adam/RNG/curve."""
    import torch
    ma,pa=a;mb,pb=b;sa=ma.state_dict();sb=mb.state_dict()
    if sa.keys()!=sb.keys():return False
    for name,x in sa.items():
        y=sb[name]
        if x.shape!=y.shape or x.dtype!=y.dtype or x.device!=y.device or not torch.equal(x,y):return False
    return fingerprint(sa)==fingerprint(sb) and fingerprint(ma.identity())==fingerprint(mb.identity()) and fingerprint(pa)==fingerprint(pb)


def assert_isolated(a,b):
    """Reject cross-pair writable sharing, including overlapping tensor storage/views.
Use storage intervals (whole storage, conservative) plus mutable container identities;
zero-sized tensors skip intervals. Also checks ndarray buffers used by NumPy RNG.
"""
    import torch,numpy as np
    def collect(pair):
        model,payload=pair;containers=set();intervals=[];seen=set()
        def walk(x):
            if id(x) in seen:return
            seen.add(id(x))
            if torch.is_tensor(x):
                containers.add(id(x))
                if x.numel()==0:return
                storage=x.untyped_storage();pointer=storage.data_ptr();size=storage.nbytes()
                if not pointer or size<=0:raise AssertionError('Nonempty tensor has invalid storage')
                intervals.append((str(x.device),pointer,pointer+size));return
            if isinstance(x,np.ndarray):
                containers.add(id(x))
                if x.dtype.hasobject:raise AssertionError('Object ndarray forbidden')
                if x.size:
                    ptr=int(x.__array_interface__['data'][0]);low=high=0
                    for length,stride in zip(x.shape,x.strides):
                        distance=(length-1)*stride;low+=min(0,distance);high+=max(0,distance)
                    if not ptr:raise AssertionError('Invalid ndarray storage')
                    intervals.append(('cpu',ptr+low,ptr+high+x.dtype.itemsize))
                return
            if isinstance(x,(dict,list,set,bytearray)):containers.add(id(x))
            if isinstance(x,dict):
                for k,v in x.items():walk(k);walk(v)
            elif isinstance(x,(list,tuple,set)):
                for item in x:walk(item)
        containers.add(id(model))
        # Explicit module containers, parameters and buffers; no arbitrary object graph.
        for module in model.modules():
            containers.add(id(module))
            for name in ('_parameters','_buffers','_modules'):walk(getattr(module,name))
        for _,tensor in model.named_parameters():walk(tensor)
        for _,tensor in model.named_buffers():walk(tensor)
        walk(payload);return containers,intervals
    ca,ia=collect(a);cb,ib=collect(b)
    if ca&cb:raise AssertionError('Cross-pair mutable container/module identity sharing')
    for da,lo,hi in ia:
        for db,otherlo,otherhi in ib:
            if da==db and lo<otherhi and otherlo<hi:raise AssertionError('Cross-pair tensor/ndarray storage overlap')
    return True


def approve_parent(m,p,standard):
    """Tensor-only postcondition; standard is private trusted (model,payload) from strict miss.
Never initializes, loads or forwards. Standard must never be a mutable returned pair.
"""
    import torch
    if p.get('step')!=600 or p.get('format_version')!='INDEPENDENT_MOTOR_TARGET_FEEDBACK_V1':raise AssertionError('Strict parent600 format required')
    if not model_payload_equal((m,p),standard):raise AssertionError('Parent differs from full strict-miss standard')
    m.verify_old();m.validate_architecture()
    if p['identity']!=m.identity():raise AssertionError('Payload/model identity mismatch')
    parameters=list(m.adapter_parameters());opt=p['optimizer_state_dict'];order=[v for group in opt['param_groups'] for v in group['params']]
    if len(parameters)!=8 or len(order)!=8 or len(set(order))!=8 or set(opt['state'])!=set(order):raise AssertionError('Canonical eight Adam slots required')
    for key,param in zip(order,parameters):
        slot=opt['state'][key];step=slot['step']
        if torch.is_tensor(step) and (step.numel()!=1 or step.device.type!='cpu' or not bool(torch.isfinite(step).all())):raise AssertionError('Invalid Adam counter')
        if float(step)!=600:raise AssertionError('Adam must remain at600')
        for name in ('exp_avg','exp_avg_sq'):
            value=slot[name]
            if value.shape!=param.shape or value.dtype!=param.dtype or value.device!=param.device or not bool(torch.isfinite(value).all()):raise AssertionError('Adam moment mismatch')
    anchor=p['config']['initial_anchor'];expected={n for n,_ in m.named_parameters() if not n.startswith('old.')}
    if set(anchor)!=expected or len(anchor)!=8 or fingerprint(anchor)!=fingerprint(standard[1]['config']['initial_anchor']):raise AssertionError('Original private anchor mismatch')
    return True


def callable_identity(m):
    """Public method source/code identity without invoking methods or forwards."""
    result={}
    def code_record(code):
        constants=[code_record(c) if isinstance(c,types.CodeType) else c for c in code.co_consts]
        return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,code.co_argcount,code.co_kwonlyargcount,code.co_flags,tuple(constants))
    for name in ('step','step_details','prime_old','cached_features','cache_sequence','initial_state','sequence'):
        bound=getattr(m,name);function=getattr(bound,'__func__',bound);code=function.__code__;path=Path(inspect.getsourcefile(function) or code.co_filename)
        if not path.is_file():raise AssertionError('Public method source unavailable: '+name)
        result[name]=dict(module=function.__module__,qualname=function.__qualname__,co_filename=code.co_filename,source_path=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),code_sha256=fingerprint(code_record(code)))
    return result


def runtime_identity():
    import torch,numpy as np
    return dict(python=platform.python_version(),executable=sys.executable,platform=platform.platform(),machine=platform.machine(),torch=torch.__version__,numpy=np.__version__,torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),default_dtype=str(torch.get_default_dtype()),default_device=str(torch.get_default_device()) if hasattr(torch,'get_default_device') else 'UNAVAILABLE',device='cpu',thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})

def rng_snapshot():
    import torch,numpy as np
    n=np.random.get_state();return dict(torch=torch.get_rng_state().clone(),numpy=(n[0],n[1].copy(),*n[2:]),python=random.getstate())

def restore_rng(state):
    import torch,numpy as np
    torch.set_rng_state(state['torch'].clone());n=state['numpy'];np.random.set_state((n[0],n[1].copy(),*n[2:]));random.setstate(state['python'])

def rng_equal(a,b):return fingerprint(a)==fingerprint(b)
