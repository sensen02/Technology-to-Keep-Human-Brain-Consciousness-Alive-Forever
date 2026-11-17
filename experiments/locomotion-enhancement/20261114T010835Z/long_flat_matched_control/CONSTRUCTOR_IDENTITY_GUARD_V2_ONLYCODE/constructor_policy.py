"""Stdlib constructor identity audit; no physical imports or construction."""
from pathlib import Path
import contextlib
import hashlib
import inspect
import sys
PHYSICS=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T152714Z/physics.py')
REPAIR=PHYSICS.parent/'sources/repair_core.py'

def normalized(filename):
    if not filename or str(filename).startswith('<'): return str(filename)
    return str(Path(str(filename).split('::',1)[0]).resolve())
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

class ConstructorAudit:
    def __init__(self,expected_hashes,counters,expected_paths=None):
        self.expected_hashes=expected_hashes
        self.expected_paths=set(expected_paths or (str(PHYSICS.resolve()),str(REPAIR.resolve())))
        self.counters=counters
        self.events=[]
        self.physical_objects={}
        self.hash_cache={}
        self.expected_code_objects=set()
    def source(self,filename):
        name=normalized(filename)
        if name not in self.hash_cache:
            p=Path(name)
            self.hash_cache[name]=sha(p) if p.is_file() else None
        return dict(co_filename=filename,normalized_source=name,source_sha256=self.hash_cache[name])
    def class_info(self,cls):
        rows=[]
        for base in cls.__mro__:
            init=vars(base).get('__init__')
            code=getattr(init,'__code__',None)
            filename=code.co_filename if code is not None else None
            rows.append(dict(module=base.__module__,qualname=base.__qualname__,init_qualname=getattr(code,'co_qualname',None),source=self.source(filename) if filename is not None else None))
        return rows
    def refresh_codes(self):
        module=sys.modules.get('physics')
        if module is not None:
            for m in (module,getattr(module,'rc',None)):
                cls=getattr(m,'Core',None)
                code=getattr(getattr(cls,'__init__',None),'__code__',None)
                if code is not None and normalized(code.co_filename) in self.expected_paths:
                    self.expected_code_objects.add(code)
    def observe(self,frame):
        code=frame.f_code
        if code.co_name!='__init__' or not code.co_qualname.endswith('Core.__init__'): return
        instance=frame.f_locals.get('self')
        info=self.source(code.co_filename)
        self.refresh_codes()
        penultimate=code.co_qualname.split('.')[-2]
        exact_source=info['normalized_source'] in self.expected_paths
        source_hash_ok=exact_source and self.expected_hashes.get(info['normalized_source'])==info['source_sha256']
        qualified=penultimate=='Core' and source_hash_ok
        row=dict(sequence=len(self.events),module_name=frame.f_globals.get('__name__'),qualname=code.co_qualname,constructor_source=info,self_id=id(instance),self_type_module=type(instance).__module__,self_type_qualname=type(instance).__qualname__,MRO=self.class_info(type(instance)),matches_expected_code_object=code in self.expected_code_objects,matches_expected_source=exact_source,source_hash_matches=source_hash_ok,classification='PHYSICAL_CONSTRUCTOR' if qualified else ('PHYSICAL_TERRAIN_WRAPPER' if penultimate=='TerrainCore' and any(normalized(getattr(getattr(c,'__init__',None),'__code__',code).co_filename) in self.expected_paths for c in type(instance).__mro__) else 'FOREIGN_CONSTRUCTOR'))
        self.events.append(row)
        if exact_source and not source_hash_ok:
            raise RuntimeError('Physical constructor source SHA mismatch')
        if qualified:
            # Retain references so object id reuse cannot conceal a second entity.
            self.physical_objects[id(instance)]=instance
            self.counters['base_Core_constructors']=len(self.physical_objects)
            if len(self.physical_objects)>1:
                raise RuntimeError('Exactly one source-qualified physical Core is permitted')
    def report(self):
        return dict(expected_source_paths=sorted(self.expected_paths),expected_source_hashes=self.expected_hashes,constructor_events=self.events,unique_physical_object_ids=list(self.physical_objects),foreign_constructor_events=sum(e['classification']=='FOREIGN_CONSTRUCTOR' for e in self.events),identification='Core.__init__ exact normalized source+frozen SHA; code-object match additionally logged; same object super layers count once',old125_second_identity='UNKNOWN_UNTIL_NEW_ACTUAL_TRACE')

@contextlib.contextmanager
def prohibit_execution(mj,np,counters,audit):
    originals=[];previous=sys.getprofile()
    def step(*args,**kwargs):
        counters['forbidden_step_attempts']+=1
        raise RuntimeError('Zero-step guard prohibits integration')
    def payload(*args,**kwargs):
        counters['payload_attempts']+=1
        raise RuntimeError('Zero-step guard prohibits array/checkpoint decoding')
    def profile(frame,event,arg):
        if event=='call':
            if frame.f_code.co_name in ('forward','_call_impl','_wrapped_call_impl','load_policies'):
                counters['neural_attempts']+=1
                raise RuntimeError('Zero-step guard prohibits neural forwards/policy loading')
            audit.observe(frame)
        if previous is not None: previous(frame,event,arg)
    try:
        for name in ('mj_step','mj_step1','mj_step2'):
            originals.append((mj,name,getattr(mj,name)));setattr(mj,name,step)
        for name in ('load','loadtxt','genfromtxt','fromfile'):
            originals.append((np,name,getattr(np,name)));setattr(np,name,payload)
        sys.setprofile(profile)
        yield
    finally:
        sys.setprofile(previous)
        for owner,name,value in reversed(originals): setattr(owner,name,value)
