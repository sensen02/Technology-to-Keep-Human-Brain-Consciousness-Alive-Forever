"""V4: audited deterministic adaptation of the frozen V3 sparse FK solver.

Only stance halfstride changes to0.125; seed both the joint curve and the single
constant orientation perleg from V3. Old sources/assets remain untouched. Literal
patches are count-checked against a frozen source hash, and the complete effective
implementation is saved and hashed before preregistration/FK for review. No physics.
"""
from pathlib import Path
import hashlib

SOURCE=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/s5-ann-walk/20261107T075536Z/prepare_stance_reference.py')
EXPECTED_SOURCE_SHA='6d444bedced45b4c34db79f082238eb7f2440e2736b072c22289ca9243e257d6'
OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T080221Z')


def adapt(text,old,new,count=1):
    actual=text.count(old)
    if actual!=count:
        raise ValueError(f'Frozen implementation patch mismatch: expected {count}, found {actual}: {old!r}')
    return text.replace(old,new)


def main():
    source=SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest()!=EXPECTED_SOURCE_SHA:
        raise ValueError('Frozen V3 source identity changed; refuse to adapt it')
    text=source.decode('utf-8')
    text=adapt(text,"PREVIOUS=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T075000Z')",
        "PREVIOUS=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T075536Z')")
    text=adapt(text,"OUT=Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T075536Z')",f"OUT=Path({str(OUT)!r})")
    text=adapt(text,"    ap.add_argument('--global-nfev',type=int,default=400)",
        "    ap.add_argument('--global-nfev',type=int,default=400)\n    ap.add_argument('--halfstride',type=float,default=.125)")
    text=adapt(text,"    assert args.solver_phases>=65 and args.export_phases>=args.solver_phases",
        "    assert args.solver_phases>=65 and args.export_phases>=args.solver_phases\n    assert 0 < args.halfstride <= .15")
    text=adapt(text,"params=dict(duty=.65,halfstride=.15,clearance=.25",
        "params=dict(duty=.65,halfstride=args.halfstride,clearance=.25")
    text=adapt(text,"constant_orientation_component_bounds_rad=[-.15,.15],**vars(args))",
        "constant_orientation_component_bounds_rad=[-.15,.15],**{k:v for k,v in vars(args).items() if k!='halfstride'})")
    text=adapt(text,"sources=[Path(__file__),OLD,",
        f"sources=[Path(__file__),Path({str(SOURCE)!r}),OUT/'EFFECTIVE_PREPARATION.py',OLD,")
    text=adapt(text,"stage='S5 FK-only v3 constant perleg stance orientation',parameters=params,",
        "stage='S5 FK-only v4 shorter stance with constant perleg orientation',parameters=params,\n        reason='V3 failed position; measured LM/RM rearward workspace at-.125 has errors~.002, versus>.005 at-.15. Only halfstride is changed, not acceptance criteria.',")
    text=adapt(text,"Warmstart v2 cyclic curve resampled to128unique phases by bounded periodic PCHIP; optimize allphase joints and3constant orientation parameters jointly perleg, max400nfev; no arbitrary extra cycles.",
        "Warmstart V3 joints AND constant orientation offsets, resampled to128unique phases by bounded periodic PCHIP; optimize allphase joints and3constant orientation parameters jointly perleg, max400nfev; no extra cycles.")
    text=adapt(text,"fixed='Original free root/q0anchor y/z, duty.65, stancehalfstride.15, clearance.25, constant backwards stancex slope unchanged.',",
        "fixed=f'Original free root/q0anchor y/z, duty.65, clearance.25 unchanged; halfstride{args.halfstride}; backwards stance slope2*halfstride/.65.',")
    text=adapt(text,"oldphase=z['solver_phase'].copy();oldq=z['solver_qpos'].copy()",
        "oldphase=z['solver_phase'].copy();oldq=z['solver_qpos'].copy();oldoffset=z['constant_orientation_offsets_rad'].copy()")
    text=adapt(text,".65,.15,.25)",".65,args.halfstride,.25)",count=2)
    text=adapt(text,"x=np.r_[curve.ravel(),np.zeros(3)]","x=np.r_[curve.ravel(),oldoffset[leg]]")
    text=adapt(text,"version='constant-stance-orientation-v3'","version='constant-stance-orientation-v4'")
    text=adapt(text,'V3 fixed-root FK only; constant nearby stance orientation; physical walking NOT_ESTABLISHED',
        'V4 fixed-root FK only; shorter stance; physical walking NOT_ESTABLISHED')
    text=adapt(text,'# V3 constant stance-orientation reference, FK only',
        '# V4 shorter-stance constant-orientation reference, FK only')
    text=adapt(text,
        '"Actual vectors and angle norms are in METADATA.json and VERIFICATION.json. Duty0.65, stance halfstride0.15, swing clearance0.25, fixed stance y/z and backward slope are unchanged.\\n\\n"',
        'f"Actual vectors and angle norms are in METADATA.json and VERIFICATION.json. Duty0.65, stance halfstride{args.halfstride}, swing clearance0.25; fixed stance y/z and constant backwards speed are retained.\\n\\n"')
    OUT.mkdir(parents=True,exist_ok=True)
    effective=OUT/'EFFECTIVE_PREPARATION.py'
    if effective.exists():
        effective.read_bytes()
        raise FileExistsError('Preserve prior effective source; choose a new timestamp')
    compile(text,str(effective),'exec')
    effective.write_text(text)
    # __file__ identifies this reviewable wrapper; preregistration additionally
    # hashes frozen V3 and the full effective implementation written above.
    exec(compile(text,str(effective),'exec'),{'__name__':'__main__','__file__':str(Path(__file__).resolve())})


if __name__=='__main__':
    main()
