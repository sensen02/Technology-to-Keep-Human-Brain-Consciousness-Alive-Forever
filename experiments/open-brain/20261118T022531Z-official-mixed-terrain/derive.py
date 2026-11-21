"""Derive the terrain runner from the frozen flat-ground comparator."""
from pathlib import Path

trial = Path(__file__).resolve().parent
prior = trial.parent / "20261118T002600Z-ascending-behavior-comparison"
source = (prior / "live.py").read_text()
source = source.replace("from neural_runtime import *", """from arena_trial import TRIAL, MixedBody, measure as terrain_measure
from neural_runtime import *
from repair_binding import FrontTouchNeck, CanonicalEvents""")
source = source.replace("from circuit import Circuit\nfrom fixed_neck import FixedCircuit\n", "")
source = source.replace("('native_full','reference_boundary','translated','no_up')",
                        "('native_full','reference_boundary','translated','repaired')")
start = source.index("def measure")
end = source.index("@torch.inference_mode()\ndef run", start)
source = source[:start] + source[end:]
source = source.replace("neck=FixedCircuit({'ports':neck_ports},batch=4)", """neck=FrontTouchNeck({'ports':neck_ports},an_roots,
                        matched_lanes=(1,3),feedback_lanes=(3,),batch=4)""")
start = source.index("    # Reference ANs are owned")
end = source.index("    # The native_full", start)
source = source[:start] + source[end:]
source = source.replace("bodies=[Body(height=0,adhesion=True,seed=seed) for _ in MODES]\n"
                        "    for body in bodies:body.restore(seed);setup(body,'terminal')",
                        "bodies=[MixedBody(seed) for _ in MODES]\n"
                        "    for body in bodies:setup(body,'terminal')")
source = source.replace("    n=20000", """    write(out/'ARENA.json',dict(initial=bodies[0].initial_receipt(), tiles=bodies[0].tiles))
    n=20000""")
source = source.replace("nonfoot=(),control=(42,),pads=(6,),normal_load=(6,),command=(2,)).items()}",
                        "nonfoot=(),control=(42,),pads=(6,),normal_load=(6,),command=(2,),\n"
                        "        terrain_normal=(7,6),feet_xyz=(6,3)).items()}")
source = source.replace("drive=np.zeros((4,len(neck.roots)),dtype=np.float32);ref_drive=np.zeros((1,len(ref.roots)),dtype=np.float32)",
                        "drive=np.zeros((4,len(neck.roots)),dtype=np.float32)")
source = source.replace("    ref_rate=np.zeros(2);ref_alpha=-np.expm1(-DT/.020);phase=np.zeros((4,2))\n"
                        "    delays=neck.neck_delay[an_cols];pending=np.zeros((int(delays.max())+1,2),dtype=bool)",
                        "    canonical=CanonicalEvents((4,2));legacy_phase=np.zeros(2)")
source = source.replace("    pins=sources(extra_files);pins.update(assets)",
                        "    extra_files.update(TRIAL.glob('*.py'));extra_files.add(TRIAL/'PROTOCOL.zh-CN.md')\n"
                        "    pins=sources(extra_files);pins.update(assets)")
source = source.replace("        native_full_brain_unchanged=True",
                        "        matched_lanes=[1,3],matched_feedback_lanes=[3],arena='FlyGym 2.1.0 MixedTerrainWorld',\n"
                        "        native_full_brain_unchanged=True")
source = source.replace("    for tick in range(n):", "    warning=None\n    for tick in range(n):")
source = source.replace("drive.fill(0.);ref_drive.fill(0.);aff_drive.fill(0.)", "drive.fill(0.);aff_drive.fill(0.)")
source = source.replace("            if use[1,c]:ref_drive[0,ref_inputs[c]]=68.75\n", "")
start = source.index("        ref_volley=")
end = source.index("        edrive.fill", start)
source = source[:start] + source[end:]
start = source.index("        ref_rate+=")
end = source.index("        active=", start)
source = source[:start] + """        events=canonical.step(arrivals[:,an_cols])
        rates=canonical.rate.copy();rates[0]=0.
        rates[2]=enc(torch.from_numpy(filter_state[2]).double()).numpy()
        events[0]=False
        legacy_phase+=rates[2]*DT;events[2]=legacy_phase>=1.;legacy_phase-=events[2]
""" + source[end:]
source = source.replace("b.apply(controls[lane],pads[lane]);load,_,nf,_=b.contact();loads[lane]=load[[0,3]]",
                        "b.apply(controls[lane],pads[lane]);load,nf,ground=b.detail();loads[lane]=load[[0,3]]")
source = source.replace("normal_load=load,command=command[lane])",
                        "normal_load=load,command=command[lane],\n"
                        "                 terrain_normal=ground,feet_xyz=b.d.xpos[b.feet])")
source = source.replace("                raise RuntimeError('physical warning/nonfinite; cannot certify this batch')",
                        "                warning=dict(lane=lane,tick=tick,model=b.d.warning.number.tolist())\n"
                        "        if warning is not None:\n"
                        "            print('physical warning',warning,flush=True);break")
source = source.replace("    result=[]", """    n=tick+1
    neural={key:array[:n] for key,array in neural.items()}
    result=[]""")
source = source.replace("v[:,lane].copy()", "v[:n,lane].copy()")
source = source.replace("metrics=measure(arrays,bodies[lane])", "metrics=terrain_measure(arrays,bodies[lane])")
source = source.replace("nominal_reference_lif_ticks=n,native_receptor_lif_ticks=n",
                        "physics_warning=warning,matched_AN_soma_ticks=n*2,native_receptor_lif_ticks=n")
assert all(x not in source for x in ("reference_arrivals", "ref_state", "no_up", "ref_drive"))
(trial / "run_live.py").write_text(source)
