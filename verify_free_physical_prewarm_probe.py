"""Independent offline verifier for the S142 bounded physical prewarm probe.

This verifier never imports MuJoCo, a model, or the producer runner.  It checks
producer metadata/arrays, the S121 prefix, native 25-us closure, and the
conservative first-failure stop.  A stopped run is valid evidence but is never
labelled prewarm success.
"""
from __future__ import annotations
from pathlib import Path
import argparse, ast, csv, hashlib, json, math, sys, traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_free_physical_prewarm_probe.py"
BASELINE_DEFAULT = Path("/home/sensen/Desktop/engine修复验收/s121-constraint-row-recording/20261024T072500Z/init80ms/TRAJECTORY.npz")

class Checks:
    def __init__(self): self.rows=[]
    @property
    def passed(self): return all(r["passed"] for r in self.rows)
    def add(self,name,passed,**details):
        self.rows.append({"name":name,"passed":bool(passed),**details})

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def load_json(p): return json.loads(Path(p).read_text())
def npz(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def arr(d,*names):
    for n in names:
        if n in d:return np.asarray(d[n])
    return None

def source_audit(checks):
    if not RUNNER.exists():
        checks.add("producer runner present",False,reason="run_free_physical_prewarm_probe.py absent")
        return {}
    tree=ast.parse(RUNNER.read_text()); imports=[]; qwrites=[]; resets=[]; steps=[]
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            s=ast.unparse(n); imports.append(s)
            # Producer imports physics only inside main; this verifier remains inert.
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            ts=n.targets if isinstance(n,ast.Assign) else [n.target]
            for t in ts:
                u=ast.unparse(t)
                if "qpos" in u:qwrites.append(dict(line=n.lineno,source=ast.unparse(n)))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            if n.func.attr in ("reset","mj_resetData","mj_resetDataKeyframe"):resets.append(dict(line=n.lineno,source=ast.unparse(n)))
            if n.func.attr in ("step","mj_step"):steps.append(n.lineno)
    checks.add("no explicit reset after initialization",not resets,resets=resets)
    checks.add("producer has stepping",bool(steps) or "mj_step" in RUNNER.read_text(),step_lines=steps)
    return dict(imports=imports,qpos_writes=qwrites,resets=resets,step_lines=steps)

def baseline_prefix(checks,run,baseline):
    bpath=Path(run.get("baseline_path",baseline))
    if bpath.is_dir(): bpath=bpath/"TRAJECTORY.npz"
    if not bpath.exists():
        checks.add("S121 baseline available",False,path=str(bpath)); return
    try:
        r=npz(run["trajectory_path"]); b=npz(bpath)
    except Exception as e:
        checks.add("trajectory arrays readable",False,error=str(e)); return
    n=min(81,len(r.get("time",[])),len(b.get("time",[])))
    keys=[k for k in ("time","qpos","qvel","act","ctrl","spikes","features","applied","current","qfrc_applied","xfrc_applied") if k in r and k in b]
    for k in keys:
        x,y=r[k][:n],b[k][:n]
        m=min(len(x),len(y)); x,y=x[:m],y[:m]
        compatible=x.shape==y.shape
        delta=float(np.max(np.abs(x-y))) if compatible and np.issubdtype(x.dtype,np.number) else None
        checks.add("S121 prefix bitexact "+k,compatible and np.array_equal(x,y),max_abs=delta,actual_shape=list(x.shape),baseline_shape=list(y.shape))

def verify(out,checks,baseline):
    meta_path = next((p for p in (out/"RUN_METADATA.json", out/"PREWARM_METADATA.json", out/"init80ms"/"RESULT.json", out/"RESULT.json", out/"STOP.json") if p.exists()), None)
    if meta_path is None:
        checks.add("producer metadata available",False,reason="expected RUN_METADATA/PREWARM_METADATA/RESULT/STOP")
        return {}
    run=load_json(meta_path)
    run["_metadata_path"] = str(meta_path)
    checks.add("metadata declares bounded diagnostic",str(run.get("evidence_class",run.get("status",""))).lower() in ("bounded_diagnostic","diagnostic","stopped","failed"),value=run.get("evidence_class",run.get("status")))
    n=int(run.get("physical_samples",run.get("n_samples",run.get("samples",0))))
    steps=int(run.get("physical_steps",max(n-1,0)))
    checks.add("variable sample bound n<=401",0<n<=401,n=n)
    checks.add("steps agree with samples",steps==n-1,steps=steps)
    duration=float(run.get("duration_s",run.get("physical_duration_s",run.get("duration_ms",0)/1000)))
    first=float(run.get("first_failure_ms",run.get("first_failed_ms",run.get("stop_time_ms",run.get("actual_duration_ms",float("nan"))))))
    checks.add("first load failure declared 1.925ms",math.isfinite(first) and abs(first-1.925)<=1e-9,actual=first)
    checks.add("stop immediately at first failure",(bool(run.get("stopped_immediately",run.get("early_stop",False))) or run.get("status")=="STOPPED") and bool(run.get("stop_reasons")),metadata=run)
    checks.add("no 5ms sensory update",not bool(run.get("sensory_update_at_5ms",False)) and (not math.isfinite(duration) or duration<.005),duration_s=duration)
    pre=int(run.get("pre_ticks",run.get("PRE",run.get("initialization_ticks",160)))); nt=int(run.get("neural_ticks",run.get("neural_tick_count",run.get("calls",{}).get("neural_ticks",0))))
    expected=pre+math.ceil(max(0,steps)/20)
    checks.add("neural ticks PRE160 plus ceil(steps/20)",pre==160 and nt==expected,pre=pre,actual=nt,expected=expected)
    checks.add("partial final tick recorded",bool(run.get("partial_final_tick",run.get("last_partial_tick",run.get("status")=="STOPPED"))),metadata=run)
    checks.add("zero applied external force",not bool(run.get("external_force_nonzero",False)),metadata=run)
    checks.add("not prewarm success",run.get("prewarm_success",False) is False,prewarm_success=run.get("prewarm_success"))
    tp=run.get("trajectory_path",out/"TRAJECTORY.npz")
    if not Path(tp).is_absolute():tp=out/tp
    run["trajectory_path"]=str(tp)
    if Path(tp).exists():
        a=npz(tp); t=arr(a,"time"); checks.add("trajectory readable",t is not None)
        if t is not None:
            checks.add("native 25us clock",len(t)==n and np.allclose(np.diff(t[:-1]),25e-6,rtol=0,atol=1e-12))
            for k,v in a.items(): checks.add("finite "+k,np.isfinite(v).all())
            for k in ("qpos","qvel","qfrc_actuator","qfrc_constraint","qfrc_passive"):
                if k in a: checks.add("full48 closure field "+k,a[k].shape[0]==n,shape=list(a[k].shape))
            # Independent dense constraint-row J.T@force reconstruction, including all 48 DOFs.
            rows_path=Path(tp).with_name("CONSTRAINT_ROWS.npz")
            if rows_path.exists() and "qfrc_constraint" in a:
                rz=np.load(rows_path,allow_pickle=False); jerrs=[]
                for i in range(n):
                    j=rz[f"sample{i:03d}_efc_J_dense"]; f=rz[f"sample{i:03d}_efc_force"]
                    if j.ndim==2 and j.shape[1]==48: jerrs.append(float(np.max(np.abs(j.T@f-a["qfrc_constraint"][i]))))
                checks.add("all48 constraint J.T force closure",len(jerrs)==n and max(jerrs,default=np.inf)<=2e-10,max_abs=max(jerrs,default=np.inf),samples=len(jerrs))
            # Trapezoid impulse and COM momentum diagnostics are evidence only;
            # they do not assert exact impulse equality or identify elastic energy.
            if all(k in a for k in ("external_force_world","external_impulse_world","time")):
                f=a["external_force_world"]; tt=a["time"]
                imp=np.vstack([np.zeros((1,3)),np.cumsum(.5*(f[1:]+f[:-1])*np.diff(tt)[:,None],axis=0)])
                checks.add("external impulse trapezoid reconstruction",np.allclose(imp,a["external_impulse_world"],atol=1e-12,rtol=1e-10))
            if all(k in a for k in ("contact_force_world","contact_impulse_world","time")):
                f=a["contact_force_world"]; tt=a["time"]
                imp=np.vstack([np.zeros((1,3)),np.cumsum(.5*(f[1:]+f[:-1])*np.diff(tt)[:,None],axis=0)])
                checks.add("contact impulse trapezoid reconstruction",np.allclose(imp,a["contact_impulse_world"],atol=1e-12,rtol=1e-10))
            if all(k in a for k in ("com_momentum_world","external_force_world","time")):
                dp=np.diff(a["com_momentum_world"],axis=0)
                fg=.5*(a["external_force_world"][1:]+a["external_force_world"][:-1])*np.diff(a["time"])[:,None]
                checks.add("COM momentum residual recorded",np.isfinite(dp-fg).all(),max_abs=float(np.max(np.abs(dp-fg))))
            report_energy = {k: bool(k in a) for k in ("native_energy","signed_power_actuator","signed_power_passive","signed_power_constraint","signed_power_contact_rows")}
            run["energy_fields_present"] = report_energy
            run["energy_interpretation"] = "power dot products and native energy fields are diagnostic; actuator work includes active and passive muscle terms; contact work cannot identify stored elastic energy versus nonconservative contact without logging"
            run["active_passive_split"] = "NOT_IDENTIFIED: aggregate actuator force/power is not pure active work; no claim of elastic release"
            # Kinematic/actuator closure when producer recorded all terms.
            if all(k in a for k in ("qfrc_actuator","qfrc_constraint","qfrc_passive","qfrc_bias")) and "qfrc_applied" in a:
                rhs=a["qfrc_actuator"]+a["qfrc_constraint"]+a["qfrc_passive"]+a["qfrc_bias"]+a["qfrc_applied"]
                if "qfrc_total" in a: checks.add("native all48 force closure",np.allclose(a["qfrc_total"],rhs,atol=1e-10,rtol=1e-10))
    baseline_prefix(checks,run,baseline)
    return run

def write_csv(out,checks):
    p=out/"INDEPENDENT_CHECK.csv"
    with p.open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["name","passed","details_json"])
        for r in checks.rows:w.writerow([r["name"],r["passed"],json.dumps(r,default=str,ensure_ascii=False)])
    return p

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True,type=Path); p.add_argument("--report",type=Path); p.add_argument("--baseline",type=Path,default=BASELINE_DEFAULT); a=p.parse_args()
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True); checks=Checks(); report={"passed":False,"evidence_valid":False,"prewarm_success":False,"behavior_claim":"NONE","physical_units":"NOT_ESTABLISHED","verifier":str(Path(__file__).resolve()),"verifier_sha256":sha(__file__)}
    try:
        report["source_audit"]=source_audit(checks)
        run=verify(out,checks,a.baseline.resolve()); report["run_metadata"]=run
        report["passed"]=checks.passed; report["evidence_valid"]=checks.passed and not run.get("prewarm_success",False)
    except Exception as e: report.update(error=type(e).__name__+": "+str(e),traceback=traceback.format_exc())
    report["checks"]=checks.rows; report["output_csv"]=str(write_csv(out,checks))
    target=(a.report or out/"INDEPENDENT_CHECK.json").resolve(); target.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+"\n")
    print(json.dumps({"passed":report["passed"],"evidence_valid":report["evidence_valid"],"report":str(target)}))
    return 0 if report["passed"] else 1
if __name__=="__main__": raise SystemExit(main())
