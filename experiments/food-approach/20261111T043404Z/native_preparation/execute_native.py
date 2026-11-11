"""Selected native recording entry. Everything physical is lazy and authorization-gated."""
from pathlib import Path
import argparse
import importlib.util
import sys
import types
from contract import HERE, ROOT, BASE, admitted, require, sha, load, save_new


def parent_module():
    require('model_io' not in sys.modules or Path(sys.modules['model_io'].__file__).resolve() == ROOT/'model_io.py', 'wrong candidate model loader already imported')
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('_selected_cem_native_evaluator', ROOT/'evaluate_motor.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def execute(plan_path, checkpoint, digest, out):
    plan, proof = admitted(plan_path, checkpoint, digest)
    out = Path(out).resolve(); case = HERE/'selected_native'
    require(out == case/'EVAL_left' and not out.exists(), 'fresh scoped native LEFT output required')
    require(load(case/'READINESS.json')['all_passed'] is True, 'actual selected-model preparation missing')
    parent = parent_module()
    function = parent.bind(case)
    globals_ = dict(function.__globals__)
    original_pin = globals_['pin']
    own_sources = [HERE/name for name in ('contract.py','execute_native.py','pipeline.py','verify_native.py','render_selected.py','METHODS_DRAFT.json','RUN_PLAN.json')]
    selected_paths = [Path(p) for p in proof['hashes']]
    def source_paths():
        paths = parent.source_paths()+own_sources+selected_paths+[case/'EVAL_PREREG.json',case/'READINESS.json',case/'SOURCE_BARRIER.json']
        for path in paths:
            require(path.is_file(), 'pre-Core native source missing: '+str(path))
        return list(dict.fromkeys(path.resolve() for path in paths))
    def pin():
        sw, original_identity, io = original_pin()
        import numpy as np
        NativeRecorder, Trace = sw.NativeRecorder, sw.Trace
        class TaggedRecorder(NativeRecorder):
            def before(self, core):
                row = super().before(core)
                row.update(native_control_source=np.asarray(core._native_source_code),
                           logical_ms_index=np.asarray(core._native_logical_index),
                           neural_decision_index=np.asarray(core._native_decision_index),
                           neural_command=core._native_neural_command.copy(),
                           thorax_rotation_world_before=core.d.xmat[core.th].reshape(3,3).copy())
                return row
        class TaggedTrace(Trace):
            def tick(self, core, action, source):
                logical = len(self.rec)
                core._native_source_code = int(source)
                core._native_logical_index = logical
                decision = -1 if logical < 1000 else (logical-1000)//5
                core._native_decision_index = decision
                core._native_neural_command = np.zeros(2) if decision < 0 else parent.command_at('left', decision)
                return super().tick(core, action, source)
        def capture(core, config, extra_paths=()):
            import mujoco as mj
            physics = sys.modules['physics']
            require(Path(physics.__file__).resolve() == BASE/'physics.py', 'unexpected native physical source')
            mesh = core.plant.geom_type == mj.mjtGeom.mjGEOM_MESH
            require(float(core.d.time) == 0., 'runtime contract must precede all integration')
            static = dict(scope='derived complete body / finite PD / native units; SI and biology NOT_ESTABLISHED',
                          mass_native=float(core.plant.body_subtreemass[core.th]),
                          gravity_native=core.plant.opt.gravity.tolist(),weight_native=float(physics.W),
                          initial_heading_world_xy=core.initial_heading.tolist(),
                          xml=str(Path(physics.PAD_XML).resolve()),xml_sha256=sha(physics.PAD_XML),
                          nq=int(core.plant.nq),nv=int(core.plant.nv),nu=int(core.plant.nu),
                          native_dt_s=float(core.plant.opt.timestep),mesh_count=int(mesh.sum()),
                          all_mesh_opaque=bool(np.all(core.plant.geom_rgba[mesh,3] == 1.)),
                          geom_friction=core.plant.geom_friction.tolist(),adhesion_enabled=bool(core.adhesion_on),
                          instrumentation='additional copied recorder columns and source tags only; original physical step/control unchanged',
                          condition='left',checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=digest,
                          run_plan=str(Path(plan_path).resolve()),run_plan_sha256=sha(plan_path),
                          engineering_left=proof['cases']['left'],teacher_prefix_s=4.8,settle_s=1.,autonomous_s=9.6)
            static_path = out/'NATIVE_STATIC.json'; save_new(static_path, static)
            cfg = dict(config,kind='selected CEM3bias LEFT once native recording',native_run_plan_sha256=sha(plan_path),
                       native_static_sha256=sha(static_path),no_actor_input_change=True)
            return original_identity.capture(core,cfg,extra_paths=list(extra_paths)+[static_path,Path(plan_path)])
        sw.NativeRecorder, sw.Trace = TaggedRecorder, TaggedTrace
        proxy = types.SimpleNamespace(**{name:getattr(original_identity,name) for name in ('atomic_json','verify','verify_runtime','sha256')})
        proxy.capture = capture
        return sw, proxy, io
    globals_.update(pin=pin,source_paths=source_paths,CHECKPOINT=Path(checkpoint).resolve())
    bound = types.FunctionType(function.__code__,globals_,function.__name__,function.__defaults__,function.__closure__)
    args = argparse.Namespace(checkpoint=str(Path(checkpoint).resolve()),sha256=digest,condition='left',out=str(out),training_complete=True,native=True)
    return bound(args)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-plan',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--launch-authorized',action='store_true')
    a=p.parse_args();require(a.launch_authorized,'explicit parent --launch-authorized required')
    execute(a.run_plan,a.checkpoint,a.sha256,a.out)

if __name__ == '__main__':
    main()
