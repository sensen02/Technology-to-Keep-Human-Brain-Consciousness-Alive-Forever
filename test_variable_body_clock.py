"""Physics-free contract for explicit body dt; never invokes a body episode."""
import ast
import inspect
import unittest
from pathlib import Path
from engine.embodied.closure_clock import body_clock_steps, push_steps

ROOT = Path(__file__).resolve().parent

class VariableBodyClock(unittest.TestCase):
    def test_declared_grids(self):
        for dt, per, total in [(25e-6,200,2400),(12.5e-6,400,4800),(6.25e-6,800,9600)]:
            self.assertEqual(body_clock_steps(dt,.0005,.005,.06),(per,total))
            self.assertEqual(push_steps(dict(start_s=.01,end_s=.015,force_model=[1,0,0]),.06,dt),(round(.01/dt),round(.015/dt)))
    def test_reject_bad_clock(self):
        for dt in [0,-1,float('inf'),float('nan'),.0003,.001,7e-6]:
            with self.assertRaises(ValueError):body_clock_steps(dt,.0005,.005,.06)
        with self.assertRaises(ValueError):body_clock_steps(25e-6,.0005,.005,.06001)
        with self.assertRaises(ValueError):body_clock_steps(25e-6,.0005,.005,.060025)
    def test_legacy_default_and_local_duration(self):
        tree=ast.parse((ROOT/'run_neural_muscle_closure.py').read_text())
        funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name in ['prepare_body','episode']:
            f=funcs[name]; args=[a.arg for a in f.args.args]
            default=dict(zip(args[-len(f.args.defaults):],f.args.defaults))['body_dt_s']
            self.assertIsInstance(default,ast.Name);self.assertEqual(default.id,'DT_BODY')
        constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('DT_')}
        self.assertEqual(constants,dict(DT_BODY=.0001,DT_NEURAL=.0005,DT_COMMAND=.005))
        self.assertEqual(body_clock_steps(constants['DT_BODY'],.0005,.005,.06),(50,600))
        # Episode must never consume historical global DT_BODY except its default.
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='DT_BODY' for s in funcs['episode'].body for n in ast.walk(s)))
        self.assertIn('model.opt.timestep = float(body_dt_s)',ast.get_source_segment((ROOT/'run_neural_muscle_closure.py').read_text(),funcs['prepare_body']))
        self.assertIn('body_dt',ast.unparse(funcs['episode']))

if __name__=='__main__':unittest.main()
