"""Inert S141 verifier tests: no runner imports/model calls or real trial execution."""
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import verify_neural_activation_initialized_probe as v


class StaticS141(unittest.TestCase):
    def test_inert(self):
        self.assertFalse(any(k in sys.modules for k in ('mujoco','torch','engine.neural_cond','run_neural_activation_initialized_probe')))
        ast.parse(Path(v.__file__).read_text())

    def test_scalar_equilibrium_and_decay(self):
        prm=np.array([.0005,.0004,0.])
        self.assertEqual(v.scalar_step(.3,.3,prm),.3)
        self.assertGreater(v.scalar_step(0.,1.,prm),0.)
        self.assertLess(v.scalar_step(1.,0.,prm),1.)
        initial=np.linspace(0.,1.,90)
        out=v.reconstruct_activation(initial,np.repeat(initial[None],6,axis=0),np.repeat(prm[None],90,axis=0))
        np.testing.assert_array_equal(out,np.repeat(initial[None],121,axis=0))

    def test_canonical_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source'; p.write_text('inert')
            alias=Path(d)/'alias'; alias.symlink_to(p)
            with self.assertRaises(ValueError): v.canonical_once({str(p):'a',str(alias):'b'})
            self.assertEqual(len(v.canonical_once({str(p):'a',str(alias):'a'})),1)

    def test_exact_source_audit(self):
        runner=v.ROOT/'run_neural_activation_initialized_probe.py'
        tree=ast.parse(runner.read_text()); values={}
        for n in tree.body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('ANCHOR','INSERTION'):
                values[n.targets[0].id]=ast.literal_eval(n.value)
        original=(v.ROOT/'run_extended_initialization_probe.py').read_text()
        compiled=original.replace(values['ANCHOR'],values['ANCHOR']+values['INSERTION'])
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); (base/'PRIVATE_RUNNER_SOURCE.py').write_text(compiled)
            pre=dict(private_source_insertion=dict(compiled_source_sha256=hashlib.sha256(compiled.encode()).hexdigest(),original_sha256=v.sha(v.ROOT/'run_extended_initialization_probe.py')))
            checks=v.Checks(); v.audit_runner(checks,pre,base/'init80ms')
            self.assertTrue(checks.passed,[r for r in checks.rows if not r['passed']])


if __name__=='__main__': unittest.main()
