"""Pure regression: native-style DOF addresses must not index qpos sweeps."""
import ast
from pathlib import Path
import unittest
import numpy as np

class QposSweepTests(unittest.TestCase):
    def test_free_root_addresses(self):
        path=Path(__file__).with_name('tools_repair_moment_arms.py')
        tree=ast.parse(path.read_text())
        assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='leg_qpos' for t in n.targets))
        class Model:
            jnt_qposadr=np.r_[0,np.arange(7,49)]
            jnt_dofadr=np.r_[0,np.arange(6,48)]
        m=Model()
        def joints(_):return {f'joint_{j}':(j,int(m.jnt_dofadr[j]))for j in range(1,43)}
        addresses=eval(compile(ast.Expression(assign.value),str(path),'eval'),{'m':m,'leg_joints':joints,'int':int})
        self.assertEqual(addresses,list(range(7,49)))
        q=np.zeros(49);q[3]=1;before=q[:7].copy();q[addresses]+=.08
        np.testing.assert_array_equal(q[:7],before)
        self.assertEqual(q[48],.08)
        self.assertEqual(np.count_nonzero(q[7:]),42)
        augmented=[n for n in ast.walk(tree) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Subscript) and isinstance(n.target.value,ast.Attribute) and n.target.value.attr=='qpos']
        self.assertTrue(any(isinstance(n.target.slice,ast.Name)and n.target.slice.id=='leg_qpos' for n in augmented))

    def test_global_status_requires_sweep(self):
        tree=ast.parse(Path(__file__).with_name('tools_repair_moment_arms.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        nodes=[]
        for n in main.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='keyframe_ok' for t in n.targets):
                start=True
            if 'start' in locals() and start:
                nodes.append(n)
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript)and isinstance(t.slice,ast.Constant)and t.slice.value=='overall' for t in n.targets):break
        code=compile(ast.Module(body=nodes,type_ignores=[]),'status_gate','exec')
        for count,poses,expected in [(0,10,'PASS'),(1,10,'FAIL'),(0,0,'FAIL')]:
            env={'res':{'bidirectional_at_keyframe':42,'arm_matches_reference_on_repaired_joints':True},'ps':{'n_poses':poses,'worst_one_way_count':count}}
            exec(code,env)
            self.assertEqual(env['res']['status'],expected)
            self.assertEqual(env['res']['overall'],expected)

if __name__=='__main__':unittest.main()
