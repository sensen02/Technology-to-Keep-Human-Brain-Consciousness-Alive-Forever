import unittest
from tools_s2_scalar_methods import step,simulate

class MethodsTests(unittest.TestCase):
    def test_linear_steps(self):
        self.assertAlmostEqual(step(1,.1,lambda a:-a,'euler'),.9)
        self.assertAlmostEqual(step(1,.1,lambda a:-a,'rk4'),1-.1+.1**2/2-.1**3/6+.1**4/24)
    def test_unknown_method(self):
        with self.assertRaises(ValueError):step(0,.1,lambda a:0,'bad')
    def test_alignment(self):
        with self.assertRaises(ValueError):simulate(.00003,'pulse',0,'rk4')
    def test_event_and_finite(self):
        t,a=simulate(.0001,'pulse',0,'rk4');self.assertEqual(len(t),100)
        self.assertGreater(a[9],a[10]);self.assertTrue(all(0<=x<=.1 for x in a))

if __name__=='__main__':unittest.main()
