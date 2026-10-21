"""Small directed graph contracts, independent of cached/project simulation data."""
import unittest
import numpy as np
from scipy import sparse
from tools_p3_path_eligibility import adjacency, search, witness

class DirectedGraphTests(unittest.TestCase):
    def graph(self):
        # W[target,source]: 0->1->2, 3->1; node4 disconnected.
        return sparse.csr_matrix(([1., 2., 3.], ([1, 2, 1], [0, 1, 3])), shape=(5,5))
    def test_transposition_and_union(self):
        w = self.graph()
        wi = sparse.csr_matrix(([1.], ([3], [2])), shape=(5,5))
        a = adjacency(w, wi)
        d, p, cut = search(a, [0])
        self.assertEqual(d.tolist(), [0,1,2,3,-1])
        self.assertEqual(witness(p,d,3), [0,1,2,3])
        self.assertFalse(cut)
        self.assertTrue(all(a[s,t] for s,t in zip([0,1,2],[1,2,3])))
        self.assertEqual(search(a,[2])[0][0], -1)
    def test_disconnected_and_empty(self):
        a = adjacency(self.graph())
        d,p,c = search(a,[4])
        self.assertIsNone(witness(p,d,0))
        self.assertEqual(witness(p,d,4),[4])
        self.assertFalse(c)
        self.assertTrue(np.all(search(a,[])[0] == -1))
    def test_cutoff_is_not_disconnection(self):
        a = adjacency(self.graph())
        d,p,c = search(a,[0],1)
        self.assertTrue(c)
        self.assertEqual(d[2],-1)
        self.assertIsNone(witness(p,d,2))
        self.assertEqual(witness(p,d,1),[0,1])
        self.assertEqual(search(a,[0],2)[0][2],2)
        with self.assertRaises(ValueError): search(a,[0],-1)
    def test_zero_support_and_multisource(self):
        w = sparse.csr_matrix(([0.,1.],([1,2],[0,1])), shape=(3,3))
        a = adjacency(w)
        self.assertEqual(a.nnz,1)
        d,p,c = search(a,[0,1,1])
        self.assertEqual(witness(p,d,2),[1,2])
        with self.assertRaises(ValueError): search(a,[-1])

if __name__ == '__main__': unittest.main()
