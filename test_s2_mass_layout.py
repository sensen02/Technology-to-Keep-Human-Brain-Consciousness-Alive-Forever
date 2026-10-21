import unittest
import numpy as np
from tools_s2_mass_layout import decode_tree,decode_csr


def fixture():
    # Branch at node1; siblings2/3 have no direct coupling in this layout.
    dense=np.array([[10.,1.,2.,3.],[1.,11.,4.,5.],[2.,4.,12.,0.],[3.,5.,0.,13.]])
    tree=np.array([10,11,1,12,4,2,13,5,3],dtype=float)
    csr=np.array([10,1,11,2,4,12,3,5,13],dtype=float)
    return dense,tree,np.array([-1,0,1,1]),csr,np.array([0,1,3,6]),np.array([1,2,3,3]),np.array([0,0,1,0,1,2,0,1,3])


class MassLayoutTests(unittest.TestCase):
    def test_explicit_independent_oracle(self):
        M,q,p,v,a,n,c=fixture()
        np.testing.assert_array_equal(decode_tree(q,p),M)
        np.testing.assert_array_equal(decode_csr(v,a,n,c,4),M)
        np.linalg.cholesky(M)

    def test_wrong_buffer_identity_not_inferable(self):
        M,q,p,v,a,n,c=fixture()
        self.assertEqual(len(q),len(v))
        wrong=decode_tree(v,p)
        self.assertFalse(np.array_equal(wrong,M))
        # Structural validation cannot certify caller buffer provenance.
        self.assertEqual(wrong[1,1],1.)

    def test_seeded_forests(self):
        rng=np.random.default_rng(12971)
        for nv in range(1,20):
            for repeat in range(4):
                p=np.array([-1]+[int(rng.integers(-1,i)) for i in range(1,nv)])
                M=np.zeros((nv,nv));support=[]
                for i in range(nv):
                    chain=[];j=i
                    while j!=-1:chain.append(j);j=int(p[j])
                    support.append(chain)
                    for j in chain[1:]:M[i,j]=M[j,i]=rng.normal()
                for i in range(nv):M[i,i]=1+sum(abs(M[i,j]) for j in range(nv))
                q=np.array([M[i,j] for i,chain in enumerate(support) for j in chain])
                columns=[sorted(chain) for chain in support]
                count=np.array([len(cs) for cs in columns]);adr=np.r_[0,np.cumsum(count)[:-1]]
                cols=np.array([j for cs in columns for j in cs]);v=np.array([M[i,j] for i,cs in enumerate(columns) for j in cs])
                np.testing.assert_array_equal(decode_tree(q,p),M)
                np.testing.assert_array_equal(decode_csr(v,adr,count,cols,nv),M)
                np.linalg.cholesky(M)

    def test_single_and_disconnected(self):
        np.testing.assert_array_equal(decode_tree([2,3],[-1,-1]),np.diag([2,3]))
        np.testing.assert_array_equal(decode_csr([2,3],[0,1],[1,1],[0,1],2),np.diag([2,3]))
        np.testing.assert_array_equal(decode_tree([2],[-1]),[[2]])

    def test_structure_does_not_require_spd(self):
        np.testing.assert_array_equal(decode_tree([-2],[-1]),[[-2]])
        np.testing.assert_array_equal(decode_csr([0],[0],[1],[0],1),[[0]])

    def test_copy_and_input_unchanged(self):
        M,q,p,v,a,n,c=fixture();before=[x.copy() for x in (q,p,v,a,n,c)]
        one=decode_tree(q,p);two=decode_csr(v,a,n,c,4);one[:]=99;two[:]=88
        for x,y in zip((q,p,v,a,n,c),before):np.testing.assert_array_equal(x,y)

    def test_bad_tree(self):
        for q,p in [([1],[-1,0]),([1,2],[0]),([1,2],[-1,1]),([1],[-2]),([1],[-1.]),([1],[True]),([],[]),([1],[[ -1 ]]),([1],np.array([2**64-1],dtype=np.uint64))]:
            with self.subTest(parent=p),self.assertRaises(ValueError):decode_tree(q,p)

    def test_bad_csr(self):
        for a,n,c,v in [([0,1],[1,2],[0,1,1],[1,2,3]),([0,2],[1,1],[0,1],[1,2]),([0,0],[1,1],[0,1],[1,2]),([0,1],[1,1],[0,0],[1,2]),([0,1],[1,1],[1,1],[1,2]),([0,1],[1,1],[0,1],[1,2,3]),([0.,1.],[1,1],[0,1],[1,2]),([0,1],[1,0],[0,1],[1,2])]:
            with self.subTest(a=a,n=n,c=c),self.assertRaises(ValueError):decode_csr(v,a,n,c,2)
        with self.assertRaises(ValueError):decode_csr([1],[0],[1],np.array([2**64-1],dtype=np.uint64),1)
        with self.assertRaises(ValueError):decode_csr([1],[0],[1],[0],True)
        with self.assertRaises(ValueError):decode_tree(np.ones(257),np.full(257,-1))

    def test_invalid_values(self):
        for bad in ([np.nan],[np.inf],[1j],['1'],[True],[[1]]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):decode_tree(bad,[-1])
            with self.subTest(bad=bad),self.assertRaises(ValueError):decode_csr(bad,[0],[1],[0],1)


if __name__=='__main__':unittest.main()
