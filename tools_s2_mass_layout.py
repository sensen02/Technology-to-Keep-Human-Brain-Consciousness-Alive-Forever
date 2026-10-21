"""Pinned strict mass layout subsets; pure NumPy, not runtime provenance."""
import numpy as np


def indices(value, size, name, lower, upper):
    a=np.asarray(value)
    if a.shape!=(size,) or a.dtype.kind not in 'iu' or np.any(a<lower) or np.any(a>upper):
        raise ValueError(name)
    return a.astype(np.int64,copy=True)


def values_array(value):
    a=np.asarray(value)
    if a.ndim!=1 or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('values must be finite real 1D')
    with np.errstate(over='raise',invalid='raise'):
        try: a=a.astype(float,copy=True)
        except FloatingPointError as exc: raise ValueError('float conversion') from exc
    if not np.isfinite(a).all(): raise ValueError('float conversion')
    return a


def dimension(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not 0<value<=256:
        raise ValueError('nv')
    return int(value)


def decode_tree(qM, parent):
    p=np.asarray(parent)
    if p.ndim!=1: raise ValueError('parent shape')
    n=dimension(len(p))
    p=indices(p,n,'parent',-1,n-1)
    if any(int(p[i])>=i for i in range(n)): raise ValueError('parent order/cycle')
    v=values_array(qM)
    expected=0
    for i in range(n):
        j=i
        while j!=-1:expected+=1;j=int(p[j])
    if len(v)!=expected:raise ValueError('tree storage length')
    M=np.zeros((n,n));address=0
    for i in range(n):
        j=i
        while j!=-1:
            M[i,j]=M[j,i]=v[address];address+=1;j=int(p[j])
    return M


def decode_csr(values, rowadr, rownnz, colind, nv):
    n=dimension(nv);v=values_array(values)
    adr=indices(rowadr,n,'rowadr',0,len(v))
    nnz=indices(rownnz,n,'rownnz',1,n)
    col=indices(colind,len(v),'colind',0,n-1)
    offset=0;M=np.zeros((n,n))
    for i in range(n):
        a=int(adr[i]);count=int(nnz[i]);end=a+count
        if a!=offset or end>len(v): raise ValueError('noncontiguous CSR storage')
        cols=col[a:end]
        if cols[-1]!=i or np.any(cols>i) or np.any(np.diff(cols)<=0):raise ValueError('lower triangle/diagonal/column order')
        for k,j in enumerate(cols):M[i,j]=M[j,i]=v[a+k]
        offset=end
    if offset!=len(v):raise ValueError('unconsumed CSR storage')
    return M
