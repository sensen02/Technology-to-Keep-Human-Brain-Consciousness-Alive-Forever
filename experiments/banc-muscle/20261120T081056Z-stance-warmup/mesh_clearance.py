"""Certify separation of convex leg meshes using transformed vertices."""
import itertools

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from solve import LEG_PREFIXES, SEGMENTS


def cross_leg_clearance(model, data, trunk=False):
    surfaces={}
    trunk_names={'Thorax_geom','Head_geom','A1A2_geom','A3_geom','A4_geom','A5_geom','A6_geom'}
    for g in range(model.ngeom):
        name=model.geom(g).name
        is_leg=name[:2] in LEG_PREFIXES and any(s in name for s in SEGMENTS)
        if not is_leg and not (trunk and name in trunk_names):
            continue
        mesh=model.geom_dataid[g]
        start=model.mesh_vertadr[mesh]
        verts=model.mesh_vert[start:start+model.mesh_vertnum[mesh]].astype(float)
        verts=verts[ConvexHull(verts).vertices] @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]
        surfaces[name]=(verts,verts.min(axis=0),verts.max(axis=0))
    margins=[]
    for a,b in itertools.combinations(surfaces,2):
        if trunk:
            if (a in trunk_names)==(b in trunk_names):
                continue
            if not any(s in (b if a in trunk_names else a) for s in ('Femur','Tibia','Tarsus')):
                continue
        if a[:2]==b[:2]:
            continue
        va,la,ha=surfaces[a]; vb,lb,hb=surfaces[b]
        box=np.linalg.norm(np.maximum(np.maximum(la-hb,lb-ha),0))
        if box>=.004:
            margins.append((box,a,b))
            continue
        # n*a <= offset, n*b >= offset+gap, abs(n[k])<=1. A positive gap is a witness.
        inequalities=np.vstack((np.column_stack((va,-np.ones(len(va)),np.zeros(len(va)))),
                                np.column_stack((-vb,np.ones(len(vb)),np.ones(len(vb))))))
        fit=linprog([0,0,0,0,-1],A_ub=inequalities,b_ub=np.zeros(len(inequalities)),
                    bounds=[(-1,1)]*3+[(None,None),(0,None)], method='highs')
        if not fit.success:
            raise RuntimeError(fit.message)
        n=fit.x[:3]
        gap=(min(vb@n)-max(va@n))/np.linalg.norm(n) if np.linalg.norm(n)>1e-10 else 0.
        margins.append((float(gap),a,b))
    return dict(pairs=len(margins),minimum_separation_lower_bound_mm=min(margins)[0],
                closest_pair=list(min(margins)[1:]), unresolved=[list(row) for row in margins if row[0]<1e-6])
