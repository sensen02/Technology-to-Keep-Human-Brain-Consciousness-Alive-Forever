"""Plane-strain contact of an idealized access wall with registered nerve traces."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTriP1, ElementVector, MeshTri, asm
from skfem.helpers import ddot, sym_grad, trace

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FIXED = BASE / "20261117T164207Z-fixed-neck" / "PARAMETERS.json"
CATALOG = BASE / "20261117T051249Z-bidirectional" / "CATALOG.json"
ROOTS = ("720575941633589179", "720575941626534090", "720575941500851362")
DT = .01
CONTACT_RADIUS = 16.0
PENETRATION = 10.0
SHEATH = 3.0
E_INTERIOR, E_SHEATH = 2.0, 8.0  # kPa, declared engineering cases
NU, TAU = .40, .075  # dimensionless, s; engineered effective solid


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reconstruct(root, parameters, catalog):
    row = parameters[root]
    path = Path(row["swc"])
    assert sha(path) == row["sha256"]
    data = np.loadtxt(path, comments="#")
    tree = {int(r[0]): (int(r[6]), r[2:5]) for r in data}
    a, b = row["crossing_children"]
    boundary = np.array([catalog[root]["geometry"][k] for k in ("front_nm", "rear_nm")])
    for i, j in ((a, b), (b, a)):
        chain = [i]
        while chain[-1] in tree and chain[-1] != j:
            chain.append(tree[chain[-1]][0])
        if chain[-1] == j:
            l, h = boundary if i == a else boundary[::-1]
            coords = np.vstack([l, *[tree[k][1] for k in chain[1:]], h]) / 1000.
            if i != a:
                coords = coords[::-1].copy()
            assert abs(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()-row["length_um"]) < 1e-7
            return coords, path
    raise ValueError(f"no continuous registered branch for {root}")


def contact_penetration(t, unload):
    if t <= .45:
        return 0.
    if t < .65:
        return PENETRATION * (t - .45) / .20
    if not unload or t <= 1.05:
        return PENETRATION
    return PENETRATION * np.clip((1.20 - t)/.15, 0., 1.)


def model(nx, ny, end, width):
    x = np.linspace(0., end, nx+1)
    y = np.r_[np.linspace(0., SHEATH, 4)[:-1], np.linspace(SHEATH, width-SHEATH, ny+1),
              np.linspace(width-SHEATH, width, 4)[1:]]
    mesh = MeshTri.init_tensor(x, y)
    base = Basis(mesh, ElementVector(ElementTriP1()))

    @BilinearForm
    def stiffness(u, v, w):
        modulus = np.where((w.x[1] < SHEATH) | (w.x[1] > width - SHEATH), E_SHEATH, E_INTERIOR)
        mu = modulus / (2*(1+NU))
        lam = modulus * NU / ((1+NU)*(1-2*NU))
        eu, ev = sym_grad(u), sym_grad(v)
        return 2*mu*ddot(eu,ev) + lam*trace(eu)*trace(ev)

    matrix = asm(stiffness, base).tocsr()
    axial, lateral = base.nodal_dofs
    coordinates = mesh.p
    ends = np.isclose(coordinates[0],0.) | np.isclose(coordinates[0],end)
    far = np.isclose(coordinates[1], width)
    fixed = np.unique(np.r_[axial[ends],lateral[ends],lateral[far]])
    wall = np.flatnonzero(np.isclose(coordinates[1],0.) & ~ends)
    return mesh, base, matrix, fixed, wall


def run_case(nx, ny, dt, traces, end, width, unfold, sticking):
    mesh, basis, k, fixed, wall = model(nx, ny, end, width)
    axial, lateral = basis.nodal_dofs
    normal, tangent = lateral[wall], axial[wall]
    x = mesh.p[0,wall]
    centered = x-end*.50
    cap = np.full(len(x), -1e6)
    on_cap = abs(centered) <= CONTACT_RADIUS
    cap[on_cap] = np.sqrt(CONTACT_RADIUS**2-centered[on_cap]**2)-CONTACT_RADIUS
    coefficient = k * (1+TAU/dt)
    past = np.zeros(basis.N)
    time = np.linspace(0,2,round(2/dt)+1)
    probes = [basis.probes(p[:,[1,0]].T).tocsr() for p in traces]
    responses = []
    snapshots = []
    contact = np.zeros(len(wall),dtype=bool)
    worst_free = worst_gap = 0.
    for t in time:
        penetration = contact_penetration(t, unfold)
        gap = penetration+cap if penetration > 0 else np.full(len(wall),-1e6)
        rhs = (TAU/dt)*(k@past)
        for _ in range(35):
            u = np.zeros_like(past)
            u[normal[contact]] = gap[contact]
            clamped = np.r_[fixed,normal[contact]]
            if sticking:
                u[tangent[contact]] = past[tangent[contact]]
                clamped = np.r_[clamped,tangent[contact]]
            free = np.setdiff1d(np.arange(len(u)),clamped,assume_unique=False)
            reduced = rhs[free]-coefficient[free][:,clamped]@u[clamped]
            u[free] = spsolve(coefficient[free][:,free], reduced)
            force = coefficient@u-rhs
            release = contact & (force[normal] < -1e-8)
            add = ~contact & (u[normal] < gap-1e-8)
            if not (release.any() or add.any()):
                break
            contact = (contact & ~release) | add
        else:
            raise RuntimeError(f"unilateral contact did not converge at {t}")
        worst_free = max(worst_free, float(np.max(np.abs(force[free]))))
        worst_gap = max(worst_gap, float(np.max(np.maximum(gap-u[normal],0.))))
        past = u
        field = basis.interpolate(u).grad
        xx, yy = field[0,0], field[1,1]
        xy = .5*(field[0,1]+field[1,0])
        center, spread = .5*(xx+yy), np.hypot(.5*(xx-yy),xy)
        peak_strain = float(np.max(np.maximum(np.abs(center+spread),np.abs(center-spread))))
        row = [t,penetration,float(np.maximum(force[normal],0.).sum()),float(np.max(np.hypot(u[axial],u[lateral]))),peak_strain]
        for coords, sample in zip(traces,probes,strict=True):
            axial_shift,lateral_shift = (sample@u).reshape(2,-1)
            deformed = coords.copy()
            deformed[:,0]+=lateral_shift
            deformed[:,1]+=axial_shift
            length = float(np.linalg.norm(np.diff(deformed,axis=0),axis=1).sum())
            transverse = np.abs(lateral_shift).max()
            row.extend([length,float(transverse)])
        responses.append(row)
        if np.any(np.isclose(t,[.45,.65,1.05,1.20,2.00],atol=1e-9)):
            snapshots.append((t,u[axial].copy(),u[lateral].copy()))
    assert worst_free < 1e-6 and worst_gap < 1e-6
    return np.asarray(responses), (mesh.p,mesh.t,snapshots), (worst_free,worst_gap)


def main():
    parameters=json.loads(FIXED.read_text())["ports"]
    catalog=json.loads(CATALOG.read_text())["ports"]
    sourced=[reconstruct(r,parameters,catalog) for r in ROOTS]
    raw=[p for p,_ in sourced]
    paths=[p.copy() for p in raw]
    start=min(p[:,1].min() for p in paths)
    lateral_min=min(p[:,0].min() for p in paths)-6.
    end=max(p[:,1].max() for p in paths)-start
    width=max(p[:,0].max() for p in paths)-lateral_min+7.
    for p in paths:
        p[:,0]-=lateral_min
        p[:,1]-=start
    # Include boundary points in the FE domain despite floating subtraction.
    for p in paths:
        p[:,0]=np.clip(p[:,0],0,width)
        p[:,1]=np.clip(p[:,1],0,end)
    print(f"registered chart length {end:.3f} um x engineered width {width:.3f} um",flush=True)
    output={}
    summaries={}
    for name,nx,ny,dt,unload,stick in (
        ("release",72,28,DT,True,False),
        ("retained",72,28,DT,False,False),
        ("stick_release",72,28,DT,True,True),
        ("refined",108,42,DT/2,True,False)):
        response,mesh,errors=run_case(nx,ny,dt,paths,end,width,unload,stick)
        print(name,response.shape, response[:,2:5].max(0).tolist(), flush=True)
        output[name]=response
        if name=="release":
            output["mesh_xy"],output["mesh_cells"],snapshots=mesh
            output["snapshot_time"]=np.array([x[0] for x in snapshots])
            output["snapshot_displacement"]=np.array([x[1:] for x in snapshots])
        length= response[:,5::2]
        summaries[name]=dict(increments=len(response), max_peak_strain=float(response[:,4].max()),
            max_displacement_um=float(response[:,3].max()), max_line_reaction_kpa_um=float(response[:,2].max()),
            max_abs_path_change_um=np.abs(length-length[0]).max(0).tolist(),
            path_change_at_end_um=(length[-1]-length[0]).tolist(),
            static_no_overlap_um=errors[1], free_dof_residual=errors[0])
    assert np.array_equal(output["release"][:106],output["retained"][:106])
    assert output["release"][-1,3]<.01 and output["retained"][-1,3]>.1
    for i,p in enumerate(paths):output[f"axon{i}"]=p
    np.savez_compressed(HERE/"MECHANICS.npz",**output)
    info=dict(label="ENGINEERING_PREOPENED_ACCESS_CONTACT_ONLY", roots=ROOTS, source_sha256={
        str(p):sha(p) for p in (FIXED,CATALOG,Path(__file__),*[p for _,p in sourced])},
        chart=dict(axis0="BANC y head to thorax",axis1="BANC x transverse", axial_origin_um=start,
            lateral_origin_um=lateral_min,width_um=width,length_um=end,unmeshed_z_retained=True),
        settings=dict(plane_strain=True,linear_viscoelastic=True,E_kPa=[E_INTERIOR,E_SHEATH],
            nu=NU,voigt_tau_s=TAU, assumed_sheath_um=SHEATH,boundary_probe_radius_um=CONTACT_RADIUS,
            imposed_contact_interference_um=PENETRATION, actual_electrode_geometry=False,
            material_parameters_measured_in_this_neck=False,tissue_rupture_predicted=False),
        summary=summaries, output_sha=sha(HERE/"MECHANICS.npz"))
    (HERE/"MECHANICS.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
