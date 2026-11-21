"""Read the user's Blender orchard and export small ground-apple approaches."""
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[sys.argv.index('--')+1])
X = np.linspace(-1.6,3.2,121)
Y = np.linspace(-2.,2.,101)
DISTANCE_M = .0176
MOUTH_HEIGHT_M = .00115
GROUND_ASSETS = ('grass','litter','twig','stone')


def geometry(obj):
    mesh=obj.data
    mesh.calc_loop_triangles()
    coords=np.empty(len(mesh.vertices)*3,dtype=np.float32)
    mesh.vertices.foreach_get('co',coords)
    faces=np.empty(len(mesh.loop_triangles)*3,dtype=np.int32)
    mesh.loop_triangles.foreach_get('vertices',faces)
    m=np.asarray(obj.matrix_world,dtype=np.float64)
    return coords.reshape(-1,3)@m[:3,:3].T+m[:3,3],faces.reshape(-1,3)


def world_raycast(obj):
    to_local=obj.matrix_world.inverted()
    to_world=obj.matrix_world

    def cast(origin,direction):
        hit,point,normal,_=obj.ray_cast(to_local@Vector(origin),to_local.to_3x3()@Vector(direction))
        if not hit:
            return None
        return np.asarray(to_world@point),np.asarray(to_local.to_3x3().transposed()@normal)
    return cast


def belt(cast,ground,center,zmin,zmax,outward):
    xy=center[:2]+.16*outward
    samples=[]
    for z in np.arange(zmin+.00016,min(zmax,zmin+.030),.00023):
        hit=cast((*xy,z),(-outward[0],-outward[1],0.))
        if hit is None:
            continue
        point,normal=hit
        height=point[2]-ground(*point[:2])
        # Require actual fruit skin above soil and a surface facing the start.
        normal/=np.linalg.norm(normal)
        if .00075<=height<=.00165 and np.dot(normal[:2],outward)>.16:
            samples.append((abs(height-MOUTH_HEIGHT_M),point,normal,height))
    return min(samples,key=lambda item:item[0]) if samples else None


def local_coordinates(world, origin, axes):
    return (world-origin)@axes.T*100.


def clip_soil(world_triangles,origin,axes):
    lo,hi=world_triangles.min(1),world_triangles.max(1)
    nearby=np.all(hi>=origin-(.043,.043,.015),axis=1)&np.all(lo<=origin+(.043,.043,.015),axis=1)
    local=local_coordinates(world_triangles[nearby],origin,axes)
    faces=[]
    for tri in local:
        polygon=tri.tolist()
        for axis,boundary,sign in ((0,X[0],1),(0,X[-1],-1),(1,Y[0],1),(1,Y[-1],-1)):
            clipped=[]
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                da,db=sign*(a[axis]-boundary),sign*(b[axis]-boundary)
                if da>=0.:
                    clipped.append(a)
                if (da<0.<db) or (db<0.<da):
                    mix=da/(da-db)
                    clipped.append([av+(bv-av)*mix for av,bv in zip(a,b)])
            polygon=clipped
        faces.extend((polygon[0],polygon[i],polygon[i+1]) for i in range(1,len(polygon)-1))
    return np.asarray(faces)


def filtered(debris, start, axes, ground_z):
    origin=np.r_[start,ground_z]
    within=[]
    for name,label,tri in debris:
        lo=np.min(tri,axis=1)
        hi=np.max(tri,axis=1)
        broad=np.all(hi>=origin-(.042,.042,.003),axis=1)&np.all(lo<=origin+(.042,.042,.014),axis=1)
        t=local_coordinates(tri[broad],origin,axes)
        if not len(t):
            continue
        a,b=t.min(1),t.max(1)
        keep=(b[:,0]>=X[0])&(a[:,0]<=X[-1])&(b[:,1]>=Y[0])&(a[:,1]<=Y[-1])
        keep&=(b[:,2]>=-.20)&(a[:,2]<=.85)
        if np.any(keep):
            within.append((name,label,t[keep]))
    return within


def count_corridor(clipped, ground):
    stats=dict(start=0,direct=0,adjacent=0)
    for _,_,tri in clipped:
        low,high=tri.min(1),tri.max(1)
        center=tri.mean(axis=1)
        soil=np.array([ground(x,y) for x,y in center[:,:2]])
        active=high[:,2]-soil>.035
        stats['start']+=int(np.count_nonzero(active&(low[:,0]<.26)&(high[:,0]>-.26)&
                                             (low[:,1]<.24)&(high[:,1]>-.24)))
        ahead=(low[:,0]<1.48)&(high[:,0]>.34)
        stats['direct']+=int(np.count_nonzero(active&ahead&(low[:,1]<.12)&(high[:,1]>-.12)))
        stats['adjacent']+=int(np.count_nonzero(active&ahead&(low[:,1]<.6)&(high[:,1]>-.6)))
    return stats


def export(name, candidate, ground, soil_mesh):
    write=OUT/name
    write.mkdir()
    point,normal,origin,axes,clipped,apple,stats,h=(
        candidate[key] for key in ('point','normal','origin','axes','clipped','apple','stats','height'))
    grid=np.empty((len(Y),len(X)),dtype=np.float64)
    for i,y in enumerate(Y):
        for j,x in enumerate(X):
            world=origin+axes.T@np.array([x,y,0.])/100.
            grid[i,j]=(ground(*world[:2])-origin[2])*100.
    fr_v,fr_i=geometry(apple)
    # Keep the original facets that can meet the body or the local range rays.
    skin=local_coordinates(fr_v[fr_i],origin,axes)
    low,high=skin.min(1),skin.max(1)
    keep=(high[:,0]>=X[0])&(low[:,0]<=X[-1]+.35)&(high[:,1]>=Y[0]-.3)&(low[:,1]<=Y[-1]+.3)
    keep&=(high[:,2]>=grid.min()-.22)&(low[:,2]<=.8)
    skin=skin[keep]
    assert len(skin)>0
    goal=local_coordinates(point,origin,axes)
    assert abs(goal[0]-100*DISTANCE_M)<1e-4 and abs(goal[1])<1e-5
    classes=[(n,c,len(t)) for n,c,t in clipped]
    labels=np.concatenate([np.full(len(t),i,dtype=np.int16) for i,(_,_,t) in enumerate(clipped)]) if clipped else np.zeros(0,dtype=np.int16)
    triangles=np.concatenate([t for _,_,t in clipped]) if clipped else np.empty((0,3,3))
    skin_normal=axes@normal
    soil=clip_soil(soil_mesh,origin,axes)
    assert len(soil)>0
    np.savez_compressed(write/'CROP.npz',terrain=grid,x=X,y=Y,skin=skin,debris=triangles,soil=soil,
                        labels=labels,goal=goal,skin_normal=skin_normal,origin=origin,axes=axes)
    meta=dict(apple=apple['fruit_id'],object=apple.name,source_blend=bpy.data.filepath,
              angle_deg=candidate['angle'],start_world_m=origin.tolist(),patch_world_m=point.tolist(),
              patch_local_cm=goal.tolist(),patch_above_soil_mm=h*1000,source_skin_triangles=len(skin),
              clipped_soil_triangles=len(soil),
              source_debris_triangles=len(triangles),objects=classes,corridor=stats,
              elevation_mm=[10*float(grid.min()),10*float(grid.max())])
    (write/'SOURCE.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
    print(name,meta,flush=True)


def main():
    tagged=[o for o in bpy.context.scene.objects if o.get('orchard_class')]
    soil,=[o for o in tagged if o['orchard_class']=='soil']
    soil_v,soil_face=geometry(soil)
    soil_mesh=soil_v[soil_face]
    cast_ground=world_raycast(soil)

    def ground(x,y):
        hit=cast_ground((float(x),float(y),.80),(0.,0.,-1.))
        if hit is None:
            raise ValueError(f'No soil below {(x,y)}')
        return float(hit[0][2])

    debris=[]
    for obj in tagged:
        if obj['orchard_class'] in GROUND_ASSETS:
            verts,faces=geometry(obj)
            debris.append((obj.name,obj['orchard_class'],verts[faces]))
    candidates=[]
    for fruit in [o for o in tagged if o.get('fruit_id') in {f'ground-{i:03d}' for i in range(1,7)}]:
        cast=world_raycast(fruit)
        corners=np.array([np.asarray(fruit.matrix_world@Vector(p)) for p in fruit.bound_box])
        center=corners.mean(0)
        for angle in range(0,360,15):
            yaw=math.radians(angle)
            outward=np.array([math.cos(yaw),math.sin(yaw)])
            located=belt(cast,ground,center,corners[:,2].min(),corners[:,2].max(),outward)
            if located is None:
                continue
            _,point,normal,h=located
            start=point[:2]+DISTANCE_M*outward
            gz=ground(*start)
            origin=np.r_[start,gz]
            axes=np.array([[-outward[0],-outward[1],0.],[outward[1],-outward[0],0.],[0.,0.,1.]])
            clipped=filtered(debris,start,axes,gz)
            def local_ground(x,y):
                wx,wy,_=origin+axes.T@np.array([x,y,0.])/100.
                return (ground(wx,wy)-gz)*100.
            stats=count_corridor(clipped,local_ground)
            rise=abs(local_ground(1.76,0.))
            if rise>.32:
                continue
            candidates.append(dict(apple=fruit,angle=angle,point=point,normal=normal,origin=origin,
                                   axes=axes,clipped=clipped,stats=stats,height=h,rise=rise))
    assert candidates
    safe=[c for c in candidates if c['stats']['start']==0]
    keyed=lambda c:(c['apple']['fruit_id'],c['angle'])
    open_cases=sorted((c for c in safe if not c['stats']['direct']),
                      key=lambda c:(c['stats']['adjacent'],c['rise'],keyed(c)))
    blocked_cases=sorted((c for c in safe if 0<c['stats']['direct']<=25),
                         key=lambda c:(abs(c['stats']['direct']-5),c['stats']['adjacent'],keyed(c)))
    assert open_cases and blocked_cases
    blocked=blocked_cases[0]
    clear=next((c for c in open_cases if c['apple']==blocked['apple']),open_cases[0])
    overview=[dict(fruit=c['apple']['fruit_id'],angle=c['angle'],stats=c['stats'],rise_cm=c['rise'])
              for c in candidates]
    (OUT/'SURVEY.json').write_text(json.dumps(overview,indent=2)+'\n')
    export('CLEAR',clear,ground,soil_mesh)
    export('DEBRIS',blocked,ground,soil_mesh)


if __name__=='__main__':
    bpy.context.view_layer.update()
    main()
