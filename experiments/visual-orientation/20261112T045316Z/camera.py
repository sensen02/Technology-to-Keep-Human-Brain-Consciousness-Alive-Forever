"""Render-only engineering binocular cameras. Heavy imports only at explicit call."""
import math


def eye_frames(head_position, head_rotation, geometry_audit, spec, optical_yaw_deg=0.):
    import numpy as np
    aabb=geometry_audit['head']['local_mesh_aabb']
    forward=aabb['max'][0]+.1*aabb['extent'][0]
    lateral=max(abs(aabb['min'][1]),abs(aabb['max'][1]))+.1*aabb['extent'][1]
    rot=np.asarray(head_rotation);pos=np.asarray(head_position);out=[]
    for side,yaw in zip((1.,-1.),spec['yaw_deg']):
        y=math.radians(yaw+optical_yaw_deg);p=math.radians(spec['pitch_deg'])
        local_forward=np.array([math.cos(p)*math.cos(y),math.cos(p)*math.sin(y),math.sin(p)])
        local_up=np.array([-math.sin(p)*math.cos(y),-math.sin(p)*math.sin(y),math.cos(p)])
        origin=pos+rot@np.array([forward,side*lateral,spec['height_offset']])
        out.append(dict(pos=origin,forward=rot@local_forward,up=rot@local_up))
    return out


def set_gl_camera(scene,frame,spec):
    import numpy as np
    near=spec['near_native'];top=near*math.tan(math.radians(spec['vertical_fov_deg'])/2.)
    f=np.asarray(frame['forward']);u=np.asarray(frame['up'])
    if max(abs(np.linalg.norm(f)-1),abs(np.linalg.norm(u)-1),abs(f@u))>1e-10:
        raise ValueError('camera optical basis must be orthonormal')
    # Renderer uses the two GL cameras for stereo averaging even in mono mode.
    # Equal settings prevent an unintended stereo eye displacement.
    for camera in scene.camera:
        camera.pos[:]=frame['pos'];camera.forward[:]=f;camera.up[:]=u
        camera.frustum_center=0.;camera.frustum_width=2*top
        camera.frustum_bottom=-top;camera.frustum_top=top
        camera.frustum_near=near;camera.frustum_far=spec['far_native'];camera.orthographic=0


def add_decor(scene,mj,kind,size,position,rotation,rgba):
    import numpy as np
    if scene.ngeom>=scene.maxgeom:raise RuntimeError('visual scene capacity')
    geom=scene.geoms[scene.ngeom]
    mj.mjv_initGeom(geom,kind,np.asarray(size,dtype=float),np.asarray(position,dtype=float),np.asarray(rotation,dtype=float).ravel(),np.asarray(rgba,dtype=np.float32))
    geom.objtype=int(mj.mjtObj.mjOBJ_UNKNOWN);geom.objid=-1;geom.segid=-1
    geom.category=int(mj.mjtCatBit.mjCAT_DECOR);scene.ngeom+=1


def render_eye_pair(renderer,model,data,frames,spec,decorator=None):
    import numpy as np
    import mujoco as mj
    option=mj.MjvOption();option.geomgroup[:]=1;option.flags[mj.mjtVisFlag.mjVIS_TENDON]=False
    images=[];metadata=[]
    for frame in frames:
        renderer.update_scene(data,scene_option=option)
        set_gl_camera(renderer.scene,frame,spec)
        if decorator is not None:decorator(renderer.scene)
        image=renderer.render().copy()
        if image.shape!=(32,32,3) or image.dtype!=np.uint8:raise ValueError('actual RGB32x32 required')
        images.append(image)
        metadata.append({key:np.asarray(value).tolist() for key,value in frame.items()})
    return np.stack(images),metadata


class PendingVision:
    """50Hz actual RGB capture,20ms release delay,200Hz held-frame queries."""
    def __init__(self):
        self.pending=None;self.capture_time=None;self.last_query=None;self.released=None;self.released_capture_time=None
    def _release(self,time_s):
        if self.pending is not None and time_s>=self.capture_time+.020-1e-10:
            self.released=self.pending;self.released_capture_time=self.capture_time;self.pending=None
    def capture(self,time_s,current_rgb):
        import numpy as np
        self._release(time_s)
        if self.capture_time is not None and abs(time_s-self.capture_time-.020)>1e-8:
            raise ValueError('camera capture interval must be20ms')
        if self.pending is not None:raise ValueError('pending exposure overwritten before release')
        self.pending=np.asarray(current_rgb).copy();self.capture_time=float(time_s)
    def query(self,time_s):
        import numpy as np
        if self.last_query is not None and abs(time_s-self.last_query-.005)>1e-8:
            raise ValueError('motor queries must be contiguous5ms')
        self._release(time_s);self.last_query=float(time_s)
        return dict(image=np.zeros((2,32,32,3),dtype=np.uint8) if self.released is None else self.released.copy(),valid=self.released is not None,capture_time_s=self.released_capture_time,release_time_s=None if self.released is None else self.released_capture_time+.020)
