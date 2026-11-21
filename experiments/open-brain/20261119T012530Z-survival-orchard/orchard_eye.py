"""Engineering binocular RGB front end for the P1 orchard."""
from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import binary_closing, find_objects, label, sobel
from scipy.optimize import linear_sum_assignment
from dm_control.mujoco import wrapper

WIDTH, HEIGHT = 384, 216
FOVY = 56.
BASELINE_CM = .044
FOCAL_PX = HEIGHT / (2*np.tan(np.deg2rad(FOVY/2)))
CONNECTIVITY = np.ones((3,3),dtype=bool)


@dataclass(frozen=True)
class Candidate:
    pixel: tuple[float,float]
    box: tuple[int,int,int,int]
    area_px: int
    bearing_deg: float
    elevation_deg: float
    clipped: bool
    hue_band: int
    depth_cm: float | None = None


def components(image):
    red,green,blue = image.astype(np.int32).transpose(2,0,1)
    warm = (red>=80) & (red-green>=31) & (red*100>=green*195) & (red-blue>=37)
    # Separate scarlet, blushed and amber regions before grouping so adjacent
    # fruit of different hues do not collapse into a single silhouette.
    amber = (green*100>=red*28) & (green-blue>=18)
    blush = (blue*100>=green*128) & (blue*100>=red*19)
    bands = (warm&amber, warm&blush&~amber, warm&~(amber|blush))
    out = []
    for hue,band in enumerate(bands):
        labels, _ = label(binary_closing(band,CONNECTIVITY,border_value=0) | band, CONNECTIVITY)
        for component, bounds in enumerate(find_objects(labels),1):
            candidate = summarize(labels,bounds,component,hue)
            if candidate is not None:
                out.append(candidate)
    kept=[]
    for item in sorted(out,key=lambda c: -c.area_px):
        if not any(box_contains(item.box,prior.box) for prior in kept):
            kept.append(item)
    return kept


def box_contains(small,large):
    x0,y0,x1,y1=small
    a,b,c,d=large
    overlap=max(0,min(x1,c)-max(x0,a))*max(0,min(y1,d)-max(y0,b))
    area=(x1-x0)*(y1-y0)
    return overlap>=.66*area and area<.70*(c-a)*(d-b)


def summarize(labels,bounds,component,hue):
    y,x = bounds
    w,h = x.stop-x.start,y.stop-y.start
    if w<4 or h<3:
        return None
    mask = labels[bounds]==component
    row,col = np.nonzero(mask)
    area = len(row)
    if area<14 or area/(w*h)<.38:
        return None
    u,v = x.start+float(col.mean()),y.start+float(row.mean())
    bearing = np.rad2deg(np.arctan2(u+.5-WIDTH/2,FOCAL_PX))
    elevation = np.rad2deg(np.arctan2(HEIGHT/2-v-.5,np.hypot(FOCAL_PX,u+.5-WIDTH/2)))
    return Candidate((u,v),(x.start,y.start,x.stop,y.stop),area,float(bearing),float(elevation),
                     x.start<=1 or y.start<=1 or x.stop>=WIDTH-1 or y.stop>=HEIGHT-1,hue)


def measure_stereo(left, right):
    if not left or not right:
        return left
    lpos,rpos = np.array([c.pixel for c in left]),np.array([c.pixel for c in right])
    disparity = lpos[:,None,0]-rpos[None,:,0]
    epipolar = abs(lpos[:,None,1]-rpos[None,:,1])
    area_cost = abs(np.log(np.array([c.area_px for c in left])[:,None]/[c.area_px for c in right]))
    same_hue = np.array([c.hue_band for c in left])[:,None]==[c.hue_band for c in right]
    valid = same_hue&(disparity>=.65)&(disparity<=45.)&(epipolar<=2.2)&(area_cost<.35)
    cost = np.where(valid, 2*epipolar + 3*area_cost + .08*disparity, 1e3)
    indices, partners = linear_sum_assignment(cost)
    matches = dict(zip(indices,partners))
    return [
        replace(item,depth_cm=float(FOCAL_PX*BASELINE_CM/disparity[i,matches[i]]))
        if i in matches and valid[i,matches[i]] and not(item.clipped or right[matches[i]].clipped)
        else item
        for i,item in enumerate(left)
    ]


def lower_field_edges(image):
    """Local contrast cue in six lower-field sectors. No depth or traversability label."""
    luma = image.astype(float) @ np.array([.299,.587,.114])
    lower = luma[HEIGHT//2:]
    edge = np.hypot(sobel(lower,0),sobel(lower,1))/8.
    return tuple(float(np.mean(tile>=13.)) for tile in np.array_split(edge,6,axis=1))


class OrchardEyes:
    """Initialize on the canonical ground pose to calibrate the head-local axes."""
    def __init__(self, phy, *, exclude_walker=True):
        self.phy = phy
        self.cameras = [phy.model.name2id('orchard_eye_'+side,'camera') for side in ('left','right')]
        self.head = phy.model.name2id('walker/head','body')
        # The stock visual mesh uses rotated head-local axes. Align the
        # optical rig in the neutral standing pose, then follow head motion.
        neutral = phy.data.xmat[self.head].reshape(3,3)
        self.local_forward = neutral.T @ np.array([1.,0.,.025])
        self.local_up = neutral.T @ np.array([0.,0.,1.])
        self.options = wrapper.MjvOption()
        if exclude_walker:
            self.options.geomgroup[1:]=0
        self.options.sitegroup[:]=0

    def pose(self, origin, forward, up):
        forward = np.asarray(forward)/np.linalg.norm(forward)
        right = np.cross(forward,up)
        right /= np.linalg.norm(right)
        up = np.cross(right,forward)
        rot = np.column_stack((right,up,-forward)).ravel()
        for sign,camera in zip((-1,1),self.cameras,strict=True):
            self.phy.data.cam_xpos[camera] = origin + sign*(BASELINE_CM/2)*right
            self.phy.data.cam_xmat[camera] = rot

    def head_pose(self):
        pos = self.phy.data.xpos[self.head]
        rot = self.phy.data.xmat[self.head].reshape(3,3)
        forward = rot @ self.local_forward
        return pos+.06*forward,forward,rot @ self.local_up

    def render(self, *, segmentation=False):
        return [self.phy.render(width=WIDTH,height=HEIGHT,camera_id=cam,scene_option=self.options,
                                segmentation=segmentation) for cam in self.cameras]

    def sample(self, pose=None):
        self.pose(*(self.head_pose() if pose is None else pose))
        left,right = self.render()
        items = measure_stereo(components(left),components(right))
        return left,right,items,lower_field_edges(left)
