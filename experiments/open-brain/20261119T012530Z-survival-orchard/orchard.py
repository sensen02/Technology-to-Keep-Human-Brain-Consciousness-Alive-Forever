"""Static survival arena for the existing P1 FlyBody, in centimetres."""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent / '20261118T072349Z-flight-descending'), str(ROOT)]
from engine.embodied.feeding import EMPTY, FeedingBody, Food
from engine.embodied.natural_scene import generate_fbm_heightfield

SEED = 821
CENTER_X = 8.
HALF_X, HALF_Y = 14., 12.
BED = -.07
RELIEF = .23
SPAWN_YAW = 0.
FRICTION = (1.02, .005, .0001)


@dataclass(frozen=True)
class Resource:
    name: str
    support: str
    x: float
    y: float
    top: float
    radius: float
    volume: float

    @property
    def geom(self):
        return 'food_' + self.name

    @property
    def odor_origin_cm(self):
        return self.x, self.y, self.top


RESOURCES = (
    Resource('ground_near', 'ground', 4.25, -.82, .023, .60, .90),
    Resource('ground_far', 'ground', 7.25, .38, .023, .56, 1.10),
    Resource('branch_outer', 'tree', 8.45, 3.85, .68, .67, 1.30),
    Resource('branch_inner', 'tree', 11.68, 7.55, 1.08, .73, 1.55),
    Resource('table_left', 'table', 12.35, -3.58, 1.23, .56, 1.20),
    Resource('table_middle', 'table', 13.57, -5.23, 1.33, .69, 1.45),
    Resource('table_right', 'table', 15.36, -4.04, 1.25, .58, 1.15),
)


def smoothstep(value):
    t = np.clip(value, 0, 1)
    return t*t*(3-2*t)


def ground_height():
    elevation, _ = generate_fbm_heightfield(257, extent_mm=140., seed=SEED,
                                             amplitude_std_mm=.25, micro_relief_mm=.045)
    y, x = np.mgrid[-HALF_Y:HALF_Y:257j, -HALF_X:HALF_X:257j]
    x += CENTER_X
    height = elevation * .18 + .025
    height += .065 * np.exp(-((x-5.8)**2/6. + (y-2.1)**2/1.8))
    height -= .032 * np.exp(-((x-9.6)**2/15. + (y+2.2)**2/.65))
    for px, py, radius in [(0., 0., .52), *((r.x, r.y, r.radius+.20) for r in RESOURCES if r.support == 'ground')]:
        alpha = smoothstep((np.hypot(x-px, y-py)-radius)/.70)
        height = alpha*height + (1-alpha)*(-BED)
    fade = smoothstep(np.minimum(HALF_X-abs(x-CENTER_X), HALF_Y-abs(y))/.65)
    return (height.clip(0, RELIEF) * fade).astype(np.float32)


def prepare_assets():
    height = ground_height()
    Image.fromarray(np.uint16(np.round(height[::-1]/height.max()*65535))).save(ASSETS/'relief.png')
    y, x = np.mgrid[-1:1:256j, -1:1:256j]
    # Values on edges are sampled from the same direction on adjacent faces.
    directions = {
        'right': (np.ones_like(x), -y, -x),
        'left': (-np.ones_like(x), -y, x),
        'up': (x, np.ones_like(x), y),
        'down': (x, -np.ones_like(x), -y),
        'front': (x, -y, np.ones_like(x)),
        'back': (-x, -y, -np.ones_like(x)),
    }
    rng = np.random.default_rng(SEED)
    for face, direction in directions.items():
        a, h, b = np.array(direction) / np.linalg.norm(direction, axis=0)
        top = np.maximum(h, 0.)
        tint = np.array([.70, .83, .89]) + top[..., None]**.48 * np.array([-.40, -.25, -.12])
        cloud = (np.sin(8*a + 4*b + 2*h) + .57*np.cos(15*b - 6*a + 3*h)
                 + .31*np.sin(27*a + 20*b) + .16*np.cos(45*b - 9*a))
        cloud = smoothstep((cloud-.25)/1.20) * smoothstep((h+.015)/.22) * .86
        sun = np.maximum(0., -.40*a + .72*h + .56*b)**34
        rgb = tint*(1-cloud[..., None]) + np.array([.96,.96,.91])*cloud[..., None]
        rgb += sun[..., None]*np.array([.22,.18,.10])
        rgb = np.where(h[..., None] >= 0, rgb, rgb + h[..., None]*np.array([.43,.43,.48]))
        Image.fromarray(np.uint8(np.clip(rgb,0,1)*255)).save(ASSETS/f'sky_{face}.png')

    def grain(name, low, high, long_axis, width):
        noise = gaussian_filter(rng.normal(size=(256,256)), long_axis, mode='wrap')
        noise /= noise.std()
        veins = np.sin(width*x + 1.3*np.sin(5*y) + .9*noise)
        mix = np.clip(.44 + .24*veins + .12*noise, 0., 1.)
        rgb = np.array(low) + mix[..., None]*(np.array(high)-np.array(low))
        Image.fromarray(np.uint8(np.clip(rgb,0,255))).save(ASSETS/f'{name}.png')

    grain('bark', (39, 29, 19), (105, 76, 48), (10, .65), 84)
    grain('wood', (118, 85, 51), (184, 149, 95), (.65, 10), 48)


def add_assets(arena):
    asset = arena.asset
    for name, repeat, specular in [('leaf_litter', (4,4), .05),
                                    ('wood', (2,1), .09), ('bark', (1,2), .05)]:
        tex = asset.add('texture', name=name, type='2d', file=str(ASSETS/f'{name}.png'))
        asset.add('material', name=name, texture=tex, texrepeat=repeat, specular=specular,
                  shininess=.12)
    sky = asset.find('texture','skybox')
    if sky is not None:
        sky.remove()
    asset.add('texture', name='orchard_sky', type='skybox',
              **{f'file{face}': str(ASSETS/f'sky_{face}.png')
                 for face in ('right','left','up','down','front','back')})
    arena.visual.headlight.ambient = (.43,.43,.42)
    arena.visual.headlight.diffuse = (.48,.46,.43)
    arena.visual.headlight.specular = (.14,.14,.13)
    arena.visual.map.znear = .001
    arena.visual.map.zfar = 40
    arena.visual.quality.shadowsize = 2048
    getattr(arena.visual,'global').offwidth = 1440
    getattr(arena.visual,'global').offheight = 960
    arena.worldbody.add('light', name='orchard_sun', directional=True, castshadow=True,
                                pos=(-12,-9,25), dir=(.38,.33,-1), diffuse=(.61,.58,.50),
                                specular=(.18,.16,.13))


def solid(world, name, kind, **kwargs):
    return world.add('geom', name=name, type=kind, group=0, contype=1,
                     conaffinity=1, friction=kwargs.pop('friction', FRICTION),
                     solref=(.0012,1.),solimp=(.90,.97,.001,.5,2.),**kwargs)


def detail(world, name, kind, **kwargs):
    return world.add('geom', name=name, type=kind, group=0, contype=0, conaffinity=0, **kwargs)


def branch(world, name, start, end, radius):
    solid(world, name, 'capsule', fromto=(*start,*end), size=(radius,),
          material='bark', rgba=(.51,.38,.25,1.))


def add_tree(world, rng):
    # Connected hard structure and seeded clusters of light decorative foliage.
    base = (10.75,5.45,-.04)
    fork = (10.78,5.49,1.16)
    crown = (10.93,5.61,2.93)
    branch(world, 'tree_trunk', base, fork, .49)
    branch(world, 'tree_crown', fork, crown, .32)
    boughs = [
        (fork, (9.51,4.63,1.44), .20),
        ((9.51,4.63,1.44), (8.17,3.48,1.36), .13),
        ((10.86,5.49,1.79), (11.80,6.99,1.86), .20),
        ((11.80,6.99,1.86), (12.67,7.63,2.32), .13),
        ((10.91,5.57,2.13), (8.96,5.36,2.67), .14),
        (crown, (10.30,4.20,3.38), .13),
        (crown, (12.50,5.47,3.19), .14),
        ((10.93,5.61,2.72), (10.18,7.17,3.01), .12),
        ((9.51,4.63,1.44), (8.91,4.45,2.27), .09),
    ]
    for i, (start,end,radius) in enumerate(boughs):
        branch(world, f'tree_bough_{i}', start, end, radius)
        for j in range(24):
            spread = np.array((.68,.63,.39)) * (1 if i<4 else 1.3)
            pos = np.array(end) + rng.normal(0,.55,3)*spread
            size = rng.uniform(.12,.34)
            green = rng.uniform(.80,1.22)
            if j%3==0:
                detail(world,f'petiole_{i}_{j}','capsule',fromto=(*end,*pos),size=(.018,),
                       rgba=(.34,.27,.12,1))
            detail(world,f'leaf_{i}_{j}', 'ellipsoid', pos=pos, size=(size,size*.43,.018),
                   euler=rng.uniform(-1.15,1.15,3), rgba=(.17*green,.41*green,.13*green,1))
    for r, anchor in zip((r for r in RESOURCES if r.support == 'tree'),
                         ((8.56,3.82,1.39),(11.97,7.29,1.97)), strict=True):
        branch(world, 'stem_'+r.name, anchor, (r.x,r.y+.15,r.top+.006), .036)


def add_table(world):
    z = .875
    # One continuous top supports contacts. Grooves and end grain are visual.
    solid(world,'table_top','box', pos=(13.90,-4.55,z-.11), size=(2.70,1.77,.11),
          material='wood', rgba=(.79,.67,.49,1))
    for x in (11.59,16.21):
        for y in (-5.96,-3.15):
            solid(world,f'table_leg_{x}_{abs(y)}','box', pos=(x,y,.35),
                  size=(.15,.15,.43), material='wood', rgba=(.59,.44,.30,1))
    for i,y in enumerate((-5.73,-5.14,-4.55,-3.96,-3.37)):
        detail(world,f'table_seam_{i}','box',pos=(13.90,y,z+.001),
               size=(2.67,.012,.002),rgba=(.28,.18,.10,1))
    for i,y in enumerate((-6.10,-3.00)):
        solid(world,f'table_apron_{i}','box',pos=(13.90,y,.59),size=(2.44,.075,.08),
              material='wood',rgba=(.59,.44,.30,1))


def berry_mesh(arena,r,thick):
    sectors=24
    angles=np.arange(sectors)*2*np.pi/sectors
    rings=((.52,0.),(.88,.18),(1.,.56),(.90,.85),(.72,1.))
    vertices=[(r.radius*size*np.cos(a),r.radius*size*np.sin(a),thick*(height-1))
              for size,height in rings for a in angles]
    vertices.extend(((0,0,-thick),(0,0,0)))
    faces=[]
    for i in range(sectors):
        j=(i+1)%sectors
        faces.append((len(vertices)-2,j,i))
        for k in range(len(rings)-1):
            a,b=k*sectors+i,k*sectors+j
            faces.extend(((a,b,b+sectors),(a,b+sectors,a+sectors)))
        end=(len(rings)-1)*sectors
        faces.append((end+i,end+j,len(vertices)-1))
    return arena.asset.add('mesh',name=r.geom,vertex=np.ravel(vertices),face=np.ravel(faces))


def add_food(arena):
    world=arena.worldbody
    for i,r in enumerate(RESOURCES):
        color=((.77,.093,.046,1),(.73,.11,.047,1),(.85,.105,.049,1))[i%3]
        color={'table_middle':(.69,.055,.21,1), 'table_right':(.87,.28,.039,1)}.get(r.name,color)
        if r.support=='ground':
            thick=.065
            solid(world,r.geom,'cylinder',pos=(r.x,r.y,r.top-thick/2),size=(r.radius,thick/2),
                  rgba=color,friction=(1.13,.005,.0001))
        else:
            thick=.50 if r.support=='tree' else r.top-.872
            solid(world,r.geom,'mesh',mesh=berry_mesh(arena,r,thick),pos=(r.x,r.y,r.top),
                  rgba=color,friction=(1.13,.005,.0001))
        for j in range(4):
            theta=j*1.42+i*.7
            detail(world,f'fruit_dimple_{i}_{j}','sphere',
                   pos=(r.x+.53*r.radius*np.cos(theta),r.y+.53*r.radius*np.sin(theta),r.top),
                   size=(.021,),rgba=(.96,.34,.072,1))
        if r.support=='table':
            for j in range(5):
                theta=j*2*np.pi/5
                detail(world,f'calyx_{r.name}_{j}','ellipsoid',
                       pos=(r.x-.28+.062*np.cos(theta),r.y+.20+.062*np.sin(theta),r.top+.012),
                       size=(.093,.026,.017),euler=(0,0,theta),rgba=(.24,.36,.08,1))


def build_orchard():
    """Return the uncompiled shared FlyBody arena and configured food bank."""
    from legged_fixture import build_env, set_ground_pose

    env = build_env()
    arena = env.task._arena.mjcf_model
    add_assets(arena)
    world = arena.worldbody
    floor = arena.find('geom','groundplane')
    floor.pos = (0,0,BED-.003)
    floor.size = (65,60,.25)
    arena.asset.add('material',name='leaf_litter_far',texture=arena.asset.find('texture','leaf_litter'),
                    texrepeat=(18,20),specular=.05,shininess=.12)
    floor.material = 'leaf_litter_far'
    floor.rgba = (.80,.83,.74,1)
    h = ground_height()
    hf = arena.asset.add('hfield', name='orchard_relief', file=str(ASSETS/'relief.png'),
                         size=(HALF_X,HALF_Y,float(h.max()),.05))
    solid(world,'orchard_terrain','hfield',hfield=hf,pos=(CENTER_X,0,BED),material='leaf_litter',
          rgba=(.86,.87,.76,1))
    rng = np.random.default_rng(SEED)
    add_tree(world,rng)
    add_table(world)
    add_food(arena)
    for i in range(62):
        x,y = rng.uniform(-4.2,22),rng.uniform(-10.9,10.9)
        if np.hypot(x,y)<2.2 or min(np.hypot(x-r.x,y-r.y) for r in RESOURCES)<1.15:
            continue
        ix, iy = np.round([(x-CENTER_X+HALF_X)/(2*HALF_X)*256, (y+HALF_Y)/(2*HALF_Y)*256]).astype(int)
        z=BED+h[iy,ix]
        size=rng.uniform(.045,.14)
        detail(world,f'litter_chip_{i}','ellipsoid',pos=(x,y,z+.013),
               size=(size,size*.63,.025),euler=(0,0,rng.uniform(0,2*np.pi)),rgba=(.28,.23,.15,1))
    for i in range(76):
        x,y=rng.uniform(-5.5,25),rng.uniform(-11.5,11.5)
        if x>21.6 or abs(y)>11.3 or (x<18 and abs(y)<8.5):
            continue
        span=rng.uniform(.19,.55)
        for j in range(6):
            theta=2*np.pi*j/6
            blade=np.array([.26*np.cos(theta),.26*np.sin(theta),rng.uniform(.46,.89)])*span
            bulb=np.array([x,y,.02])+blade*.75
            green=rng.uniform(.63,1.2)
            detail(world,f'grass_{i}_{j}','ellipsoid',pos=bulb,size=(.032,.025,span*.55),
                   euler=(.35*np.cos(theta),.40*np.sin(theta),theta),
                   rgba=(.26*green,.36*green,.12*green,1))
    for side in ('left','right'):
        world.add('camera',name='orchard_eye_'+side,fovy=56)
    bank = FoodBank(RESOURCES)
    task = env.task
    initialize = task.initialize_episode

    def start_grounded(physics, random_state):
        initialize(physics, random_state)
        set_ground_pose(env)
        task._ref_qpos[:,:3] = task._walker.get_pose(physics)[0]
        task._ref_qpos[:,3:7] = (1.,0.,0.,0.)
        task._ref_qvel[:] = 0.
        bank.reset()

    task.initialize_episode = start_grounded
    return env, bank


class FoodBank:
    """Transfer from the contacted patch into the same body's feeding state."""
    def __init__(self, resources):
        self.resources = resources
        self.reset()

    def reset(self):
        self.stocks = {r.name: Food(r.volume, .9*r.volume) for r in self.resources}

    def contact(self, mouth_cm, supported):
        if not supported:
            return None
        x,y,z = mouth_cm
        for r in self.resources:
            if (self.stocks[r.name].volume > 0 and -.002 <= z-r.top <= .14
                    and np.hypot(x-r.x,y-r.y) <= r.radius*(1. if r.support=='ground' else .72)-.035):
                return r.name
        return None

    def advance(self, engine: FeedingBody, state, dt_s, *, feed, contact, activity_use=0.):
        row = engine.advance(replace(state,patch=self.stocks.get(contact, EMPTY)),dt_s,
                             feed=feed,contact=contact is not None,activity_use=activity_use)
        if contact is not None:
            self.stocks[contact] = row.state.patch
        return replace(row,state=replace(row.state,patch=EMPTY))


def manifest():
    return dict(units='cm', seed=SEED, terrain_bounds_cm=[[-6,22],[-12,12]],
                heightfield_grid=[257,257], spawn_xy_cm=[0,0], spawn_yaw_rad=SPAWN_YAW,
                food=[dict(**asdict(r), geom=r.geom, odor_origin_cm=r.odor_origin_cm) for r in RESOURCES])


if __name__=='__main__':
    prepare_assets()
    (HERE/'SCENE.json').write_text(json.dumps(manifest(),indent=2)+'\n')
