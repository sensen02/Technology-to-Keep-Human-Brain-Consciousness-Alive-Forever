"""Authorized framing correction, frozen predecessor read only."""
from pathlib import Path
import runpy
import sys
HERE=Path(__file__).resolve().parent
PREDECESSOR=HERE/'render_actual3000.py'
loaded=runpy.run_path(str(PREDECESSOR),run_name='read_only_predecessor')
source=loaded['source']
assert source.count('camera.distance=8.')==1
source=source.replace('camera.distance=8.','camera.distance=14.')
source=source.replace("camera.lookat[:]=z['com'][0]", "camera.lookat[:]=(z['com'].min(axis=0)+z['com'].max(axis=0))/2")
source=source.replace("paths += [Path(__file__).resolve(),HERE/'PREREG.json',REFERENCE,VERIFY,run/'runtime_geometry.npz']", "paths += [Path(__file__).resolve(),HERE/'WIDE_PREREG.json',REFERENCE,VERIFY,run/'runtime_geometry.npz',PREDECESSOR]")
source=source.replace("image=Image.new('RGB',(1040,480),(246,246,246));image.paste(Image.fromarray(renderer.render()),(0,0));draw=ImageDraw.Draw(image)", """pixels=renderer.render()
            renderer.enable_segmentation_rendering()
            segmentation=renderer.render()
            renderer.disable_segmentation_rendering()
            visible_mesh=np.isin(segmentation[:,:,0],np.flatnonzero(mesh)) & (segmentation[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))
            ys,xs=np.where(visible_mesh)
            assert len(xs)>0 and xs.min()>5 and xs.max()<634 and ys.min()>80 and ys.max()<434, 'fly silhouette clipped or covered by panel'
            image=Image.new('RGB',(1040,480),(246,246,246));image.paste(Image.fromarray(pixels),(0,0));draw=ImageDraw.Draw(image)""")
source=source.replace("fixed_world_camera=True,phase_counts=", "fixed_world_camera=True,all770_silhouettes_segmentation_margin_pass=True,camera_distance=14,phase_counts=")
namespace=dict(__name__='wide_render',__file__=str(Path(__file__).resolve()),REFERENCE=loaded['REFERENCE'],VERIFY=loaded['VERIFY'],digest=loaded['digest'],PREDECESSOR=PREDECESSOR)
exec(compile(source,str(Path(__file__).resolve())+' [adapted]','exec'),namespace)
if __name__=='__main__':namespace['main']()
