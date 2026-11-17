"""Parent-only saved actual XML/body/RGB video, 0 integration/0 NN forwards."""
import argparse
import hashlib
import json
import os
import subprocess
import runtime as r

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-after-parent-release',action='store_true')
    args = parser.parse_args()
    if not args.execute_after_parent_release:
        print('CODE_ONLY no arrays/Core/render')
        return
    r.verify()
    import numpy as np
    import mujoco as mj
    from PIL import Image,ImageDraw,ImageFont
    folder = r.HERE/'trials'/r.CASE
    result = r.read(folder/'RESULT.json')
    audit = r.read(folder/'VERIFY.json')
    if not audit['integrity_all_pass'] or not all(audit['checks'].values()) or audit['result_sha256']!=r.sha(folder/'RESULT.json'):
        raise ValueError('Fresh336 audit required')
    def arrays(name):
        with np.load(folder/name,allow_pickle=False) as data:
            return {key:data[key].copy() for key in data.files}
    paths = list(folder.glob('*'))
    hashes = {str(path):r.sha(path) for path in paths if path.is_file()}
    t,d,g,terrain = [arrays(name) for name in ('trace_1ms.npz','neural_decisions.npz','geometry_1ms.npz','terrain_1ms.npz')]
    rgb = arrays('capture_rgb.npz')['rgb']
    caps = r.read(folder/'CAPTURES.json')
    ident = r.read(folder/'IDENTITY.json')
    if t['time'].shape!=(r.TOTAL_MS,) or d['time'].shape!=(r.DECISIONS,) or rgb.shape!=(r.FRAMES,2,32,32,3):
        raise ValueError('Full long window required')
    attempts = []
    def forbidden(*args,**kwargs):
        attempts.append(True)
        raise RuntimeError('Saved video forbids integration')
    mj.mj_step = mj.mj_step1 = mj.mj_step2 = forbidden
    namespace,state = r.context(audit=True)
    sw,identity = namespace['pin']()
    identity.verify(ident)
    c = sw.Core(adhesion=False,recorder=None)
    identity.verify_runtime(ident,c)
    if str(c.derived_xml)!=result['actual_xml'] or r.sha(c.derived_xml)!=result['actual_xml_sha256']:
        raise ValueError('Actual derived XML mismatch')
    model = c.plant
    mesh = model.geom_type==mj.mjtGeom.mjGEOM_MESH
    if mesh.sum()!=71 or not np.all(model.geom_rgba[mesh,3]==1):
        raise ValueError('Complete71opaque required')
    ledger = r.read(r.HERE/'PREPARED/MODEL_LEDGER.json')['cases'][r.CASE]
    blocks = [item for item in ledger['terrain'] if item['name'].startswith('terrain_block_')]
    ids = {item['name']:model.geom(item['name']).id for item in blocks}
    rear = max(item['footprint'][1] for item in blocks)
    # Bound full model geometry from actual saved poses, without altering XML.
    low,high = np.full(3,np.inf),np.full(3,-np.inf)
    final_body_min_x = None
    for i in list(range(0,r.TOTAL_MS,20))+[r.TOTAL_MS-1]:
        c.d.qpos[:] = t['qpos'][i]
        c.d.qvel[:] = t['qvel'][i]
        mj.mj_forward(model,c.d)
        points = c.d.geom_xpos[mesh]
        radius = model.geom_rbound[mesh,None]
        low = np.minimum(low,(points-radius).min(0))
        high = np.maximum(high,(points+radius).max(0))
        if i==r.TOTAL_MS-1:
            final_body_min_x = float((points[:,0]-radius[:,0]).min())
    look = (low+high)/2
    extent = float(np.linalg.norm(high-low)/2)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance = max(12.,extent/np.sin(np.deg2rad(float(model.vis.global_.fovy)/2))*1.8)
    cam.azimuth,cam.elevation = 95.,-22.
    detail = mj.MjvCamera()
    detail.type = mj.mjtCamera.mjCAMERA_FREE
    detail.azimuth,detail.elevation,detail.distance = 90.,-10.,2.0
    option = mj.MjvOption()
    option.geomgroup[:] = 1
    option.geomgroup[5] = 1
    option.flags[mj.mjtVisFlag.mjVIS_TENDON] = False
    out = r.HERE/'VIDEO'
    out.mkdir(exist_ok=False)
    video = out/'ACTUAL_LONG38P4_DERIVED_TERRAIN.mp4'
    oldbuffer = [int(model.vis.global_.offwidth),int(model.vis.global_.offheight)]
    model.vis.global_.offwidth,model.vis.global_.offheight = 960,640
    renderer,zoom,writer = None,None,None
    mapping,bounds = [],[]
    visibility = {name:dict(main_max_pixels=0,detail_max_pixels=0) for name in ids}
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    counts = result['native']['post_native_contacts_by_terrain']
    head = model.body('Head').id
    try:
        renderer = mj.Renderer(model,height=640,width=960)
        zoom = mj.Renderer(model,height=192,width=288)
        writer = subprocess.Popen(['ffmpeg','-v','error','-threads','1','-filter_threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','50','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        for frame,i in enumerate(range(0,r.TOTAL_MS,20)):
            now = float(t['time'][i])
            qi = min(r.DECISIONS-1,(i+1)//5)
            c.d.qpos[:],c.d.qvel[:],c.d.time = t['qpos'][i],t['qvel'][i],now
            mj.mj_forward(model,c.d)
            if not np.array_equal(c.d.qpos,t['qpos'][i]) or not np.array_equal(c.d.qvel,t['qvel'][i]) or not np.allclose(c.d.subtree_com[c.th],t['com'][i],rtol=0,atol=1e-10):
                raise ValueError('Saved pose/COM reconstruction')
            if not np.allclose(c.d.xpos[head],g['head_xyz'][i],rtol=0,atol=1e-10) or not np.allclose(c.d.xmat[head].reshape(3,3),g['head_rotation'][i],rtol=0,atol=1e-10):
                raise ValueError('Actual head reconstruction')
            rot = c.d.xmat[c.th].reshape(3,3)
            yaw = float(np.arctan2(rot[1,0],rot[0,0]))
            if abs(np.arctan2(np.sin(yaw-g['thorax_yaw'][i]),np.cos(yaw-g['thorax_yaw'][i])))>1e-10:
                raise ValueError('Actual yaw reconstruction')
            renderer.update_scene(c.d,camera=cam,scene_option=option)
            renderer.enable_segmentation_rendering()
            seg = renderer.render()
            renderer.disable_segmentation_rendering()
            mask = np.isin(seg[:,:,0],np.flatnonzero(mesh)) & (seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))
            yy,xx = np.where(mask)
            if len(xx)==0 or xx.min()<8 or xx.max()>951 or yy.min()<76 or yy.max()>563:
                raise ValueError('Fullbody camera containment failed; revise only visual framing in new render namespace')
            bounds.append([int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())])
            for name,gid in ids.items():
                visibility[name]['main_max_pixels'] = max(visibility[name]['main_max_pixels'],int(np.count_nonzero((seg[:,:,0]==gid)&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM)))))
            # Close-up follows nearest original block; camera-only zoom, no geometry scaling.
            nearest = min(blocks,key=lambda item:abs((item['footprint'][0]+item['footprint'][1])/2-float(t['com'][i,0])))
            detail.lookat[:] = [(nearest['footprint'][0]+nearest['footprint'][1])/2,0,.08]
            zoom.update_scene(c.d,camera=detail,scene_option=option)
            zoom.enable_segmentation_rendering()
            zs = zoom.render()
            zoom.disable_segmentation_rendering()
            for name,gid in ids.items():
                visibility[name]['detail_max_pixels'] = max(visibility[name]['detail_max_pixels'],int(np.count_nonzero((zs[:,:,0]==gid)&(zs[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM)))))
            image = Image.new('RGB',(1280,720),'white')
            image.paste(Image.fromarray(renderer.render()),(0,0))
            image.paste(Image.fromarray(zoom.render()),(976,310))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,960,72),fill='white')
            draw.rectangle((0,568,960,640),fill='white')
            phase = {0:'SETTLE',2:'ORIGINAL TEACHER WARMUP',3:'OLD336 AUTONOMOUS'}[int(t['control_source'][i])]
            lines = [f'{phase} | actual {now:.3f}/44.200s | 1x / 50fps',f'Autonomous window38.4s; exact old XML heights .02/.04; first front1.35',f'Actual source neck / old336 / motor789f | SI NOT_ESTABLISHED']
            for j,text in enumerate(lines):
                draw.text((10,3+j*22),text,font=font,fill='black')
            for j,text in enumerate([f'Full-window slip {result["terrain_summary"]["relative_five"]["slip_ratio_max"]:.6f} <=.5 required | rough PASS={result["terrain_summary"]["rough_terrain_pass"]}',f'All6 cycles={result["autonomous_cycles"]["all_legs_completed"]}; NFnormal={result["native"]["post_nonfoot_normal_peak_native"]}; external={result["native"]["extra_force_max"]}',f'Contacts are native solver events, not trials. No hook/pulvilli claim.']):
                draw.text((10,573+j*21),text,font=font,fill='black')
            valid = bool(d['image_valid'][qi])
            fid = int(d['image_capture_id'][qi]) if valid else None
            draw.text((974,10),'ACTUAL RELEASED ACTOR RGB',font=small,fill='black')
            if valid:
                if not 0<=fid<r.FRAMES or caps[fid]['release_time_s']>float(d['time'][qi])+1e-8:
                    raise ValueError('Released RGB association')
                for eye in range(2):
                    image.paste(Image.fromarray(rgb[fid,eye]).resize((128,128),Image.Resampling.NEAREST),(976+eye*144,40))
                draw.text((974,181),f'Capture {fid}, release {caps[fid]["release_time_s"]:.3f}s',font=small,fill='black')
            else:
                if int(d['image_capture_id'][qi])!=-1 or i+1>=20:
                    raise ValueError('Initial invalid RGB')
                draw.text((976,70),'NO RELEASED RGB',font=font,fill='black')
            draw.text((974,219),f'Vision u {float(d["visual_raw_turn"][qi]):+.5f}',font=small,fill='black')
            draw.text((974,240),f'COM relative {float(terrain["com_relative_height"][i]):.5f}',font=small,fill='black')
            draw.text((974,280),'LOW SIDE CLOSE-UP / REAL HEIGHT',font=small,fill='black')
            draw.text((974,509),nearest['name']+' / camera zoom only',font=small,fill='black')
            draw.text((974,535),'FINAL native contacts per block:',font=small,fill='black')
            for j,item in enumerate(blocks):
                n = counts.get(item['name'],0)
                draw.text((974,555+j*20),f'{item["name"][-2:]}: {n} '+('CONTACTED' if n else 'NOTTESTED'),font=small,fill='black')
            draw.text((12,661),'Actual long trial, full body + actual low-side terrain close-up. No physical height exaggeration.',font=font,fill='black')
            draw.text((12,690),'flat_box_control is intentionally flush top_z0: a measurement control, not a raised obstacle.',font=small,fill='black')
            pixels = np.asarray(image).tobytes()
            writer.stdin.write(pixels)
            mapping.append(dict(frame=frame,trace_index=i,time_s=now,decision_index=qi,actor_capture_id=fid,detail_block=nearest['name'],detail_lookat=detail.lookat.tolist(),qpos_sha256=hashlib.sha256(t['qpos'][i].tobytes()).hexdigest(),frame_rgb_sha256=hashlib.sha256(pixels).hexdigest()))
        writer.stdin.close()
        if writer.wait()!=0:
            raise ValueError('ffmpeg encode')
    finally:
        if writer is not None and writer.poll() is None:
            writer.stdin.close()
            writer.wait()
        if renderer is not None:
            renderer.close()
        if zoom is not None:
            zoom.close()
        model.vis.global_.offwidth,model.vis.global_.offheight = oldbuffer
    identity.verify_runtime(ident,c)
    r.verify()
    if attempts or any(r.sha(path)!=digest for path,digest in hashes.items()):
        raise ValueError('Integration/source drift')
    probe = json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)
    if int(probe['streams'][0]['nb_read_frames'])!=r.FRAMES or abs(float(probe['format']['duration'])-44.2)>.001:
        raise ValueError('Full44p2second2210frame video required')
    selected = [0,49,50,289,290,400,650,900,1100,r.FRAMES-1]
    montage = Image.new('RGB',(2560,3600))
    decoder = subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(video),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
    count = 0
    while True:
        data = decoder.stdout.read(1280*720*3)
        if not data:
            break
        if len(data)!=1280*720*3:
            raise ValueError('Partial decoded frame')
        if count in selected:
            j = selected.index(count)
            montage.paste(Image.frombytes('RGB',(1280,720),data),((j%2)*1280,(j//2)*720))
        count += 1
    decoder.stdout.close()
    if decoder.wait()!=0 or count!=r.FRAMES:
        raise ValueError('Complete decode required')
    montage.save(out/'DECODED_MONTAGE.png')
    all_contacted = all(counts.get(item['name'],0)>0 for item in blocks)
    traversal = bool(all_contacted and final_body_min_x>rear)
    r.save(out/'FRAME_MAPPING.json',mapping)
    r.save(out/'FFPROBE.json',probe)
    r.save(out/'VIDEO.json',dict(kind=r.KIND+'_ACTUAL_VIDEO',video=str(video),video_sha256=r.sha(video),frames=count,seconds=44.2,fps=50,Core_constructions=1,physics_steps=0,NN_forwards=0,actual_XML=result['actual_xml'],actual_XML_SHA=result['actual_xml_sha256'],main_camera=dict(lookat=look.tolist(),distance=cam.distance,azimuth=cam.azimuth,elevation=cam.elevation),all_frame_body_bounds=bounds,block_visible_pixels=visibility,all_blocks_have_rendered_pixels=all(max(v.values())>0 for v in visibility.values()),height_exaggeration=False,actual_released_RGB=True,all71opaque=True,full_body_final_min_x_conservative_spheres=final_body_min_x,last_block_rear_x=rear,all_blocks_actual_native_contact=all_contacted,whole_course_traversal_conservative_proof=traversal,failed_conservative_bound_means='NOT_ESTABLISHED, not proof traversal impossible',parent_actual_review='PENDING'))
    print(str(out/'VIDEO.json'))

if __name__=='__main__':
    main()
