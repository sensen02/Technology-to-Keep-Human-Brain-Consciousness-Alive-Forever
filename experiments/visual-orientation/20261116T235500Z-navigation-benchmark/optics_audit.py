"""Check camera observability on matching archived flat-walk poses."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate import Eyes, SharedBrain, JOINT_WEIGHTS, SCENES, HERE, WIDTH, COLOR, mj, np, prepare, torch, sha
from far_preflight import read_green

ROOT=HERE/"OPTICS"
ROOT.mkdir()
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
ARC=HERE.parent/"20261116T170000Z/WALK_HELD_normal/TRACE.npz"
scenes=json.loads(SCENES.read_text())["scenes"]
torch.set_num_threads(1)
model=SharedBrain().eval()
model.load_state_dict(torch.load(JOINT_WEIGHTS,map_location="cpu",weights_only=True)["brain"])

with np.load(ARC,allow_pickle=False) as trace:
    ms=np.array([0,1000,2600,4000,6200,8450])
    poses=trace["qpos"][ms]
assert poses.ndim==2
cases=("flat_paired","narrow_trench","wide_trench","one_offset_box","staggered_boxes")
samples={}
for case in cases:
    scene=scenes[case]
    body,geometry=prepare(ROOT/case,scene)
    assert body.d.qpos.shape==poses[0].shape
    eyes=Eyes(body)
    target=eyes.place(scene["target_angle"],scene["target_distance"])
    backdrop=(eyes.pose()[0].copy(),.13,.27)
    images=[]
    try:
        for qpos in poses:
            body.d.qpos[:]=qpos
            mj.mj_forward(body.plant,body.d)
            images.append(eyes.image(target,WIDTH,COLOR,backdrop))
    finally:
        eyes.close()
    images=np.asarray(images)
    with torch.inference_mode():
        visual=[float(model.visual_feature(torch.from_numpy(frame[None])).item()) for frame in images]
    green=read_green(images).sum((1,2,3))
    samples[case]=dict(images=images,visual=visual,green=green.tolist(),geometry=geometry)
plain=samples["flat_paired"]["images"].astype(np.int16)
narrow=samples["narrow_trench"]["images"].astype(np.int16)
out={}
sheet=Image.new("RGB",(1250, len(cases)*157),"#eaf0f2")
font=ImageFont.truetype(FONT,16)
draw=ImageDraw.Draw(sheet)
labels=["同向平地","浅窄横坑","较宽横坑","偏置单箱","错列双箱"]
for j,(label,case) in enumerate(zip(labels,cases,strict=True)):
    s=samples[case]
    rgb=s["images"]
    difference=np.abs(rgb.astype(np.int16)-plain)
    out[case]=dict(
        changed_pixels_vs_narrow_by_at_least_12=(np.abs(rgb.astype(np.int16)-narrow).max(4)>=12).sum((1,2,3)).tolist(),
        mean_delta_rgb=difference.mean((1,2,3,4)).tolist(),
        changed_pixels_by_at_least_12=(difference.max(4)>=12).sum((1,2,3)).tolist(),
        frozen_walk_logit=s["visual"],green_detector_pixels=s["green"],geometry=s["geometry"])
    for i,(stamp,eye_pair) in enumerate(zip(ms,rgb,strict=True)):
        x=i*208
        y=j*157
        draw.text((x+5,y+4),f"{label}  {stamp/1000:.2f}s",font=font,fill="#253948")
        for eye,image in enumerate(eye_pair):
            sheet.paste(Image.fromarray(image).resize((96,96),Image.Resampling.NEAREST),(x+eye*99+5,y+34))
sheet.save(ROOT/"配对双眼原图.png")
(ROOT/"RESULT.json").write_text(json.dumps(dict(archived_pose_source=str(ARC),sha256=sha(ARC),
    diagnostic_only=True,synthetic_rollouts=0,stamps_ms=ms.tolist(),cases=out),indent=2)+"\n")
print(json.dumps({k:{p:v for p,v in row.items() if p!="geometry"} for k,row in out.items()}))
