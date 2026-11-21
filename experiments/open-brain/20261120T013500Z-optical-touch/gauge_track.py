"""Reuse the published color detector on a calibration fixture; no reference inputs."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

source,images,out=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location('marker_tracker',source/'track_markers.py')
tracker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)
info=json.loads((source/'input.json').read_text())
cams=list(info['projection'])
projections=np.asarray(list(info['projection'].values()))
colors=np.asarray(list(info['colors'].values()))
xy=np.array([tracker.detect(Image.open(images/('0_'+cam+'.png')),colors) for cam in cams])
upper=np.asarray([cam.startswith('H') for cam in cams])
pose,count=tracker.recover(projections,xy)
pose_upper,_=tracker.recover(projections[upper],xy[upper])
np.savez_compressed(out,frames=[0], names=list(info['colors']), views_used=count[None],
                    **{'8_views':pose[None], '4_upper':pose_upper[None]})
