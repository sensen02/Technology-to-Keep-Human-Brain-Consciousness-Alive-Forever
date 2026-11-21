from pathlib import Path
import os,sys
import numpy as np
from optics import capture,decode
for arg in sys.argv[1:]:
    dest=Path(arg);old=dest.with_name(dest.name.replace('aligned-','teacher-'))
    if (old/'cameras.zip').exists() and (old/'camera_cal.npz').exists():
        # Only reuse optical exposures for a bitwise identical geometry and pose trajectory.
        from hashlib import file_digest
        def digest(x):
            with x.open('rb') as f:return file_digest(f,'sha256').digest()
        identical=digest(old/'scene.mjb')==digest(dest/'scene.mjb')
        with np.load(old/'BODY.npz') as a,np.load(dest/'BODY.npz') as b:
            identical &= np.array_equal(a['poses'],b['poses'])
    else:identical=False
    if identical:
        for name in ('cameras.zip','camera_cal.npz'):
            if not (dest/name).exists():os.link(old/name,dest/name)
    else:capture(dest)
    decode(dest)
