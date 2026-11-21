import subprocess
from pathlib import Path
import mujoco

here=Path(__file__).resolve().parent
package=Path(mujoco.__file__).resolve().parent
library=next(package.glob('libmujoco.so.*'))
subprocess.run(['c++','-O3','-shared','-fPIC','-std=c++17','-I'+str(package/'include'),
               str(here/'measured_contacts.cpp'),str(library),'-Wl,-rpath,'+str(package),
               '-o',str(here/'measured_contacts.so')],check=True)
