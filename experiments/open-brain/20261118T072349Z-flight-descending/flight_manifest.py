"""Shared names and evidence hashing used by the flight experiment."""
import hashlib
from pathlib import Path

ROOT = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/open-brain')
GROUPS = ('wing_amplitude', 'flight_steer', 'takeoff', 'landing')
LABELS = tuple(f'{group}/{side}' for group in GROUPS for side in ('left', 'right'))
CASES = ('sham', 'amp_low', 'amp_high', 'amp_cut',
         'left', 'right', 'left_cut', 'right_cut', 'combined')
COLUMNS = ('time_s', 'x_cm', 'y_cm', 'z_cm', 'yaw_rad', 'roll_rad', 'pitch_rad',
           'contacts', 'wing_left_rad', 'wing_right_rad',
           'target_left_rad', 'target_right_rad', 'vnc_amp_left', 'vnc_amp_right',
           'vnc_steer_left', 'vnc_steer_right', 'wing_force', 'external_applied_max',
           'reference_yaw_rad')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
