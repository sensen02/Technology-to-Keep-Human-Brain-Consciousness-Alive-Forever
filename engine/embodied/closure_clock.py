"""Strict integer-grid timing for the research neural/body closure."""
import math


def grid_steps(value, step, *, name='duration', allow_zero=False):
    if not math.isfinite(value) or not math.isfinite(step) or step <= 0 or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'{name} must be finite and positive (zero allowed: {allow_zero})')
    n=round(value/step)
    if not math.isclose(value,n*step,rel_tol=0,abs_tol=1e-12):
        raise ValueError(f'{name} must align with {step}s grid')
    return n


def body_clock_steps(body_dt, neural_dt, command_dt, duration):
    """Validate a body grid without changing neural/command clocks."""
    grid_steps(neural_dt, body_dt, name='neural interval')
    per_command = grid_steps(command_dt, body_dt, name='command interval')
    total = grid_steps(duration, body_dt)
    intervals = grid_steps(duration, command_dt)
    if total != intervals * per_command:
        raise ValueError('inconsistent body/command duration')
    return per_command, total


def push_steps(push, duration, body_dt):
    """Half-open interval in BODY steps, never quantized to command intervals."""
    if push is None:return None
    start=grid_steps(push['start_s'],body_dt,name='push start',allow_zero=True)
    end=grid_steps(push['end_s'],body_dt,name='push end')
    total=grid_steps(duration,body_dt)
    if end<=start or end>total:raise ValueError('push window must fit episode and have positive length')
    if len(push['force_model']) != 3 or not all(math.isfinite(v) for v in push['force_model']):
        raise ValueError('push force must be three finite components')
    return start,end
