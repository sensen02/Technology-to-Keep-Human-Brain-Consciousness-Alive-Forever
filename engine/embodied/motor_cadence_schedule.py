"""Pure causal schedule for independent sensory and MN-to-muscle clocks.

An engineering schedule only: it does not create spikes, commands, or body
control. The caller must observe/hold sensory current at sensory boundaries,
step its real network at neural boundaries, and decode only completed spikes.
"""
from dataclasses import dataclass

from .closure_clock import grid_steps


@dataclass(frozen=True)
class MotorWindow:
    sensory_index: int
    start_tick: int
    end_tick: int
    apply_source_end_tick: int | None


def motor_windows(duration_s, *, sensory_dt_s=.005, neural_dt_s=.0005,
                  motor_dt_s=.005):
    """Return body application windows and their *previous* source endpoints.

    Window [start_tick,end_tick) holds a command decoded at
    apply_source_end_tick. At the end of this window the caller consumes real
    neural samples [start_tick,end_tick) and makes that output available only
    in the next window. The first window must apply a zero command.
    """
    sensory_ticks = grid_steps(sensory_dt_s, neural_dt_s, name='sensory interval')
    motor_ticks = grid_steps(motor_dt_s, neural_dt_s, name='motor interval')
    if sensory_ticks % motor_ticks:
        raise ValueError('motor interval must divide sensory hold interval')
    sensory_count = grid_steps(duration_s, sensory_dt_s)
    windows = []
    for sensory_index in range(sensory_count):
        block_start = sensory_index * sensory_ticks
        for offset in range(0, sensory_ticks, motor_ticks):
            start = block_start + offset
            windows.append(MotorWindow(sensory_index, start, start + motor_ticks,
                                       None if start == 0 else start))
    return tuple(windows)
