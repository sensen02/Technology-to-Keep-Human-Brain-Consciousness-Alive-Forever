"""Explicit numerical-branch scheduling; no neural or body side effects."""
from dataclasses import dataclass
import math
from .closure_clock import grid_steps

@dataclass(frozen=True)
class RK4ClosureClock:
    body_dt_s: float = .000025
    neural_dt_s: float = .0005
    command_dt_s: float = .005

    def __post_init__(self):
        for value in (self.body_dt_s, self.neural_dt_s, self.command_dt_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('clocks must be finite and positive')
        grid_steps(self.neural_dt_s, self.body_dt_s)
        grid_steps(self.command_dt_s, self.neural_dt_s)
        grid_steps(self.command_dt_s, self.body_dt_s)

    @property
    def body_per_command(self):
        return grid_steps(self.command_dt_s, self.body_dt_s)

    @property
    def neural_per_command(self):
        return grid_steps(self.command_dt_s, self.neural_dt_s)

    def command_row(self, body_step):
        if isinstance(body_step, bool) or not isinstance(body_step, int) or body_step < 0:
            raise ValueError('body step must be a nonnegative Python integer')
        return body_step // self.body_per_command

    def validate_source(self, interval, source_interval, valid_from_s):
        if any(isinstance(i, bool) or not isinstance(i, int) or i < -1
               for i in (interval, source_interval)) or interval < 0:
            raise ValueError('invalid interval identity')
        # Initial zero command is the only source=-1; normal later commands must
        # originate in the immediately preceding completed neural interval.
        if source_interval != interval - 1:
            raise ValueError('not previous completed interval')
        expected = interval * self.command_dt_s
        if not math.isfinite(valid_from_s) or not math.isclose(
                valid_from_s, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError('command validity does not match interval start')
        return True

    def stage_times(self, body_step):
        self.command_row(body_step)
        t = body_step * self.body_dt_s
        return tuple(t + c*self.body_dt_s for c in (0., .5, .5, 1.))
