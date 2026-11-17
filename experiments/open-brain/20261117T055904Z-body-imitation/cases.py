"""Predeclared whole-episode splits; times are native 0.1 ms ticks."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    start: int = 2300
    stop: int = 12200
    brake: int = 0
    cut_start: int = 0
    cut_end: int = 0
    push_side: int = 0
    pulse_every: int = 60


TRAIN = (
    Case('tr01', 17), Case('tr02', 29, 1900, 12600), Case('tr03', 31, 2600, 11980),
    Case('tr04', 43, 2100, 13100), Case('tr05', 39, 2390, 12200, 10500),
    Case('tr06', 65, 2670, 12400, 10330), Case('tr07', 70, 1800, 12900, 11210),
    Case('tr08', 41, 2350, 12300, push_side=1), Case('tr09', 24, 2180, 13100, push_side=-1),
    Case('tr10', 19, 2330, 12700, cut_start=0, cut_end=4900),
    Case('tr11', 26, 2260, 12950, pulse_every=65), Case('tr12', 38, 2130, 13100, pulse_every=55),
    Case('tr13', 76, 1900, 12900, cut_start=7500, cut_end=17000),
    Case('tr14', 78, 2150, 13200, cut_start=0, cut_end=17000),
)
DEVELOP = (Case('dev01', 59, 2500, 12100),
           Case('dev02', 53, 2050, 13000, 10800),
           Case('dev03', 94, 2200, 13200, cut_end=3900))
HELD = (Case('held_walk', 111, 2430, 12810), Case('held_brake', 123, 2250, 13090, 10560),
        Case('held_push', 126, 2590, 12940, push_side=-1),
        Case('held_cut', 111, 2430, 12810, cut_end=17000),
        Case('held_restore', 129, 2190, 13230, cut_end=5080))
CORRECT = (Case('dag01', 155, 2340, 12800), Case('dag02', 157, 2550, 13100, 10150),
           Case('dag03', 159, 2220, 12900, push_side=1), Case('dag04', 160, 2340, 12730, cut_end=4560),
           Case('dag05', 162, 2480, 13250, 10710, push_side=-1),
           Case('dag06', 164, 2310, 13380, cut_end=17000))
CASES = {c.name: c for c in TRAIN + DEVELOP + HELD + CORRECT}
