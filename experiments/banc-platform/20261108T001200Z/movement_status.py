"""Movement-quality status as an EXPLICIT platform state.

Round 14 recorded the ANN movement layer's known deficiency only in a report field. This
module makes it a first-class platform state: every platform tick carries the movement
controller identity, its acceptance status, and the reason it is not accepted, so no later
analysis can quietly treat platform progress as accepted walking.

Nothing here measures or improves the controller; it only labels it.
"""
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class MovementStatus:
    controller: str
    accepted: bool
    support_gates: str
    slip_criterion: str
    source: str
    deficiencies: tuple = field(default_factory=tuple)

    def validate(self):
        if not self.controller:
            raise ValueError('controller identity is required')
        if self.accepted and self.deficiencies:
            raise ValueError('an accepted controller cannot carry declared deficiencies')
        if not isinstance(self.accepted, bool):
            raise ValueError('accepted must be a bool')

    def record(self):
        self.validate()
        return dict(controller=self.controller, accepted=self.accepted, support_gates=self.support_gates,
                    slip_criterion=self.slip_criterion, source=self.source, deficiencies=list(self.deficiencies))


ANN_U32 = MovementStatus(
    controller='replacement_ANN_not_connectome',
    accepted=False,
    support_gates='D1-D8 pass (frozen verifier)',
    slip_criterion='FAILS material foot slip criterion',
    source='/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T123634Z/evaluation/ann_residual_u32/',
    deficiencies=('material foot slip above the frozen threshold',
                  'no food sensing: feeding cannot feed back into locomotion',
                  'not connectome-derived and not evidence about the fly'),
)

CONNECTOME_ZERO = MovementStatus(
    controller='connectome_module_zero_readout',
    accepted=False,
    support_gates='not applicable: no displacement produced',
    slip_criterion='not applicable: no steps taken',
    source='/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-physical-learning/20261107T195435Z/',
    deficiencies=('zero forward displacement', 'rounds 10-13 established the module does not walk in this reconstruction'),
)
