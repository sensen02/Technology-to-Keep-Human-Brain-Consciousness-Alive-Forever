"""Replace invalid mixed-unit figure with a visible withdrawal notice.
Historical numerical artifacts are retained, not silently relabelled as valid evidence.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'outputs/workspace_recheck'


def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.text(.03, .95, 'WITHDRAWN: previous static-support verdict',
            fontsize=20, color='firebrick', va='top', transform=ax.transAxes)
    lines = [
        'The previous chart does NOT establish mechanical infeasibility.',
        '',
        '1. qacc mixes linear and angular acceleration; its norm / gravity is not dimensionless.',
        '2. M @ qacc mixes force and torque; multiplying by mass does not fix this.',
        '3. A local search failing to find equilibrium is not an impossibility proof.',
        '4. The old keyframe branch did not actually reset to the model keyframe.',
        '5. Excluded muscles had zero activation, not the native control-floor steady state.',
        '6. Neural-control failure and the cause of falling have NOT been ruled out.',
        '',
        'Next: declared length-scale normalization, correct initialization, native floors,',
        'deterministic residuals, explicit stance gates and independent dynamic validation.',
        '',
        'Canonical XML unchanged. Hand-built research prototype; not a finished controller.',
    ]
    ax.text(.03, .82, '\n'.join(lines), fontsize=12, linespacing=1.6,
            va='top', transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(OUT / 'workspace_recheck.png', dpi=140)
    plt.close(fig)
    print(OUT / 'workspace_recheck.png')


if __name__ == '__main__':
    main()
