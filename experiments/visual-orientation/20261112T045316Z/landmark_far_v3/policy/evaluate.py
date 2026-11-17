"""Once-only newV4/V5 far-domain final500 evaluation; old thresholds unchanged."""
from adapt_source import build
_ns={'__name__':'_far_validation_functions','__file__':__file__}
exec(compile(build('evaluate.py'),__file__+'::frozen_visual_metrics','exec'),_ns)
metrics_flags=_ns['metrics_flags']
if __name__=='__main__':_ns['main']()
