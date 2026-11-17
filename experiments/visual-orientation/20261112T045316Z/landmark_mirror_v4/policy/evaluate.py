"""Onlyfinal500 and newactualV6/V7frames; digitalmirrors never count freshvalidation."""
from adapt_source import build
_ns={'__name__':'_mirror_validation_metrics','__file__':__file__}
exec(compile(build('evaluate.py'),__file__+'::unchangedgates','exec'),_ns)
metrics_flags=_ns['metrics_flags']
if __name__=='__main__':_ns['main']()
