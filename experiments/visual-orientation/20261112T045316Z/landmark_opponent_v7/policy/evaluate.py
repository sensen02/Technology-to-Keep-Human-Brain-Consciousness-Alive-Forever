"""Final1000newactualV12/V13only,originalfunctionalRGBgatesunchanged."""
from adapt_source import build
_ns={'__name__':'_opponent_validation_metrics','__file__':__file__}
exec(compile(build('evaluate.py'),__file__+'::unchangedfunctionalmetrics','exec'),_ns)
metrics_flags=_ns['metrics_flags']
if __name__=='__main__':_ns['main']()
