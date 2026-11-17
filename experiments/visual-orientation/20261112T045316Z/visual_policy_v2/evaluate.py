"""Same frozenvisual thresholds, newV2/V3 only; exported scorer for independentaudit."""
from adapt_source import build,HERE
_namespace={'__name__':'_wallvision_validation_scorer','__file__':str(HERE/'evaluate.py')}
exec(compile(build('evaluate.py'),str(HERE/'evaluate.py')+'::unchanged_scoring','exec'),_namespace)
metrics_flags=_namespace['metrics_flags'];main=_namespace['main']
if __name__=='__main__':main()
