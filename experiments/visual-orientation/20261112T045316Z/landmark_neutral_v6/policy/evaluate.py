"""FinalnewactualV10/V11 validation, originalRMS/signgates unscaled."""
from adapt_source import build
_ns={'__name__':'_neutral_validation_metrics','__file__':__file__}
exec(compile(build('evaluate.py'),__file__+'::unchanged_original_gates','exec'),_ns)
metrics_flags=_ns['metrics_flags']
if __name__=='__main__':_ns['main']()
