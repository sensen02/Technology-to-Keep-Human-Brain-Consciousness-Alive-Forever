"""Exactsame720traininghistories/scales;newactualV12/V13validation only."""
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parents[1]/'landmark_neutral_v6/policy'
def main():
 src=(OLD/'generate_data.py').read_text().replace('actualRGB672=True','actualRGB728=True');ns={'__name__':'_opponent_actual_data','__file__':str(HERE/'generate_data.py')};exec(compile(src,str(HERE/'generate_data.py')+'::same720data','exec'),ns);ns['main']()
if __name__=='__main__':main()
