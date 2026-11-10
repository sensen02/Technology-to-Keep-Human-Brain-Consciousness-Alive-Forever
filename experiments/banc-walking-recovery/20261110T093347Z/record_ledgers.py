"""Record explicit old/new source identities for current controlled variants."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent.parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    variants=[('20261110T091009Z','20261110T074936Z','actor previous-action-only masking; teacher/physics unchanged'),
              ('20261110T091600Z','20261110T074936Z','expert active-swing timewarp and smooth contactforce transfer; physics unchanged'),
              ('20261110T093347Z','20261110T091600Z','bounded contact-height/time/stride configurations and typed verifier arguments')]
    for new,old,reason in variants:
        newdir,olddir=ROOT/new,ROOT/old
        entries=[]
        for filename in ['models.py','student_wave.py','physics.py','identity.py','wave_reference.py','static_teacher.py','PREREG.json','verify_smooth.py']:
            p=newdir/filename
            if not p.exists():continue
            previous=olddir/filename
            if filename=='verify_smooth.py' and not previous.exists():previous=olddir/'verify_static.py'
            entries.append(dict(old_file=str(previous) if previous.exists() else None,
                 old_sha256=digest(previous) if previous.exists() else None,new_file=str(p),new_sha256=digest(p),
                 changed=not previous.exists() or digest(previous)!=digest(p),reason=reason))
        out=newdir/'CHANGE_LEDGER.json'
        if out.exists():raise FileExistsError('refuse overwriting ledger')
        out.write_text(json.dumps(dict(parent=str(olddir),variant=str(newdir),entries=entries,
          source_policy='all recorded input code frozen afterrun; future changes require newdirectory'),indent=2)+'\n')
        print(str(out))

if __name__=='__main__':main()
