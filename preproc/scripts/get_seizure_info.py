
import sys
import json

import pipelineloader as pl


def is_generalized(note):
    if note in ["Partial seizure with secondary generalisation",
                "Type 2 partial seizure with secondary generalisation",
                "Partial with secondary generalisation",
                "Partial seizure with secondary genealisation"]:
        return True
    else:
        return False


out_file = sys.argv[1]

data = []
for sid in range(1, 51):
    subj = pl.Subject(sid)
    for rid, rec in enumerate(subj.seizure_recordings):
        duration = rec.termination - rec.onset
        generalized = is_generalized(rec.note)
        data.append({'sid': sid, 'rid': rid, 'onset': min(rec.onset, 60.), 'duration': duration, 'generalized': generalized})
 
with open(out_file, 'w') as fh:
    json.dump(data, fh)

