
import sys

import numpy as np
import pipelineloader as pl


atlas = sys.argv[1]
out_file = sys.argv[2]
subj = pl.Subject(1, atlas=atlas)
np.savetxt(out_file, subj.region_names, fmt="%s")
