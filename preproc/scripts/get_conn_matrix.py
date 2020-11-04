
import sys

import numpy as np
import pipelineloader as pl


HIPPOCAMPUS_CONN_PERC = 98


def save_conn_matrix(sid, atlas, out_file):

    subj = pl.Subject(sid, atlas)
    w = subj.weights

    # Remove self connections
    w = w - np.diag(np.diag(w))

    # Scale by region volumes
    w = np.divide(w, subj.volumes[:, None], out=np.zeros_like(w), where=(w != 0))

    # Add hippocampus self-connections
    if atlas == 'vep':
        nreg = w.shape[0]
        inds_rhip = [i for i in range(nreg) if 'Right-Hippocampus' in subj.region_names[i]]
        inds_lhip = [i for i in range(nreg) if 'Left-Hippocampus'  in subj.region_names[i]]

        perc = np.percentile(w[~np.eye(nreg, dtype=bool)], HIPPOCAMPUS_CONN_PERC)
        if len(inds_rhip) == 2:
            w[inds_rhip[0], inds_rhip[1]] += perc
            w[inds_rhip[1], inds_rhip[0]] += perc
        if len(inds_lhip) == 2:
            w[inds_lhip[0], inds_lhip[1]] += perc
            w[inds_lhip[1], inds_lhip[0]] += perc

    # Normalize
    w /= np.max(np.sum(w, axis=1))

    np.savetxt(out_file, w, fmt="%.10e")



if __name__ == "__main__":
    sid = int(sys.argv[1])
    atlas = sys.argv[2]
    out_file = sys.argv[3]
    save_conn_matrix(sid, atlas, out_file)
