import os
import json
import sys
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd

from . import io, simprop


def split_path(path):
    path = os.path.normpath(path)
    return path.split(os.sep)


def compare_pre_post(onset, atlas, subject, rid, resected):
    CHAINS = [1, 2]

    direc = f"run/solo/{onset}/{atlas}/{subject}"

    # Input data
    data = io.rload(f"{direc}/input/r{rid:02d}_all.R")
    wpre = data['w']
    q = [data['q11'], data['q12'], data['qa21'], data['qa22']]
    tlim = float(data['t_lim'])

    # Inference results
    resfiles = [f"{direc}/output/r{rid:02d}_all/chain_{ch}.csv" for ch in CHAINS]
    statuses = [f"{direc}/output/r{rid:02d}_all/chain_{ch}.status" for ch in CHAINS]
    res = io.parse_csv([r for r, s in zip(resfiles, statuses) if int(open(s).read().strip())])
    cinf = res['c']

    # Post-op connectivity matrix
    wpost = np.copy(wpre)
    for reg in resected:
        wpost[reg, :] = 0
        wpost[:, reg] = 0

    # Pre-op and post-op simulations
    nsamples, nreg = cinf.shape
    tpre  = np.zeros((nsamples, nreg))
    tpost = np.zeros((nsamples, nreg))
    for i in range(nsamples):
        tpre[i, :]  = simprop.prop(cinf[i], wpre,  q)
        tpost[i, :] = simprop.prop(cinf[i], wpost, q)
    tpost[:, resected] = np.inf

    nsz_pre  = np.sum(np.mean(tpre  < tlim, axis=0) > 0.5)
    nsz_post = np.sum(np.mean(tpost < tlim, axis=0) > 0.5)
    avg_psz_pre  = np.mean(np.mean(tpre  < tlim, axis=0))
    avg_psz_post = np.mean(np.mean(tpost < tlim, axis=0))

    return (nsz_pre, nsz_post, avg_psz_pre, avg_psz_post)




def virtual_resection(surgery_file, region_names_file, summary_files, outfile):
    RESECTION_THRESHOLD = 0.501

    region_names = list(np.genfromtxt(region_names_file, usecols=(0,), dtype=str))

    with open(surgery_file) as fh:
        surgeries = json.load(fh)['data']
        surgeries = {k[0:5]: v for k, v in surgeries.items()}

    rows = []
    for i, summary_file in enumerate(summary_files):
        print(f"{i} / {len(summary_files)} ... ", end="", flush=True)
        dirname = os.path.dirname(summary_file)
        _, _, onset, atlas, subject, _, loocase = split_path(dirname)
        rid = int(loocase[1:3])

        if subject not in surgeries:
            continue

        resection = surgeries[subject]['resection']
        resection_indices = [region_names.index(regname) for regname, fraction in resection.items()
                             if fraction > RESECTION_THRESHOLD]
        if len(resection_indices) == 0:
            resection_indices = [region_names.index(max(resection, key=resection.get))]  # Get max element

        nsz_pre, nsz_post, avg_psz_pre, avg_psz_post = compare_pre_post(
            onset, atlas, subject, rid, resection_indices)

        rows.append(OrderedDict(
            onset=onset,
            atlas=atlas,
            subject=subject,
            engel=surgeries[subject]['engel'],
            nresected=len(resection_indices),
            rid=rid,
            nsz_pre=nsz_pre,
            nsz_post=nsz_post,
            avg_psz_pre=avg_psz_pre,
            avg_psz_post=avg_psz_post
        ))

    df = pd.DataFrame(rows)
    df.to_pickle(outfile)
