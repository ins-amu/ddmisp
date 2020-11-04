
import json
import glob
import os
from collections import OrderedDict
import re

import numpy as np
import pandas as pd
import scipy.stats as stats

import util.io as io


CHAINS = [1, 2]
C_EZ = 2.0

def split_path(path):
    path = os.path.normpath(path)
    return path.split(os.sep)


def get_tfit(ttru, tinf, temp, weights, t_lim, sig_t):
    issz = ttru < t_lim

    tt = np.maximum(ttru, 0.) if issz else t_lim

    pszinf = np.average(tinf < t_lim)
    pszemu = np.average(temp < t_lim)
    pszemw = 0 if np.sum(weights) == 0 else np.average(temp < t_lim, weights=weights)

    res = OrderedDict(
        pszinf=pszinf,
        pszemu=pszemu,
        pszemw=pszemw,
        pstatinf=(pszinf if issz else 1. - pszinf),
        pstatemu=(pszemu if issz else 1. - pszemu),
        pstatemw=(pszemw if issz else 1. - pszemw),
        # pdinf=np.average(stats.norm(tt, sig_t).pdf(np.clip(tinf, 0., t_lim))),
        # pdemu=np.average(stats.norm(tt, sig_t).pdf(np.clip(temp, 0., t_lim))),
        # pdemw=(0 if np.sum(weights) == 0
        #          else np.average(stats.norm(tt, sig_t).pdf(np.clip(temp, 0., t_lim)), weights=weights)),
    )

    for t in [5, 10, 20]:
        if ttru >= t_lim - t:
            res[f"pt{t:02d}inf"] = np.nan
            res[f"pt{t:02d}emu"] = np.nan
            res[f"pt{t:02d}emw"] = np.nan
        else:
            res[f"pt{t:02d}inf"] = np.average(np.abs(ttru - tinf) <  t)
            res[f"pt{t:02d}emu"] = np.average(np.abs(ttru - temp) <  t)
            res[f"pt{t:02d}emw"] = (0.0 if   np.sum(weights) == 0
                                        else np.average(np.abs(ttru - temp) <  t, weights=weights))

    return res



def calc_fragility(dirname, rec, tinf, t_lim):
    try:
        dirname_all = os.path.join(*split_path(dirname)[:-1], f"r{rec:02d}_all")
        resfiles_all = [os.path.join(dirname_all, f"chain_{chain}.csv") for chain in CHAINS]
        statuses_all = [os.path.join(dirname_all, f"chain_{chain}.status") for chain in CHAINS]
        tinf_all = io.parse_csv([r for r, s in zip(resfiles_all, statuses_all)
                                 if int(open(s).read().strip())])['t']
        fragility = np.mean(np.abs(np.mean(tinf < t_lim, axis=0) - np.mean(tinf_all < t_lim, axis=0)))
    except KeyError:
        print(f"Fragility calculation: incomplete data for {dirname_all}")
        fragility = np.nan

    return fragility



def aggreg_loo(summary_files, pickle_file):
    rows = []

    for i, summary_file in enumerate(summary_files):
        print(f"{i} / {len(summary_files)}")

        dirname = os.path.dirname(summary_file)
        _, _, onset, atlas, subject, _, loocase = split_path(dirname)

        # Process only LOO cases
        match = re.match("^r(\d{2})_wo(\d{3})$", loocase)
        if match is None:
            continue
        rec = int(match.group(1))
        reg = int(match.group(2))

        resfiles = [os.path.join(dirname, f"chain_{chain}.csv") for chain in CHAINS]
        statuses = [os.path.join(dirname, f"chain_{chain}.status") for chain in CHAINS]
        good_resfiles = [r for r, s in zip(resfiles, statuses) if int(open(s).read().strip())]
        if len(good_resfiles) == 0:
            continue

        res = io.parse_csv(good_resfiles)
        _, summary = io.parse_summary_csv(summary_file)
        checks = [np.mean(summary['c'].N_Eff) > 10, np.mean(summary['c'].R_hat) < 1.5]
        converged = all(checks)

        nsamples, nreg = res['c'].shape

        indata = io.rload(os.path.join(dirname, f"../../input/{loocase}.R"))
        t_lim = indata['t_lim']
        sig_t = indata['sig_t']
        seizing = indata['tleftout'] < t_lim
        ttru = indata['tleftout']

        nns = int(indata['n_ns']) + (0 if seizing else 1)
        nsz = int(indata['n_sz']) + (1 if seizing else 0)
        nobs = nsz + nns

        first_seizing = all(indata['tleftout'] < np.array(indata['t_sz']))
        t_first = np.min(indata['t_sz'])

        # Empirical estimates
        observed_t    = np.concatenate([indata['t_sz'], np.repeat(np.inf, int(indata['n_ns']))])
        observed_regs = np.concatenate([np.array(indata['reg_sz'], dtype=int),
                                        np.array(indata['reg_ns'], dtype=int)])
        w = np.array(indata['w'])
        weights = w[reg, observed_regs] + w[observed_regs, reg]

        # Inferred estimates
        tinf = res['t']
        cinf = res['c']

        row = OrderedDict(
            onset=onset,
            subject=subject,
            rid=rec,
            region=reg,
            observed=False,
            seizing=seizing,
            ttru=ttru,

            nobs=nobs,
            nsz=nsz,
            nns=nns,
            fracsz=nsz/nobs,

            rhat=summary['c'].R_hat[reg, 0],
            neff=summary['c'].N_Eff[reg, 0],

            pez=np.mean(cinf[:, reg] > C_EZ),
            cinf=np.mean(cinf[:, reg]),

            # fragility=calc_fragility(dirname, rec, tinf, t_lim),

            firstseizing=first_seizing,
            pfirst=np.mean(np.argmin(tinf, axis=1) == reg),
        )

        tfit = get_tfit(ttru, tinf[:, reg], observed_t, weights, t_lim, sig_t)
        row.update(tfit)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_pickle(pickle_file)


def aggreg_synth_solo(summary_files, series, pickle_file):
    rows = []

    for i, summary_file in enumerate(summary_files):
        dirname = os.path.dirname(summary_file)

        resfiles = [os.path.join(dirname, f"chain_{chain}.csv") for chain in CHAINS]
        statuses = [os.path.join(dirname, f"chain_{chain}.status") for chain in CHAINS]
        res = io.parse_csv([r for r, s in zip(resfiles, statuses) if int(open(s).read().strip())])
        _, summary = io.parse_summary_csv(summary_file)

        with open(os.path.join(dirname, "../groundTruth.json")) as fh:
            ground_truth = json.load(fh)
        indata = io.rload(os.path.join(dirname, "../input/input.R"))

        nsamples, nreg = res['c'].shape

        ctru = ground_truth['c']
        cinf = res['c']
        # cfit = np.mean(stats.norm(ctru, 1).pdf(cinf), axis=0)
        cfit = np.mean(np.abs(ctru - cinf) < 0.5, axis=0)
        c0 = np.random.normal(0, 1, size=10000)
        # cfit0 = np.mean(stats.norm(ctru, 1).pdf(c0[:, None]), axis=0)  # TODO: analytically
        cfit0 = np.mean(np.abs(ctru - c0[:, None]) < 0.5, axis=0)

        sig_t = indata['sig_t']
        t_lim = indata['t_lim']
        t_first = np.min(indata['t_sz'])
        tinf = res['t']
        ttru = np.array(ground_truth['t']) + indata['t_shift']
        nsz = np.sum(ttru < t_lim)
        nns = nreg - nsz

        zscore = np.abs((np.mean(cinf, axis=0) - ctru) / np.std(cinf, axis=0))
        shrinkage = 1 - np.std(cinf, axis=0)**2 / 1.

        observed_t_all    = np.concatenate([indata['t_sz'], np.repeat(np.inf, int(indata['n_ns']))])
        observed_regs_all = np.concatenate([np.array(indata['reg_sz'], dtype=int),
                                            np.array(indata['reg_ns'], dtype=int)])
        w = np.array(indata['w'])

        for reg in range(nreg):
            observed_t    = np.array([t for r, t in zip(observed_regs_all, observed_t_all) if r != reg])
            observed_regs = np.array([r for r in observed_regs_all if r != reg])
            weights = w[reg, observed_regs] + w[observed_regs, reg]
            seizing = ttru[reg] < t_lim
            
            _, _, _, _, config, sim, _ = split_path(dirname)
            sim_id = int(sim[1:])
            if series == "random":
                match = re.match("^f(.+)_nobs(\d+)$", config)
                function = match.group(1)
                nobs = match.group(2)
                row = OrderedDict(function=function, nobs=nobs)
            elif series == "ez":
                row = OrderedDict(config=config)

            row.update(OrderedDict(
                sim=sim_id,
                region=reg,
                observed=(reg in ground_truth['reg_obs']),
                seizing=seizing,
                ttru=ttru[reg],

                nsz=nsz,
                nns=nns,
                fracsz=nsz/nreg,

                rhat=summary['c'].R_hat[reg, 0],
                neff=summary['c'].N_Eff[reg, 0],

                pez=np.mean(cinf[:, reg] > C_EZ),
                cinf=np.mean(cinf[:, reg]),
                ctru=ctru[reg],
                cfit=cfit[reg],
                cfit0=cfit0[reg],

                zscore=zscore[reg],
                shrinkage=shrinkage[reg],
            ))

            tfit = get_tfit(ttru[reg], tinf[:, reg], observed_t, weights, t_lim, sig_t)
            row.update(tfit)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_pickle(pickle_file)
