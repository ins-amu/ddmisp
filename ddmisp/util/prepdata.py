
import os
import json
from collections import Counter

import numpy as np
import scipy.stats as stats

from . import io
from .simprop import prop


MAX_SZ_PER_SUBJECT = 2
T_FIRST = 30.0
T_LIM = 90.0
SIG_T = 5.0


def move_late_seizing_regs(reg_ns, reg_sz, t_sz, t_lim):
    inds_late = np.where(t_sz >= t_lim)[0]
    reg_ns = np.append(reg_ns, reg_sz[inds_late])
    reg_sz = np.delete(reg_sz, inds_late)
    t_sz   = np.delete(t_sz,   inds_late)
    reg_ns.sort()

    return reg_ns, reg_sz, t_sz


def pad_zeros(x, n):
    xnew = np.zeros(n, dtype=x.dtype)
    xnew[:len(x)] = x
    return xnew


def prep_learn(conn_files, reg_onset_files, atlas, stan_file):
    """
    Create a .R data file `stan_file` for cmdstan for multi-seizure learning step.

    TODO: rewrite using `Observation` class and `prep_group_file`
    """

    nreg = 0   # to be overwritten
    ws = []
    n_ns = []
    n_sz = []
    regs_ns = []
    regs_sz = []
    ts_sz = []
    sids = []
    rids = []
    t_shifts = []

    for conn_file, reg_onset_file in zip(conn_files, reg_onset_files):
        # Load connectome
        conn = np.genfromtxt(conn_file)
        nreg = conn.shape[0]

        # Load region onsets for all seizures
        with open(reg_onset_file) as fh:
            reg_onsets = json.load(fh)

        # Randomly select seizures
        inds_seizing = [i for i, rec in enumerate(reg_onsets) if len(rec['regions_seizing']) > 0]
        inds = np.random.choice(inds_seizing, size=min(len(inds_seizing), MAX_SZ_PER_SUBJECT), replace=False)
        for i in inds:
            rec = reg_onsets[i]

            reg_ns = np.array(rec['regions_nonseizing'], dtype=int)
            reg_sz = np.array(rec['regions_seizing'], dtype=int)
            t_sz = np.array(rec['onset_times'])

            t_shift = -np.min(t_sz) + T_FIRST
            t_shifts.append(t_shift)
            reg_ns, reg_sz, t_sz = move_late_seizing_regs(reg_ns, reg_sz, t_sz + t_shift, T_LIM)

            ws.append(conn)
            n_ns.append(len(reg_ns))
            n_sz.append(len(reg_sz))
            regs_ns.append(pad_zeros(reg_ns, nreg))
            regs_sz.append(pad_zeros(reg_sz, nreg))
            ts_sz.append(pad_zeros(t_sz, nreg))

            sids.append(rec['sid'])
            rids.append(rec['rid'])

    stan_data = {
        'n_seizures': len(ws),
        'nreg': nreg,
        'w': np.array(ws),
        'n_sz': np.array(n_sz), 'reg_sz': np.array(regs_sz), 't_sz': np.array(ts_sz),
        'n_ns': np.array(n_ns), 'reg_ns': np.array(regs_ns),
        't_lim': T_LIM, 'sig_t': SIG_T,
        'sids': np.array(sids, dtype=int), 'rids': np.array(rids, dtype=int),
        't_shifts': np.array(t_shifts)
    }
    io.rdump(stan_file, stan_data)


def prep_solo_file(rec, conn, q, leftout, out_file, shift=True):
    t_sz = np.array(rec['onset_times'])
    reg_ns = np.array(rec['regions_nonseizing'], dtype=int)
    reg_sz = np.array(rec['regions_seizing'], dtype=int)

    tleftout = None
    if (leftout is not None) and (leftout in list(reg_ns)):
        i = list(reg_ns).index(leftout)
        reg_ns = np.delete(reg_ns, i)
    elif (leftout is not None) and (leftout in list(reg_sz)):
        i = list(reg_sz).index(leftout)
        tleftout = t_sz[i]
        reg_sz = np.delete(reg_sz, i)
        t_sz = np.delete(t_sz, i)

    if len(reg_sz) == 0:
        return

    t_shift = 0.0
    if shift:
        t_shift = -np.min(t_sz) + T_FIRST
        reg_ns, reg_sz, t_sz = move_late_seizing_regs(reg_ns, reg_sz, t_sz + t_shift, T_LIM)

    tleftout = tleftout + t_shift if tleftout is not None else T_LIM + 1

    stan_data = {
        'nreg': conn.shape[0],
        'w': np.array(conn),
        'q11': q[0], 'q12': q[1], 'qa21': q[2], 'qa22': q[3],
        'n_ns': len(reg_ns), 'reg_ns': reg_ns,
        'n_sz': len(reg_sz), 'reg_sz': reg_sz, 't_sz': t_sz,
        't_lim': T_LIM, 'sig_t': SIG_T,
        'sid': rec['sid'], 'rid': rec['rid'], 't_shift': t_shift,
        'leftout': leftout if leftout is not None else -1,
        'tleftout': tleftout
    }

    io.rdump(out_file, stan_data)



def prep_solo(conn_file, reg_onset_file, res_files, out_dir, create_loo_files=False):
    os.makedirs(out_dir)

    res = io.parse_csv(res_files, merge=True)
    q = [np.mean(res[p]) for p in "q11 q12 qa21 qa22".split(" ")]
    conn = np.genfromtxt(conn_file)

    with open(reg_onset_file, 'r') as fl:
        reg_onsets = json.load(fl)

    for rec in reg_onsets:
        rid = rec['rid']

        # Leave-out files
        if create_loo_files:
            for reg in list(rec['regions_seizing']) + list(rec['regions_nonseizing']):
                prep_solo_file(rec, conn, q, leftout=reg, out_file=os.path.join(out_dir, f"r{rid:02d}_wo{reg:03d}.R"))

        # Solo file with nothing left out
        prep_solo_file(rec, conn, q, leftout=None, out_file=os.path.join(out_dir, f"r{rid:02d}_all.R"))


def select_observed_regions(c, t, nobs):
    nreg = len(c)

    szmask = t < T_LIM

    # Select one seizing
    reg_obs_sz = [np.random.choice(np.r_[:nreg][szmask])]

    # Add the rest randomly
    reg_obs_ot = np.random.choice(list(set(np.r_[:nreg]) - set(reg_obs_sz)), size=nobs-1,
                                  replace=False)

    reg_obs = np.sort(np.append(reg_obs_sz, reg_obs_ot))
    return reg_obs


def prep_synth_solo_random(conn_files, q, nobs, sim_id, out_file, ground_truth_file):
    np.random.seed(sim_id)
    q = np.array(q)

    conn_file = np.random.choice(conn_files)
    sid = int(os.path.split(conn_file)[1][2:5])
    conn = np.genfromtxt(conn_file)
    nreg = conn.shape[0]

    nsz = 0
    while nsz == 0:
        c = np.random.normal(0, 1, size=nreg)
        t = prop(c, conn, q)
        szmask = t < T_LIM
        nsz = sum(szmask)

    reg_obs = select_observed_regions(c, t, nobs)

    # Seizing regions will be adjusted later anyway
    reg_sz = [reg for reg in reg_obs if t[reg] <  T_LIM]
    reg_ns = [reg for reg in reg_obs if t[reg] >= T_LIM]

    with open(ground_truth_file, 'w') as fh:
        json.dump({
            'q': q.tolist(),
            'c': c.tolist(),
            't': t.tolist(),
            'reg_obs': reg_obs.tolist()
        },
        fh, indent=4)

    rec = {'onset_times': [t[reg] for reg in reg_sz],
           'regions_nonseizing': reg_ns,
           'regions_seizing':    reg_sz,
           'sid': sid, 'rid': 0}

    prep_solo_file(rec, conn, q, leftout=None, out_file=out_file, shift=False)


def prep_synth_solo_ez(conn_files, onset_files, q, config, sim_id, out_file, ground_truth_file):
    C_EZ = 2.0
    N_EZ = 2
    W_STRONG = 0.02418  # 97-percentile of all connections in all normalized connectomes
    N_STRONG = 3        # Required number of strong connections

    np.random.seed(sim_id)
    q = np.array(q)

    # Select the connectome and get the observed regions
    reg_obs = None
    while reg_obs is None:
        i = np.random.choice(len(conn_files))
        conn_file = conn_files[i]
        onset_file = onset_files[i]
        with open(onset_file) as fh:
            onsets = json.load(fh)
        if len(onsets) > 0:
            reg_obs = onsets[0]['regions_seizing'] + onsets[0]['regions_nonseizing']
            reg_obs = np.array(reg_obs)

    sid = int(os.path.split(conn_file)[1][2:5])
    conn = np.genfromtxt(conn_file)
    nreg = conn.shape[0]
    reg_hid = [r for r in np.r_[:nreg] if r not in reg_obs]

    nszobs = 0
    while nszobs == 0:
        c_ez = stats.truncnorm.rvs(C_EZ, np.inf, size=N_EZ)
        c_not_ez = stats.truncnorm.rvs(-np.inf, C_EZ, size=nreg-N_EZ)

        c = np.zeros(nreg)
        if config == "observed":
            reg_ez = np.random.choice(reg_obs, size=N_EZ, replace=False)
        elif config == "hidden-random":
            reg_ez = np.random.choice(reg_hid, size=N_EZ, replace=False)
        elif config == "hidden-nearmiss":
            w_strong = W_STRONG
            reg_connected = []
            while len(reg_connected) < N_EZ:
                reg_connected = np.where(np.sum(conn[reg_obs, :] >= w_strong, axis=0) > N_STRONG)[0] 
                reg_connected = [reg for reg in reg_connected if reg not in reg_obs] 
                w_strong = 0.9 * w_strong
            reg_ez = np.random.choice(reg_connected, size=N_EZ, replace=False)

        reg_not_ez = [r for r in np.r_[:nreg] if r not in reg_ez]
        c[reg_ez] = c_ez
        c[reg_not_ez] = c_not_ez

        t = prop(c, conn, q)
        szmask = t < T_LIM
        nszobs = sum(szmask[reg_obs])

    # Seizing regions will be adjusted later anyway
    reg_sz = [reg for reg in reg_obs if t[reg] <  T_LIM]
    reg_ns = [reg for reg in reg_obs if t[reg] >= T_LIM]

    with open(ground_truth_file, 'w') as fh:
        json.dump({
            'q': q.tolist(),
            'c': c.tolist(),
            't': t.tolist(),
            'reg_obs': reg_obs.tolist()
        },
        fh, indent=4)

    rec = {'onset_times': [t[reg] for reg in reg_sz],
           'regions_nonseizing': reg_ns,
           'regions_seizing':    reg_sz,
           'sid': sid, 'rid': 0}

    prep_solo_file(rec, conn, q, leftout=None, out_file=out_file, shift=False)


class Observation():
    def __init__(self, sid, rid, regions_nonseizing, regions_seizing, onset_times):
        assert len(regions_seizing) == len(onset_times)

        self.sid = sid
        self.rid = rid
        self.regions_nonseizing = regions_nonseizing
        self.regions_seizing = regions_seizing
        self.onset_times = onset_times


def prep_group_file(observations, conns, out_file, shift):
    """
    Create .R data file for cmdstan
    """

    nreg = conns[0].shape[0]
    nseizures = len(observations)

    n_ns = np.zeros(nseizures, dtype=int)
    n_sz = np.zeros(nseizures, dtype=int)
    regs_ns = np.zeros((nseizures, nreg), dtype=int)
    regs_sz = np.zeros((nseizures, nreg), dtype=int)
    ts_sz = np.zeros((nseizures, nreg), dtype=float)
    sids = np.zeros(nseizures, dtype=int)
    rids = np.zeros(nseizures, dtype=int)
    t_shifts = np.zeros(nseizures, dtype=float)

    for i, observation in enumerate(observations):
        reg_ns = observation.regions_nonseizing
        reg_sz = observation.regions_seizing
        t_sz = observation.onset_times

        t_shift = 0.0
        if shift:
            t_shift = -np.min(t_sz) + T_FIRST
            reg_ns, reg_sz, t_sz = move_late_seizing_regs(reg_ns, reg_sz, t_sz + t_shift, T_LIM)

        n_ns[i] = len(reg_ns)
        n_sz[i] = len(reg_sz)
        regs_ns[i, :len(reg_ns)] = reg_ns
        regs_sz[i, :len(reg_sz)] = reg_sz
        ts_sz[i, :len(t_sz)] = t_sz
        sids[i] = observation.sid
        rids[i] = observation.rid
        t_shifts[i] = t_shift


    stan_data = {
        'n_seizures': len(observations),
        'nreg': nreg,
        'w': np.array(conns),
        'n_sz': n_sz, 'reg_sz': regs_sz, 't_sz': ts_sz,
        'n_ns': n_ns, 'reg_ns': regs_ns,
        't_lim': T_LIM, 'sig_t': SIG_T,
        'sids': sids, 'rids': rids, 't_shifts': t_shifts
    }
    io.rdump(out_file, stan_data)



def prep_synth_group(conn_files, q, nobs, nseizures, out_file, ground_truth_file):
    np.random.seed(42)

    q = np.array(q)
    nreg = np.genfromtxt(conn_files[0]).shape[0]

    conns = []
    observations = []
    ground_truth = {'q': q.tolist(), 'seizures': []}

    for i in range(nseizures):
        conn_file = np.random.choice(conn_files)
        conn = np.genfromtxt(conn_file)
        sid = int(os.path.split(conn_file)[1][2:5])
        rid = i

        nsz = 0
        while nsz == 0:
            c = np.random.normal(0, 1, size=nreg)
            t = prop(c, conn, q)
            szmask = t < T_LIM
            nsz = sum(szmask)

        reg_obs = select_observed_regions(c, t, nobs)
        reg_sz = [reg for reg in reg_obs if t[reg] <  T_LIM]
        reg_ns = [reg for reg in reg_obs if t[reg] >= T_LIM]

        observation = Observation(sid, rid, reg_ns, reg_sz, [t[reg] for reg in reg_sz])

        observations.append(observation)
        conns.append(conn)
        ground_truth['seizures'].append({
            'sid': sid,
            'rid': rid,
            'q': q.tolist(),
            'c': c.tolist(),
            't': t.tolist(),
            'reg_obs': reg_obs.tolist()
        })

    prep_group_file(observations, conns, out_file, shift=False)
    with open(ground_truth_file, 'w') as fh:
        json.dump(ground_truth, fh, indent=4)
