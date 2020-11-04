#!/usr/bin/env python3

import json
import os
import itertools
import sys

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mne


import pipelineloader as pl

def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def smoothen(x, fs, window, axis=-1):
    """
    Smoothen over the last dimension.

    Based on http://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
    """
    window_len = round_up_to_odd(window * fs)
    wh = window_len // 2

    xm = np.moveaxis(x, axis, -1)

    if xm.shape[-1] < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    w = np.ones(window_len, 'd')
    xsm = np.zeros_like(xm)

    inds = itertools.product(*[range(_) for _ in xm.shape[:-1]])
    for ind in inds:
        xp = np.r_[np.repeat(xm[ind][0], wh), xm[ind], np.repeat(xm[ind][-1], wh)]
        xsm[ind] = np.convolve(w/w.sum(), xp, mode='valid')

    return np.moveaxis(xsm, -1, axis)


def remove_short_sequences(x, fs, lim, axis=-1):
    """
    t: array with times
    x: binary array
    lim: temporal limit
    """
    xm = np.moveaxis(x, axis, -1)
    xmnew = np.copy(xm)


    inds = itertools.product(*[range(_) for _ in xm.shape[:-1]])
    for ind in inds:
        xr = xm[ind]
        # changes = np.nonzero(xr[:-1] != xr[1:])[0]
        # changes = np.nonzero(xr != np.insert(xr[:-1], 0, False))[0]
        changes = np.nonzero(np.append(xr, False) != np.insert(xr, 0, False))[0]

        for fr, to in zip(changes[0::2], changes[1::2]):
            if (to - fr) / fs < lim:
                xmnew[ind][fr:to] = 0

    return np.moveaxis(xmnew, -1, axis)


def get_norm_log_power(time, data, t_onset):
    FINAL_FS = 32.
    FREQLIM = 12.4

    nch = data.shape[0]

    fs = 1./(time[1] - time[0])
    decim = round(float(fs/FINAL_FS))

    freqs = np.linspace(1., 100., 50)
    tfr = mne.time_frequency.tfr_array_multitaper(np.expand_dims(data[:, :], 0),
                                                  fs, freqs,
                                                  time_bandwidth=4.0, n_cycles=2*freqs,
                                                  decim=decim, zero_mean=False, output='avg_power', n_jobs=4)
    tdecim = time[::decim]
    tfrw = tfr * freqs[None, :, None]

    lpl = np.log10(np.sum(tfrw[:, freqs <= FREQLIM, :], axis=1))
    lpl -= np.mean(lpl[:, tdecim < t_onset], axis=1)[:, None]

    lph = np.log10(np.sum(tfrw[:, freqs > FREQLIM, :], axis=1))
    lph -= np.mean(lph[:, tdecim < t_onset], axis=1)[:, None]

    return tdecim, lpl, lph


def plot_onset_detection(img_file, method, ch_names, time, t_onset, data,
                         tlp, lpl, lph, logthreshold, szmask, onset_times):
    nch = len(ch_names)

    plt.figure(figsize=(20, nch))

    for i in range(nch):
        # Raw time series
        preictal_mask = time < t_onset
        scaling = 0.05/np.percentile(np.abs(data[i, preictal_mask]), 95)
        plt.plot(time, scaling*data[i, :] + nch - i - 1, 'b', lw=0.3, zorder=1)

        # band powers
        fac = 0.3
        plt.plot(tlp, fac*lpl[i] + nch - i - 1, color='g', lw=1.0, zorder=5)
        if method == 'ABN' or method == 'INC':
            plt.plot(tlp, fac*lph[i] + nch - i - 1, color='r', lw=1.0, zorder=5)

        plt.axhspan(nch-i-1 - fac*logthreshold, nch-i-1 + fac*logthreshold, color='0.75', zorder=-1)

        # onset time
        if szmask[i]:
            # plt.plot([onset_times[i], onset_times[i]], [nch-i-1-0.3, nch-i-1+0.3], color='k', zorder=10)
            plt.scatter([onset_times[i]], [nch-i-1-0.2], marker='^', color='k', zorder=10)


    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10.))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    plt.axvline(t_onset, color='gray', ls='--')
    plt.yticks(np.r_[:nch], reversed(ch_names))
    plt.ylim([-1, nch])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def get_onset_times(time, data, t_onset, ch_names, method):
    THR = 5.0
    SMOOTH_T = 20.0
    SHORT_SEQ_T = 20.0

    logthr = np.log10(THR)

    # Get log-power high, log-power low
    tlp, lpl, lph = get_norm_log_power(time, data, t_onset)
    fs = 1./(tlp[1] - tlp[0])

    # Allow for user-defined threshold
    if len(method) > 3 and method[3] == '_':
        thr = float(method[4:])
        logthr = np.log10(thr)
        method = method[:3]

    # Get seizure mask
    if method == 'ABN':
        # Increase or drop of power in low or high frequencies
        mask_raw = (np.abs(lpl) > logthr) | (np.abs(lph) > logthr)
    elif method == 'INC':
        # Increase of power in low or high frequencies
        mask_raw = (lpl > logthr) | (lph > logthr)
    elif method == 'LOW':
        # Increase of power in low frequencies only
        mask_raw = (lpl > logthr)
    else:
        raise ValueError("Unknown method '%s'.")

    # Only after clinically marked onset
    mask_raw = mask_raw * (tlp > t_onset)

    # Clean seizure mask
    mask_smooth = smoothen(mask_raw.astype(float), 1./(tlp[1] - tlp[0]), SMOOTH_T) > 0.5
    mask_final = remove_short_sequences(mask_smooth, fs, SHORT_SEQ_T)

    # Get onset time
    onset_inds = np.argmax(mask_final, axis=1)
    sz_ch_mask = onset_inds > 0
    onset_times = tlp[onset_inds]

    plot_artifacts = dict(method=method, ch_names=ch_names, time=time, t_onset=t_onset, data=data,
                          tlp=tlp, lpl=lpl, lph=lph, logthreshold=logthr,
                          szmask=sz_ch_mask, onset_times=onset_times)

    return sz_ch_mask, onset_times, plot_artifacts


def get_seizure_data_bip(rec):
    ts = max(rec.time[0], rec.onset - 60.)
    te = min(rec.time[-1], rec.termination + 0.)
    t_szo = rec.onset - ts

    tmask = (rec.time >= ts) * (rec.time <= te)

    time = rec.time[tmask]
    time -= time[0]
    data = rec.get_data_bipolar()[:, tmask]
    ch_names = rec.get_ch_names_bipolar()

    return time, data, t_szo, ch_names


def get_onset_times_ch(rec, onset_method, img_file=None):
    time, data, t_szo, ch_names = get_seizure_data_bip(rec)
    sz_mask, onset_times, plot_artifacts = get_onset_times(time, data, t_szo, ch_names, onset_method)
    if img_file is not None:
        plot_onset_detection(img_file, **plot_artifacts)

    chind_sz = np.nonzero(sz_mask)[0]
    chind_ns = np.nonzero(~sz_mask)[0]

    ch_ns = np.array(ch_names)[chind_ns]
    ch_sz = np.array(ch_names)[chind_sz]
    t_sz = onset_times[chind_sz]

    return ch_ns.tolist(), ch_sz.tolist(), t_sz.tolist()


def get_onset_on_channels(sid, onset_method, out_file, plot=False):
    SHORT_SEIZURE_LIM = 30.0

    subj = pl.Subject(sid)
    recs = []

    img_dir = os.path.join(os.path.split(out_file)[0], 'img')
    os.makedirs(img_dir, exist_ok=True)

    for rid, rec in enumerate(subj.seizure_recordings):
        print("Processing id%03d: %d" % (sid, rid))
        duration = rec.termination - rec.onset
        if duration < SHORT_SEIZURE_LIM:
            print("...skipping: duration=%.2f" % duration)
            continue

        rec.load()
        img_file = os.path.join(img_dir, 'id%03d_rec_%02d.png' % (sid, rid)) if plot else None
        ch_ns, ch_sz, t_sz = get_onset_times_ch(rec, onset_method, img_file=img_file)
        rec.clear()

        recs.append({
            'sid': sid,
            'rid': rid,
            'channels_nonseizing': ch_ns,
            'channels_seizing': ch_sz,
            'onset_times': t_sz
        })


    with open(out_file, 'w') as fl:
        json.dump(recs, fl, indent=4)


if __name__ == "__main__":
    sid = int(sys.argv[1])
    onset_method = sys.argv[2]
    out_file = sys.argv[3]
    plot = len(onset_method) == 3
    get_onset_on_channels(sid, onset_method, out_file, plot)
