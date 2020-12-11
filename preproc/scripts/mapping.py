#!/usr/bin/env python3


import itertools
import json
import os
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

import pipelineloader as pl

from onsettimes import get_seizure_data_bip

class LabelVol():
    """Wrapper around nifti label volume"""

    def __init__(self, filename, dist_lim):
        label_vol = nib.load(filename)

        self.data = label_vol.get_data().astype(int) - 1
        self.affine = label_vol.affine
        self.set_coord_sequence(dist_lim)
        self.outside_index = -1
        self.dist_lim = dist_lim


    def set_coord_sequence(self, lim):
        ncellsx = int(lim/np.linalg.norm(self.affine[0:3, 0:3] @ np.array([1, 0, 0]))) + 1
        ncellsy = int(lim/np.linalg.norm(self.affine[0:3, 0:3] @ np.array([0, 1, 0]))) + 1
        ncellsz = int(lim/np.linalg.norm(self.affine[0:3, 0:3] @ np.array([0, 0, 1]))) + 1

        indx = np.arange(-ncellsx, ncellsx + 1, 1)
        indy = np.arange(-ncellsy, ncellsy + 1, 1)
        indz = np.arange(-ncellsz, ncellsz + 1, 1)

        coords = np.array(list(itertools.product(indx, indy, indz)), dtype=int)
        dists = np.array([np.linalg.norm(self.affine[0:3, 0:3] @ np.array(c)) for c in coords])
        mask = dists <= lim

        sdists, scoords = zip(*sorted(zip(dists[mask].tolist(), coords[mask, :].tolist())))
        self.dists = np.array(sdists)
        self.coords = np.array(scoords)


    def get_all_close_regions(self, point):
        regs = []
        dists = []

        pcoords = np.linalg.solve(self.affine, np.append(point, 1.0))[0:3].astype(int)
        for dist, coords_delta in zip(self.dists, self.coords):
            vcoords = pcoords + coords_delta
            if np.any([vcoords[i] < 0 or vcoords[i] >= self.data.shape[i] for i in range(3)]):
                continue

            reg = self.data[tuple(vcoords)]
            if (reg == self.outside_index) or (reg in regs):
                continue
            else:
                regs.append(reg)
                dists.append(dist)

        return list(zip(regs, dists))


def plot_regions(reg_onsets, reg_ns, reg_sz, t_sz, region_names, img_file, ez_hyp=None):
    if ez_hyp is None:
        ez_hyp = np.zeros(len(region_names), dtype=bool)

    nreg = len(region_names)
    try:
        mint = np.min([ro[-1] for ros in reg_onsets for ro in ros if ro[3] == 'sz']) - 20.0
        maxt = np.max([ro[-1] for ros in reg_onsets for ro in ros if ro[3] == 'sz'])
    except ValueError:
        # No seizing regions
        mint = 0.
        maxt = 60.

    plt.figure(figsize=(12, nreg/3))
    plt.subplot(1, 1, 1)
    for reg in range(nreg):
        if len(reg_onsets[reg]) == 0:
            continue
        plt.axhline(reg, color='gray', ls='--', zorder=-2)

        nsctr = 0
        for k, (ch, dist, distrat, state, tsz) in enumerate(reg_onsets[reg]):
            if state == 'ns':
                tsz = 1.05*maxt + ((maxt-mint)/30)*nsctr
                nsctr += 1

            plt.scatter([tsz], [reg], color='blue', alpha=0.3, clip_on=False, s=120)
            plt.text(tsz, reg, "%s (%.1f, %.1f)" % (ch, dist, distrat),
                     rotation=45, ha='left', va='bottom', fontsize=9, clip_on=False)

        for reg in reg_ns:
            plt.scatter([1.05*maxt], [reg], color='red', clip_on=False)
        for reg, t in zip(reg_sz, t_sz):
            plt.scatter([t], [reg], color='red')


    plt.xlim([0.95*mint, 1.02*maxt])
    labels = [name if not ez else "$\\bf{%s}$" % name.replace("_", "\_").replace("-", u"\u2010")
              for name, ez in zip(region_names, ez_hyp)]
    plt.yticks(np.r_[:nreg], labels)
    plt.subplots_adjust(right=0.7, bottom=0.01, top=0.99, left=0.2)
    plt.savefig(img_file)
    plt.close()


def plot_traces_regions(subj, rec, ch_sz, tch_sz, reg_onsets, img_file):
    time, data, t_onset, ch_names = get_seizure_data_bip(rec)
    nch = data.shape[0]
    plt.figure(figsize=(20, int(nch/2)))
    for i in range(nch):
        # Raw time series
        preictal_mask = time < t_onset
        scaling = 0.05/np.percentile(np.abs(data[i, preictal_mask]), 95)
        plt.plot(time, scaling*data[i, :] + nch - i - 1, 'b', lw=0.3, zorder=1)

    for ch, tch in zip(ch_sz, tch_sz):
        i = ch_names.index(ch)
        plt.scatter([tch], [nch-i-1], color='r', s=100, zorder=10)

    channels_reg = [[] for _ in range(nch)]
    for reg, chinfos in enumerate(reg_onsets):
        for ch, _, _, _, _ in chinfos:
            channels_reg[ch_names.index(ch)].append(subj.region_names[reg])
    labels = ["%s (%s)" % (ch, ",".join(regs)) for ch, regs in zip(ch_names, channels_reg)]

    plt.axvline(t_onset, color='gray', ls='--')
    plt.yticks(np.r_[:nch], reversed(labels))
    plt.ylim([-1, nch])
    plt.tight_layout()
    plt.savefig(img_file)
    plt.close()


def map_dist(subj, rec_id, labelvol, ch_ns, ch_sz, tch_sz, data_file=None, img_file=None, trace_img_file=None):
    nreg = len(subj.region_names)
    nns = len(ch_ns)

    # Map each contact to the closest region
    reg_onsets = [[] for _ in range(nreg)]
    for i, ch in enumerate(np.append(ch_ns, ch_sz)):
        closeregs = labelvol.get_all_close_regions(subj.contacts.get_coords(ch))
        if len(closeregs) == 0:
            continue
        elif len(closeregs) == 1:
            (r1, d1), d2 = closeregs[0], labelvol.dist_lim
        else:
            (r1, d1), (_, d2) = closeregs[:2]

        # TODO: Include upper limit?
        drat = d2/(d1 + 0.5)
        if drat > 2:
            if i < nns:
                reg_onsets[r1].append((ch, d1, drat, 'ns', 0.0))
            else:
                reg_onsets[r1].append((ch, d1, drat, 'sz', tch_sz[i - nns]))

    # Select the onset time to be used
    reg_ns = []
    reg_sz = []
    t_sz = []
    for reg in range(nreg):
        if len(reg_onsets[reg]) == 0:
            continue
        onset_times = [ro[4] if ro[3] == 'sz' else np.inf for ro in reg_onsets[reg]]
        onset_time = np.percentile(onset_times, 50, interpolation='lower')
        if np.isinf(onset_time):
            reg_ns.append(reg)
        else:
            reg_sz.append(reg)
            t_sz.append(onset_time)

        # imax = np.argmax([ro[2] for ro in reg_onsets[reg]])
        # if reg_onsets[reg][imax][3] == 'ns':
        #     reg_ns.append(reg)
        # else:
        #     reg_sz.append(reg)
        #     t_sz.append(reg_onsets[reg][imax][4])

    # Save data
    if data_file is not None:
        with open(data_file, 'w') as fl:
            json.dump({'regions_nonseizing': reg_ns, 'regions_seizing': reg_sz, 'onset_times': t_sz,
                       'regions_onsets': reg_onsets},
                      fl, indent=4)

    # Plot
    if img_file is not None:
        plot_regions(reg_onsets, reg_ns, reg_sz, t_sz, subj.region_names, img_file, subj.ez_hypothesis)

    if trace_img_file is not None:
        rec = subj.seizure_recordings[rec_id]
        rec.load()
        plot_traces_regions(subj, rec, ch_sz, tch_sz, reg_onsets, trace_img_file)
        rec.clear()

    return reg_ns, reg_sz, t_sz


def get_pos_vol(labels, affine):
    inds = np.moveaxis(np.indices(labels.shape), 0, -1)
    pos = np.zeros_like(inds, dtype=float)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            pos[i, j, :, :] = np.dot(
                affine,
                np.append(inds[i, j], np.ones(labels.shape[2])[:, None], axis=1).T
            )[0:3].T
    return pos


def map_channels_to_regions(sid, atlas, ch_onset_file, reg_onset_file):
    subj = pl.Subject(sid, atlas)
    subject_name = f"id{sid:03d}"
    labelvol = LabelVol(os.path.join(subj.direc, "dwi", "label_in_T1.%s.nii.gz" % atlas), 10.0)

    with open(ch_onset_file, 'r') as fl:
        recordings = json.load(fl)

    img_dir = os.path.join(os.path.split(reg_onset_file)[0], 'img')
    os.makedirs(img_dir, exist_ok=True)
    data_dir = os.path.join(os.path.split(reg_onset_file)[0], 'data')
    os.makedirs(data_dir, exist_ok=True)

    data = []
    for rec in recordings:
        rid = rec['rid']
        img_file = os.path.join(img_dir, f"{subject_name}_r{rid:02d}.png")
        trace_img_file = os.path.join(img_dir, f"traces_{subject_name}_r{rid:02d}.png")
        data_file = os.path.join(data_dir, f"{subject_name}_r{rid:02d}.json")

        reg_ns, reg_sz, t_sz = map_dist(subj, rid, labelvol, rec['channels_nonseizing'], rec['channels_seizing'],
                                        rec['onset_times'],
                                        data_file=data_file, img_file=img_file, trace_img_file=trace_img_file)

        data.append({
            'sid': sid,
            'rid': rid,
            'regions_nonseizing': reg_ns,
            'regions_seizing': reg_sz,
            'onset_times': t_sz
        })

    with open(reg_onset_file, 'w') as fl:
        json.dump(data, fl, indent=4)


if __name__ == "__main__":
    sid = int(sys.argv[1])
    atlas = sys.argv[2]
    ch_onset_file = sys.argv[3]
    reg_onset_file = sys.argv[4]
    map_channels_to_regions(sid, atlas, ch_onset_file, reg_onset_file)
