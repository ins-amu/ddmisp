import sys
import os
import json
import glob
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, lines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import sklearn.metrics as sklm

sys.path.append("../")
from util import io, plot3d, plot

CHAINS = [1, 2]
EZ_THRESHOLD = 2.0
RESEC_THRESHOLD = 0.3
NREG = 162
SUBJECTS_DIR = "/home/vep/RetrospectivePatients/1-Processed/"

def get_data(surgeries, onset):
    rows = []
    for subject, surgery in surgeries.items():     
        res_dir = f"run/solo/{onset}/vep/{subject}/output/"
        rec_dirs = glob.glob(os.path.join(res_dir, "r*_all/"))

        for rec_dir in rec_dirs:
            rid = int(rec_dir[-7:-5])

            indata = io.rload(f"run/solo/{onset}/vep/{subject}/input/r{rid:02d}_all.R")
            reg_sz = list(indata['reg_sz'].astype(int))
            reg_ns = list(indata['reg_ns'].astype(int))
            reg_obs = reg_sz + reg_ns

            resfiles = [os.path.join(rec_dir, f"chain_{chain}.csv") for chain in CHAINS]
            statuses = [os.path.join(rec_dir, f"chain_{chain}.status") for chain in CHAINS]
            res = io.parse_csv([r for r, s in zip(resfiles, statuses)
                                if int(open(s).read().strip())])
            pez = np.mean(res['c'] > EZ_THRESHOLD, axis=0)

            resec = np.array(surgery['resection'])
            is_resected = np.array(resec) > RESEC_THRESHOLD
            if not np.any(is_resected):
                is_resected[np.argmax(resec)] = True

            for i in range(NREG):
                rows.append(OrderedDict(
                    onset=onset,
                    subject=subject,
                    rid=rid,
                    region=i,

                    observed=(i in reg_obs),
                    seizing=(i in reg_sz),
                    fracsz=(len(reg_sz) / len(reg_obs)),

                    pez=pez[i],
                    resection=resec[i],
                    is_resected=is_resected[i],
                    engel=surgery['engel']
                ))

    df = pd.DataFrame(rows)
    dfg = df.groupby(['onset', 'subject', 'region']).agg(
             {'observed': 'any', 'seizing': 'any', 'fracsz': 'first',
              'pez': 'mean',  'resection': 'first', 'is_resected': 'first',
              'engel': 'first'}).reset_index()
    
    return dfg
    
    

def plot_resection(outfile):
    VMAX_PEZ = 0.25
    
    with open("data/surgeries.vep.json") as fh:
        surgeries = json.load(fh)
    surgeries = {subj[:5]: data for subj, data in surgeries.items()}        
    
    df = get_data(surgeries, 'INC')
    
    # Patient specific data
    sid, rid = 17, 0
    pos, surfs, contacts = plot3d.read_structural_data(sid, SUBJECTS_DIR)
    w, obsmask = plot3d.read_input_data(f"run/solo/INC/vep/id{sid:03d}/input/r{rid:02d}_all.R")
    dfs = df[df.subject == f"id{sid:03d}"]
    nreg = w.shape[0]
    
    # Create 3D plots
    imgs = {}
    for method, scalar, vmax in [('inf', dfs.pez.to_numpy(), VMAX_PEZ),
                                 ('res', dfs.is_resected.to_numpy(), 1)]:
        for view in ['topdown', 'leftright']:
            imgs[method, view] = plot3d.viz_network_scalar(
                pos, scalar, w, obsmask, surfs, vmin=0, vmax=VMAX_PEZ, cmap='Reds', view=view)
    
    # Calculate the precision-recall curves
    subsets = {1: df.engel == 1, 2: df.engel == 2, 3: df.engel.isin([3, 4])}    
    res = {}
    for subset, mask in subsets.items():
        dff = df[mask]
        nsubjects = len(dff[['subject']].drop_duplicates())
        prec, recall, threshold = sklm.precision_recall_curve(dff.is_resected, dff.pez)
        npred = np.zeros_like(prec) 
        for i, thr in enumerate(threshold):
            npred[i] = np.sum(dff.pez >= thr)
        
        res[subset] = dict(prec=prec, recall=recall, threshold=threshold,
                           npred=npred, nsubjects=nsubjects,
                           nresected=dff.is_resected.sum(), n=len(dff))
    
    # Plot --------------------------------------------------------------------------------
    fig = plt.figure(figsize=(13, 7))

    gs = gridspec.GridSpec(1, 4, wspace=0.2, width_ratios=[0.25, 2, 0.05, 1.7],
                           left=0.04, right=0.98, top=0.9, bottom=0.07)
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, gs[0], wspace=0.3, hspace=0.1, height_ratios=[1, 0.04])
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, gs[1], wspace=0.05, hspace=0.2)
    gs3 = gridspec.GridSpecFromSubplotSpec(3, 2, gs[3], hspace=0.5, wspace=0.5)

    # Bar plots
    ax1 = plt.subplot(gs1[0, 0])
    im = plt.imshow(dfs.pez[:, None], aspect='auto', cmap='Reds', vmin=0, vmax=VMAX_PEZ,
                    origin='lower', extent=[0, 1, 0, nreg])
    plt.xlim(0, 1); plt.xticks([]);  plt.yticks([])
    plt.title("$p(c > c_h)$", fontsize=10, rotation='vertical', ha='center', va='bottom')
    plot.add_brain_regions(ax1, pad=5, width=5, coord='display', labels=True)

    ax1cb = plt.subplot(gs1[1, 0])
    cb = plt.colorbar(im, cax=ax1cb, ticks=[0, VMAX_PEZ], orientation='horizontal')
    ax1cb.set_xticklabels(["0", f"{VMAX_PEZ:.2f}"])

    ax2 = plt.subplot(gs1[0, 1])
    plt.imshow(dfs.is_resected[:, None], aspect='auto', cmap='Reds', vmin=0, vmax=1,
               origin='lower', extent=[0, 1, 0, nreg])
    plt.xlim(0, 1); plt.xticks([]);  plt.yticks([])
    plt.title("Resect.", fontsize=10, rotation='vertical', ha='center', va='bottom')

    plot.add_mask(ax1, dfs.observed, width=8, pad=0, coord='display')

    # 3D plots
    axes = []
    for i, (orientation, leg_pos) in enumerate([('topdown', 4), ('leftright', 3)]):
        ax3 = plt.subplot(gs2[0, i])
        if i == 0:
            # Fake data for colorbar must go first
            im = plt.imshow(np.array([[0, VMAX_PEZ]]), cmap="Reds").set_visible(False)   
            axins = inset_axes(ax3, width="4%", height="25%", loc=leg_pos, borderpad=0.6)
            plt.colorbar(im, cax=axins, orientation="vertical", ticks=[0, VMAX_PEZ])
            axins.yaxis.set_ticks_position('left' if leg_pos == 4 else 'right')
            plt.sca(ax3)
        plt.imshow(imgs['inf', orientation], interpolation='none')
        plt.xticks([]); plt.yticks([]);
        plot3d.add_orientation(ax3, orientation, fontsize=10)

        ax4 = plt.subplot(gs2[1, i])
        plt.imshow(imgs['res', orientation], interpolation='none')
        plt.xticks([]); plt.yticks([]);
        plot3d.add_orientation(ax4, orientation, fontsize=10)

        if i == 0:
            axes.extend([ax3, ax4])

    # Precision-recall plot
    prlines = [(1, 'Engel I', 'tab:blue', '-'), 
               (2, 'Engel II', 'tab:green', '--'),
               (3, 'Engel III-IV', 'tab:red', '-.')]

    ax5 = plt.subplot(gs3[0, 0])
    for subset, label, color, ls in prlines:
        r = res[subset]
        plt.plot(r['recall'], r['prec'], color=color, ls=ls,
                 label=f"{label}\n(n={r['nsubjects']} subjects)")    
        plt.axhline(r['nresected']/r['n'], color=color, ls=ls)

    plt.legend(loc='center left', bbox_to_anchor=(1.45, 0.5),
               bbox_transform=ax5.transAxes)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.xticks([0, 0.5, 1.0]); plt.yticks([0, 0.5, 1.0])

    # Precision
    ax6 = plt.subplot(gs3[1, 0])
    for subset, label, color, ls in prlines:
        r = res[subset]
        plt.plot(r['threshold'], r['prec'][1:], color=color, ls=ls)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0, 0.5, 1.0])

    # Recall
    ax7 = plt.subplot(gs3[1, 1])
    for subset, label, color, ls in prlines:
        r = res[subset]
        plt.plot(r['threshold'], r['recall'][1:], color=color, ls=ls)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0, 0.5, 1.0])

    # Number of resected
    ax8 = plt.subplot(gs3[2, 0])
    tmin = 0
    tmax = max([np.max(r['threshold']) for r in res.values()])
    for subset, label, color, ls in prlines:
        r = res[subset]
        plt.plot([tmin, tmax], np.repeat(r['nresected']/r['nsubjects'], 2),
                 color=color, ls=ls)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("N resected\nper subject", fontsize=12)
    plt.ylim(-0.05, 8.05)

    # Number of predicted
    ax9 = plt.subplot(gs3[2, 1])
    for subset, label, color, ls in prlines:
        r = res[subset]
        plt.plot(r['threshold'], r['npred'][1:]/r['nsubjects'], color=color, ls=ls)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("N predicted\nper subject", fontsize=12)
    plt.ylim(-0.05, 8.05)

    # Marks, annotations etc
    targs = {'fontsize': 18, 'ha': 'center', 'va': 'center'}
    axbg = plt.axes([0,0,1,1], facecolor=None)
    axbg.add_line(lines.Line2D([0.105, 0.105], [0, 1], color='0.5', lw=1))
    axbg.add_line(lines.Line2D([0.56, 0.56], [0, 1], color='0.5', lw=1))
    # axbg.add_line(lines.Line2D([0., 1], [0.5, 0.5], color='0.5', lw=1))
    plt.text(0.34, 0.95, "Inference: $p(c > c_h)$", **targs)
    plt.text(0.34, 0.47, "Resection", **targs)
    plt.axis('off')

    plot.add_panel_letters(fig, [ax1, axes[0], ax5], fontsize=22, 
                           xpos=[-1.4, -0.05, -0.3], ypos=[1.05, 1.12, 1.16])

    plt.savefig(outfile, dpi=300)    


if __name__ == "__main__":
    plot_resection(sys.argv[1])