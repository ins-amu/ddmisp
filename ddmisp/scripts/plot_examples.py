import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from util.post import seizure_fig

subjects_dir = "/home/vep/RetrospectivePatients/1-Processed/"
infer_dir = "run/solo/INC/vep/"
chains = [1, 2]
surgery_file = "data/surgeries.vep.json"
region_name_file = "data/conn/region_names.vep.txt"

def plot_seizure(sid, rid, times=(40, 60, 80), views=('topdown', 'rightleft')):
    datafile = os.path.join(infer_dir, f"id{sid:03d}/input/r{rid:02d}_all.R")
    resfiles = [os.path.join(infer_dir, f"id{sid:03d}/output/r{rid:02d}_all/chain_{ch}.csv") for ch in chains]
    statuses = [os.path.join(infer_dir, f"id{sid:03d}/output/r{rid:02d}_all/chain_{ch}.status") for ch in chains]
    resfiles = [r for r, s in zip(resfiles, statuses) if int(open(s).read().strip())]
    
    fig, axes = seizure_fig(sid, datafile, resfiles, surgery_file, subjects_dir, region_name_file, 
                            variant='paper', times=times, views=views)
    return fig, axes

def mark(axes, x, y, **args):
    defargs = dict(ec='k', lw=0.5, zorder=10, clip_on=False)
    defargs.update(args)
    for ax in axes:
        ax.scatter(x, y, **defargs)  
        
        
if __name__ == "__main__":
    out_direc = sys.argv[1]
    os.makedirs(out_direc)
        
    fig, axes = plot_seizure(17, 3, views=('topdown', 'leftright'))
    mark([axes[0], axes[1]], [26], [73], color='yellow', marker='o', s=50)
    mark([axes[0], axes[1]], [26], [52], color='lime', marker='s', s=40)
    mark([axes[4]], [1], [73], color='yellow', marker='o', s=50)
    mark([axes[4]], [1], [52], color='lime', marker='s', s=40)
    plt.savefig(os.path.join(out_direc, "inf-s017-r03.pdf"), dpi=300)
    plt.close()
    
    fig, axes = plot_seizure(33, 9)
    mark([axes[0], axes[1]], [27], [125], color='yellow', marker='o', s=50)
    mark([axes[1]], [27], [147], color='magenta', marker='s', s=40)
    mark([axes[4]], [1], [126], color='lime', marker='o', s=50)
    mark([axes[4]], [1], [147], color='royalblue', marker='s', s=40)
    plt.savefig(os.path.join(out_direc, "inf-s033-r09.pdf"), dpi=300)
    plt.close()    

    fig, axes = plot_seizure(1, 2)
    mark([axes[0], axes[1]], [26], [154], color='yellow', marker='o', s=50)
    mark([axes[0], axes[1]], [38], [122], color='magenta', marker='s', s=40)
    mark([axes[0], axes[1]], [77], [37],  color='lime', marker='>', s=50)
    mark([axes[4]], [1], [154],  color='royalblue', marker='o', s=50)
    plt.savefig(os.path.join(out_direc, "inf-s001-r02.pdf"), dpi=300)
    plt.close()
    
    fig, axes = plot_seizure(36, 0)
    plt.savefig(os.path.join(out_direc, "inf-s036-r00.pdf"), dpi=300)
    plt.close()       