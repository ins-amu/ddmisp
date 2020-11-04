
import os
import glob
import zipfile

import numpy as np

import matplotlib.pyplot as plt
import vtkplotter as vp

from . import io



def read_structural_data(sid, subject_dir):
    subject = os.path.basename(glob.glob(os.path.join(subject_dir, f"id{sid:03d}_*"))[0])

    # Load data
    with zipfile.ZipFile(os.path.join(subject_dir, subject, "tvb/connectivity.vep.zip")) as zf:
        with zf.open("centres.txt") as fh:
            regpos = np.genfromtxt(fh, usecols=(1,2,3))

    with zipfile.ZipFile(os.path.join(subject_dir, subject, "tvb/surface_cort.vep.zip")) as zf:
        with zf.open("vertices.txt") as fh:
            verts_ctx = np.genfromtxt(fh)
        with zf.open("triangles.txt") as fh:
            triangs_ctx = np.genfromtxt(fh)

    with zipfile.ZipFile(os.path.join(subject_dir, subject, "tvb/surface_subcort.vep.zip")) as zf:
        with zf.open("vertices.txt") as fh:
            verts_sub = np.genfromtxt(fh)
        with zf.open("triangles.txt") as fh:
            triangs_sub = np.genfromtxt(fh)

    contacts = np.genfromtxt(os.path.join(subject_dir, subject, "elec/seeg.xyz"), usecols=(1,2,3))

    return regpos, [(verts_ctx, triangs_ctx), (verts_sub, triangs_sub)], contacts

def read_input_data(datafile):
    indata = io.rload(datafile)
    w = np.array(indata['w'])
    reg_obs = np.concatenate([indata['reg_ns'], indata['reg_sz']]).astype(int)
    nreg = w.shape[0]
    obsmask = np.zeros(nreg, dtype=bool)
    obsmask[reg_obs] = True
    
    return w, obsmask
    

def read_data(sid, datafile, resfiles, subject_dir):
    # Load structure
    regpos, surfs, contacts = read_structural_data(sid, subject_dir)

    # Load input
    indata = io.rload(datafile)
    w = np.array(indata['w'])
    reg_obs = np.concatenate([indata['reg_ns'], indata['reg_sz']]).astype(int)
    nreg = w.shape[0]
    obsmask = np.zeros(nreg, dtype=bool)
    obsmask[reg_obs] = True

    # Load results
    res = io.parse_csv(resfiles)

    return {'pos': regpos, 'w': w, 'surf': surfs, 'contacts': contacts,
            'obsmask': obsmask,
            'reg_ns': indata['reg_ns'], 'reg_sz': indata['reg_sz'], 't_sz': indata['t_sz'],
            'tinf': res['t'], 'cinf': res['c']}


def viz_structure(regpos, w, surfaces, w_perc=3):
    # Connections
    wmax = np.max(w)
    vlines = []
    for reg1, reg2 in zip(*np.where(w > np.percentile(w.flat, (100-w_perc)))):
        vlines.append(vp.Line(regpos[reg1], regpos[reg2], c='k', lw=5*w[reg1, reg2]/wmax))

    # Brain surfaces
    vmeshes = [vp.Mesh([verts, triangs], 'grey', alpha=0.05) for verts, triangs in surfaces]

    return vlines, vmeshes


def viz_network_scalar(centres, scalars, weights, obsmask, surfs, view='topdown', size=(1000, 1000),
                       cmap='viridis', vmin=None, vmax=None):
    vp.embedWindow(False)

    if vmin is None:
        vmin = np.min(scalars)
    if vmax is None:
        vmax = np.max(scalars)

    vlines, vmeshes = viz_structure(centres, weights, surfs, w_perc=3)
    nreg = centres.shape[0]

    # Regions
    vpoints = []
    for i in range(nreg):
        if not obsmask[i]:
            vpoints.append(vp.Sphere(centres[i], r=4, c=vp.colorMap(scalars[i], cmap, vmin, vmax)))
        else:
            vpoints.append(vp.Cube(centres[i], side=6, c=vp.colorMap(scalars[i], cmap, vmin, vmax)))

    if view == 'topdown':
        args = dict(elevation=0, azimuth=0, roll=0, zoom=1.4)
    elif view == 'leftright':
        args = dict(elevation=0, azimuth=270, roll=90, zoom=1.4)
    elif view == 'rightleft':
        args = dict(elevation=0, azimuth=90, roll=270, zoom=1.4)
    else:
        raise ValueError(f"Unexpected view: {view}")

    vplotter = vp.Plotter(axes=0, offscreen=True, size=size)
    vplotter.show(vpoints, vlines, vmeshes,  N=1, **args)
    img = vp.screenshot(None, scale=1, returnNumpy=True)
    vp.clear()
    vp.closePlotter()

    return img


def add_orientation(ax, view, scale=1.0, fontsize=6):
    plt.sca(ax)

    args = dict(xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', fc='k', shrinkA=0, shrinkB=0))

    if view == "topdown":
        plt.annotate("", xy=(0.20, 0.04), xytext=(0.04, 0.04), **args)
        plt.annotate("", xy=(0.04, 0.20), xytext=(0.04, 0.04), **args)
        plt.text(0.20, 0.06, "R", ha='right', va='bottom', fontsize=fontsize, transform=plt.gca().transAxes)
        plt.text(0.07, 0.20, "A", ha='left',  va='top',    fontsize=fontsize, transform=plt.gca().transAxes)
    elif view == "leftright":
        plt.annotate("", xy=(0.80, 0.04), xytext=(0.96, 0.04), **args)
        plt.annotate("", xy=(0.96, 0.20), xytext=(0.96, 0.04), **args)
        plt.text(0.80, 0.06, "A", ha='left',  va='bottom', fontsize=fontsize, transform=plt.gca().transAxes)
        plt.text(0.93, 0.20, "S", ha='right', va='top',    fontsize=fontsize, transform=plt.gca().transAxes)
    elif view == "rightleft":
        plt.annotate("", xy=(0.20, 0.04), xytext=(0.04, 0.04), **args)
        plt.annotate("", xy=(0.04, 0.20), xytext=(0.04, 0.04), **args)
        plt.text(0.20, 0.06, "A", ha='right', va='bottom', fontsize=fontsize, transform=plt.gca().transAxes)
        plt.text(0.07, 0.20, "S", ha='left',  va='top',    fontsize=fontsize, transform=plt.gca().transAxes)
