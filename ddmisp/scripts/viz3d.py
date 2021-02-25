import glob
import os
import sys
import zipfile

import numpy as np

import vedo as vp

from util import io

SUBJECT_DIR = "/home/vep/RetrospectivePatients/1-Processed/"


def read_structural_data(sid, rid):
    subject = os.path.basename(glob.glob(os.path.join(SUBJECT_DIR, f"id{sid:03d}_*"))[0])

    # Load data
    with zipfile.ZipFile(os.path.join(SUBJECT_DIR, subject, "tvb/connectivity.vep.zip")) as zf:
        with zf.open("centres.txt") as fh:
            regpos = np.genfromtxt(fh, usecols=(1,2,3))
    w = np.genfromtxt(f"data/conn/vep/id{sid:03d}.txt")

    with zipfile.ZipFile(os.path.join(SUBJECT_DIR, subject, "tvb/surface_cort.vep.zip")) as zf:
        with zf.open("vertices.txt") as fh:
            verts_ctx = np.genfromtxt(fh)
        with zf.open("triangles.txt") as fh:
            triangs_ctx = np.genfromtxt(fh)

    with zipfile.ZipFile(os.path.join(SUBJECT_DIR, subject, "tvb/surface_subcort.vep.zip")) as zf:
        with zf.open("vertices.txt") as fh:
            verts_sub = np.genfromtxt(fh)
        with zf.open("triangles.txt") as fh:
            triangs_sub = np.genfromtxt(fh)

    contacts = np.genfromtxt(os.path.join(SUBJECT_DIR, subject, "elec/seeg.xyz"), usecols=(1,2,3))

    # Load input
    indata = io.rload(f"run/solo/INC/vep/id{sid:03d}/input/r{rid:02d}_all.R")
    reg_obs = np.concatenate([indata['reg_ns'], indata['reg_sz']]).astype(int)
    obsmask = np.zeros(162, dtype=bool)
    obsmask[reg_obs] = True

    return regpos, w, obsmask, [(verts_ctx, triangs_ctx), (verts_sub, triangs_sub)], contacts


def viz_structure(regpos, w, surfaces, contacts):
    # Connections
    wmax = np.max(w)
    vlines = []
    for reg1, reg2 in zip(*np.where(w > np.percentile(w.flat, 97))):
        vlines.append(vp.Line(regpos[reg1], regpos[reg2], c='k', lw=15*w[reg1, reg2]/wmax))

    # Brain surfaces
    vmeshes = [vp.Mesh([verts, triangs], 'grey', alpha=0.05) for verts, triangs in surfaces]

    # Electrodes
    ncontacts = contacts.shape[0]
    vcontacts = []
    for i in range(ncontacts):
        vcontacts.append(vp.Sphere(contacts[i], r=0.8, c='green'))

    return vlines, vmeshes, vcontacts



def viz_mean_excitability(sid, rid):
    regpos, w, obsmask, surfaces, contacts = read_structural_data(sid, rid)
    vlines, vmeshes, vcontacts = viz_structure(regpos, w, surfaces, contacts)

    # Load results
    nreg = regpos.shape[0]
    res = io.parse_csv([f"run/solo/INC/vep/id{sid:03d}/output/r{rid:02d}_all/chain_{chain}.csv" for chain in [1,2]])
    cinf = res['c']
    # cmean = np.mean(cinf, axis=0)
    # pexc = np.mean(cinf > 2.0, axis=0)
    scalar = np.percentile(cinf, 50, axis=0)

    # Regions
    cmap = 'plasma'
    # vmin = np.min(scalar)
    # vmax = np.max(scalar)
    vmin, vmax = -2, 2

    vpoints = []
    for i in range(nreg):
        if not obsmask[i]:
            vpoints.append(vp.Sphere(regpos[i], r=4, c=vp.colorMap(scalar[i], cmap, vmin, vmax)))
        else:
            vpoints.append(vp.Cube(regpos[i], side=6, c=vp.colorMap(scalar[i], cmap, vmin, vmax)))

    vbar = vp.Points(regpos, r=0.01).pointColors(scalar, cmap=cmap, vmin=vmin, vmax=vmax)
    vbar.addScalarBar(horizontal=True, pos=(0.8, 0.02))

    def slider(widget, event):
        percentile = widget.GetRepresentation().GetValue()
        scalar = np.percentile(cinf, percentile, axis=0)
        for i in range(nreg):
            vpoints[i].color(vp.colorMap(scalar[i], cmap, vmin, vmax))


    vplotter = vp.Plotter(axes=0)
    vplotter.addSlider2D(slider, 0., 100., value=50.0, pos=3, title="Percentile")
    vplotter.show(vpoints, vlines, vmeshes, vcontacts, vbar)


def viz_excitability(sid, rid):
    regpos, w, obsmask, surfaces, contacts = read_structural_data(sid, rid)
    vlines, vmeshes, vcontacts = viz_structure(regpos, w, surfaces, contacts)

    # Load results
    nreg = regpos.shape[0]
    res = io.parse_csv([f"run/solo/INC/vep/id{sid:03d}/output/r{rid:02d}_all/chain_{chain}.csv" for chain in [1,2]])
    cinf = res['c']

    ctr = 2.0
    pexc = np.mean(cinf > ctr, axis=0)

    # Regions
    cmap = 'Reds'
    vmin, vmax = 0, 0.15
    # vmin, vmax = -2, 0

    vpoints = []
    for i in range(nreg):
        if not obsmask[i]:
            vpoints.append(vp.Sphere(regpos[i], r=4, c=vp.colorMap(pexc[i], cmap, vmin, vmax)))
        else:
            vpoints.append(vp.Cube(regpos[i], side=6, c=vp.colorMap(pexc[i], cmap, vmin, vmax)))

    vbar = vp.Points(regpos, r=0.01).pointColors(pexc, cmap=cmap, vmin=vmin, vmax=vmax)
    vbar.addScalarBar(horizontal=True, pos=(0.8, 0.02))

    def cslider(widget, event):
        ctr = widget.GetRepresentation().GetValue()
        pexc = np.mean(cinf > ctr, axis=0)
        for i in range(nreg):
            vpoints[i].color(vp.colorMap(pexc[i], cmap, vmin, vmax))

    vplotter = vp.Plotter(axes=0)
    vplotter.addSlider2D(cslider, -3.0, 3.0, value=2.0, pos=3, title="c")
    vplotter.show(vpoints, vlines, vmeshes, vcontacts, vbar)


def video_pause(vplotter, video, vobjects, cam, nframes):
    print("Pause...")
    for i in range(nframes):
        vplotter.show(*vobjects, camera={'pos': cam['pos'], 'focalPoint': cam['focalPoint'], 'viewup': cam['viewup']})
        video.addFrame()
        
def video_fly(vplotter, video, vobjects, param_range, nframes, pos, focalpoint, viewup, startpoint=True, endpoint=True):
    print("Fly...")
    
    if startpoint and endpoint:
        params = np.linspace(param_range[0], param_range[1], nframes)
    elif startpoint and not endpoint:
        params = np.linspace(param_range[0], param_range[1], nframes+1)[:-1]
    elif not startpoint and endpoint:
        params = np.linspace(param_range[0], param_range[1], nframes+1)[1:]
    else:
        params = np.linspace(param_range[0], param_range[1], nframes+2)[1:-1]
    
    for param in params:
        p = pos        if not callable(pos)        else pos(param)
        f = focalpoint if not callable(focalpoint) else focalpoint(param)
        v = viewup     if not callable(viewup)     else focalpoint(vparam)            
        vplotter.show(*vobjects, camera={'pos': p, 'focalPoint': f, 'viewup': v})
        video.addFrame()
        
        

def video_timestep(vplotter, video, vpoints, vtext, vobjects, tfrom, tto, nframes, tinf, cmap, cam):
    print("Run...")
    
    for t in np.linspace(tfrom, tto, nframes):
        pszs = np.mean(tinf < t, axis=0)
        for vpoint, psz in zip(vpoints, pszs):
            vpoint.color(vp.colorMap(psz, cmap, 0, 1))
            
        # vtext.SetText(1, f"t = {t:4.1f} s")
        # vtext.SetNonlinearFontScaleFactor(2.0/2.7)
        
        # Recreate vtext
        # print(vtext.s)        
        
        # vplotter.remove(vtext)
        # vtext = vp.Text2D(f"t = {t:4.1f} s", pos=(0.1, 0.1), s=3, c='black')
        # vtext = vp.Text2D(f"t = {t:4.1f} s", pos=(0.1, 0.1), s=3, c='black')
        
        # vplotter.show(vpoints, vtext, *vobjects, camera={'pos':cam['pos'], 'focalPoint':cam['focalPoint'], 'viewup':cam['viewup']})
        # vplotter.clear()
        # vplotter.show(vpoints, vtext, *vobjects, camera={'pos':cam['pos'], 'focalPoint':cam['focalPoint'], 'viewup':cam['viewup']})
        vtext = vp.Text2D(f"t = {t:4.1f} s", (10, 10), s=6, c='black')
        vplotter += vtext
        vplotter.show(camera={'pos':cam['pos'], 'focalPoint':cam['focalPoint'], 'viewup':cam['viewup']})
        video.addFrame()
        vplotter -= vtext
        

def animate(vplotter, video, nframes, vpoints, tinf, prange=(0.,1.), time=0.0, pos=(1,0,0), foc=(0,0,0), viewup=(0,1,0),
            startpoint=True, endpoint=True):

    cmap = 'bwr'    
    
    if startpoint and endpoint:
        params = np.linspace(prange[0], prange[1], nframes)
    elif startpoint and not endpoint:
        params = np.linspace(prange[0], prange[1], nframes+1)[:-1]
    elif not startpoint and endpoint:
        params = np.linspace(prange[0], prange[1], nframes+1)[1:]
    else:
        params = np.linspace(prange[0], prange[1], nframes+2)[1:-1]
        
    for param in params:
        p = pos     if not callable(pos)    else pos(param)
        f = foc     if not callable(foc)    else foc(param)
        v = viewup  if not callable(viewup) else viewup(param)  
        t = time    if not callable(time)   else time(param)    
        
        pszs = np.mean(tinf < t, axis=0)
        for vpoint, psz in zip(vpoints, pszs):
            vpoint.color(vp.colorMap(psz, cmap, 0, 1))
                         
        vtext = vp.Text2D(f"t = {t:4.1f} s", (10, 10), s=6, c='black')
        vplotter += vtext
        vplotter.show(camera=dict(pos=p, focalpoint=f, viewup=v))
        video.addFrame()
        vplotter -= vtext                         
        


def make_video(sid, rid, video_file):
    regpos, w, obsmask, surfaces, contacts = read_structural_data(sid, rid)
    vlines, vmeshes, vcontacts = viz_structure(regpos, w, surfaces, contacts)

    # Load results
    res = io.parse_csv([f"run/solo/INC/vep/id{sid:03d}/output/r{rid:02d}_all/chain_{chain}.csv" for chain in [1,2]])
    tinf = res['t']

    t = 0.0
    psz = np.mean(tinf < t, axis=0)
    nreg = regpos.shape[0]

    # Regions
    cmap = 'bwr'
    vpoints = []
    for i in range(nreg):
        if not obsmask[i]:
            vpoints.append(vp.Sphere(regpos[i], r=4, c=vp.colorMap(psz[i], cmap, 0, 1)))
        else:
            vpoints.append(vp.Cube(regpos[i], side=6, c=vp.colorMap(psz[i], cmap, 0, 1)))

    vbar = vp.Points(regpos, r=0.01).pointColors(psz, cmap=cmap, vmin=0, vmax=1)
    vbar.addScalarBar(horizontal=True, pos=(0.8, 0.02))
    vtext = vp.Text2D(f"t = {t:4.1f} s", pos=0, s=2, c='black')


    center = np.mean(regpos, axis=0)
    dist = 2.5*(np.max(regpos[:, 1]) - np.min(regpos[:, 1]))

    # Video -------------------------------------------------------
    vplotter = vp.Plotter(axes=0, interactive=0, offscreen=True, size=(1800, 1800))
    nframes = 3000
    
    vplotter += vpoints
    vplotter += vlines
    vplotter += vmeshes    
    
    video = vp.Video(name=video_file, duration=90)
    ratios = np.array([30, 3, 5, 30, 5, 3, 30, 10])
    frames = (nframes * ratios/np.sum(ratios)).astype(int)
   
    # Run and pause
    animate(vplotter, video, frames[0], vpoints, tinf, pos=center+dist*np.r_[0, 0, 1], foc=center, viewup=(0,1,1), prange=(0, 45), time=lambda p:p)
    animate(vplotter, video, frames[1], vpoints, tinf, pos=center+dist*np.r_[0, 0, 1], foc=center, viewup=(0,1,1), time=45.)
    
    ## Fly around    
    pos = lambda angle: center + dist*np.array([0, -np.sin(angle), np.cos(angle)])
    animate(vplotter, video, frames[2], vpoints, tinf, pos=pos, foc=center, viewup=(0,1,1), prange=(0, np.pi/2), time=45., endpoint=False)
    
    pos = lambda angle: center + dist * np.array([-np.sin(angle), -np.cos(angle), 0])
    animate(vplotter, video, frames[3], vpoints, tinf, pos=pos, foc=center, viewup=(0,0,1), prange=(0, 2*np.pi), time=45.)
    
    pos = lambda angle: center + dist*np.array([0, -np.sin(angle), np.cos(angle)])
    animate(vplotter, video, frames[4], vpoints, tinf, pos=pos, foc=center, viewup=(0,1,1), prange=(np.pi/2, 0), time=45., startpoint=False)

    # Pause + run + pause
    animate(vplotter, video, frames[5], vpoints, tinf, pos=center+dist*np.r_[0, 0, 1], foc=center, viewup=(0,1,1), time=45.)
    animate(vplotter, video, frames[6], vpoints, tinf, pos=center+dist*np.r_[0, 0, 1], foc=center, viewup=(0,1,1), prange=(45, 90), time=lambda p:p)
    animate(vplotter, video, frames[7], vpoints, tinf, pos=center+dist*np.r_[0, 0, 1], foc=center, viewup=(0,1,1), time=90.)
    

    video.close()
    # ------------------------------------------------------------------



def viz_seizure(sid, rid):
    regpos, w, obsmask, surfaces, contacts = read_structural_data(sid, rid)
    vlines, vmeshes, vcontacts = viz_structure(regpos, w, surfaces, contacts)

    # Load results
    nreg = regpos.shape[0]
    res = io.parse_csv([f"run/solo/INC/vep/id{sid:03d}/output/r{rid:02d}_all/chain_{chain}.csv" for chain in [1,2]])
    tinf = res['t']
    
    t = 0.0
    psz = np.mean(tinf < t, axis=0)

    # Regions
    cmap = 'bwr'
    vmin, vmax = 0, 1

    vpoints = []
    for i in range(nreg):
        if not obsmask[i]:
            vpoints.append(vp.Sphere(regpos[i], r=4, c=vp.colorMap(psz[i], cmap, vmin, vmax)))
        else:
            vpoints.append(vp.Cube(regpos[i], side=6, c=vp.colorMap(psz[i], cmap, vmin, vmax)))

    vbar = vp.Points(regpos, r=0.01).pointColors(psz, cmap=cmap, vmin=vmin, vmax=vmax)
    vbar.addScalarBar(horizontal=True, pos=(0.8, 0.02))

    def tslider(widget, event):
        t = widget.GetRepresentation().GetValue()
        psz = np.mean(tinf < t, axis=0)
        for i in range(nreg):
            vpoints[i].color(vp.colorMap(psz[i], cmap, vmin, vmax))
            
    vplotter = vp.Plotter(axes=0)
    vplotter.addSlider2D(tslider, 0, 90.0, value=0.0, pos=3, title="t")
    vplotter.show(vpoints, vlines, vmeshes, vcontacts, vbar)


if __name__ == "__main__":
    if sys.argv[1] == "video":
        assert len(sys.argv) == 5
    else:
        assert len(sys.argv) == 4

    if sys.argv[1] == "video":
        make_video(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])

    elif sys.argv[1] == "seizure":
        viz_seizure(int(sys.argv[2]), int(sys.argv[3]))
    
    elif sys.argv[1] == "meanexc":
        viz_mean_excitability(int(sys.argv[2]), int(sys.argv[3]))

    elif sys.argv[1] == "exc":
        viz_excitability(int(sys.argv[2]), int(sys.argv[3]))


# --------------------------------------------------------

# vp.show(vpoints, vlines, vmesh_ctx, vmesh_sub, vtext, interactive=1)

#for t in np.linspace(0, 90, 181):
    #psz = np.mean(tinf < t, axis=0)
    #vpoints.pointColors(psz, cmap='Reds', vmin=0, vmax=1)
    ## vtext = vp.Text2D(f"t = {t:5.2f} s", pos=1, s=0.8)
    #vtext.SetText(0, f"t = {t:5.2f} s")
    #vp.show(vpoints, vlines, vmesh_ctx, vmesh_sub, vtext)
#vp.interactive()
