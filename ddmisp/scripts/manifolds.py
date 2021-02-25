import os
import sys

import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

import vedo as vp
import vtk

sys.path.append(os.getcwd())
from util import simprop, plot



def subdivide_adapt(mesh, max_edge_length, max_triangle_area, max_num_triangles):
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(mesh._polydata)
    triangles.Update()
    originalMesh = triangles.GetOutput()
    sdf = vtk.vtkAdaptiveSubdivisionFilter()
    sdf.SetMaximumEdgeLength(max_edge_length)
    sdf.SetMaximumTriangleArea(max_triangle_area)
    sdf.SetMaximumNumberOfTriangles(max_num_triangles)
    sdf.SetInputData(originalMesh)
    sdf.Update()
    return mesh._update(sdf.GetOutput())

def viz_observation_manifold(t3, tlim, size):
    tmin = 0
    tmax = 2*tlim

    # tlim line
    vline1 = vp.Tube([[tmin, tlim, t3], [tmax, tlim, t3]], r=2.0)
    vline1.color('g')

    # t = 0 line
    vline2 = vp.Tube([[tmin, tlim, t3], [tmin, tmax, t3]], r=2.0)
    vline2.color((1, 1, 1))

    # Manifold
    verts = [[tmin, tlim, t3], [tmax, tlim, t3], [tmin, tmax, t3], [tmax, tmax, t3]]
    triangs = [[0, 1, 3], [0, 3, 2]]
    vmesh1 = vp.Mesh([verts, triangs])
    vmesh1.color((1, 1, 1))

    # Inverse manifold
    verts = [[tmin, tmin, t3], [tmax, tmin, t3], [tmin, tlim, t3], [tmax, tlim, t3]]
    triangs = [[0, 1, 3], [0, 3, 2]]
    vmesh2 = vp.Mesh([verts, triangs])
    vmesh2.color((0.9, 0.9, 0.9)).alpha(0.0)

    # Invisible points to set the extent
    vpoints = vp.Points([(tmin-0.1, tmin-0.1, tmin-0.1),
                         (1.01*tmax, 1.01*tmax, 1.01*tmax)]).alpha(0.0)

    lpos = [(p, str(p)) for p in [0, 50, 100, 150]]

    vplotter = vp.Plotter(
        offscreen=True, size=size,
        axes=dict(
            xyGrid=True, yzGrid=True, zxGrid=True,
            xTitleSize=0, yTitleSize=0, zTitleSize=0,
            xPositionsAndLabels=lpos,
            yPositionsAndLabels=lpos,
            zPositionsAndLabels=lpos[1:],
            axesLineWidth=5, tipSize=0.02, gridLineWidth=2,
            xLabelSize=0.05, yLabelSize=0.05, zLabelSize=0.05,
            xLabelOffset=0.05, yLabelOffset=0.05, zLabelOffset=0.0,
            zTitleRotation=225)
    )
    vlabels = [vp.Text2D("H", (0.09 * size[0], 0.10 * size[1]), s=3, font='Arial'),
               vp.Text2D("N", (0.87 * size[0], 0.16 * size[1]), s=3, font='Arial'),
               vp.Text2D("S", (0.49 * size[0], 0.90 * size[1]), s=3, font='Arial')]

    vp.show([vline1, vline2, vmesh1, vpoints] + vlabels,
            camera=dict(pos=(378, 324, 450), focalPoint=(tlim,tlim,tlim+27), viewup=(0, 0, 1))
    )

    img = vp.screenshot(None, scale=1, returnNumpy=True)
    vp.clear()
    vp.closePlotter()

    return img

def viz_param_manifold(filename, size):
    data = np.load(filename)

    vline = vp.Tube(data['boundary_hns'], r=0.08)
    vline.color('g')

    # HNS manifold
    vmesh_hns = vp.Mesh([data['verts_hns'], data['triangs_hns']])
    k = 3
    prior = (2*np.pi)**(-k/2) * (np.exp(-0.5 * np.sum(vmesh_hns.points()**2, axis=1)))
    vmesh_hns.pointColors(prior, cmap='Reds', vmin=0)
    vmesh_hns.addScalarBar(horizontal=True, nlabels=6, c='k', pos=(0.74, 0.01),
                           titleFontSize=44)
    vmesh_hns.scalarbar.SetLabelFormat("%.2g")
    vmesh_hns.scalarbar.SetBarRatio(1.0)

    # Inverted HNS manifold
    vmesh_hnsi = vp.Mesh([data['verts_hnsi'], data['triangs_hnsi']])
    # vmesh_hnsi.color([0.68, 0.68, 0.68])
    vmesh_hnsi.color([0.9, 0.9, 0.9]).alpha(0.0)


    # Invisible points to set the extent
    vpoints = vp.Points([(-5.01, -5.01, -5.01), (5.01, 5.01, 5.01)]).alpha(0.0)

    vplotter = vp.Plotter(
        offscreen=True, size=size,
        axes=dict(
            xyGrid=True, yzGrid=True, zxGrid=True,
            xTitleSize=0, yTitleSize=0, zTitleSize=0,
            xHighlightZero=True, yHighlightZero=True, zHighlightZero=True,
            xHighlightZeroColor='b', yHighlightZeroColor='b', zHighlightZeroColor='b',
            numberOfDivisions=10, axesLineWidth=5, tipSize=0.02, gridLineWidth=2,
            xLabelSize=0.05, yLabelSize=0.05, zLabelSize=0.05,
            xLabelOffset=0.05, yLabelOffset=0.05, zLabelOffset=0.0,
            zTitleRotation=225)
    )
    vlabels = [vp.Text2D("H", (0.09 * size[0], 0.10 * size[1]), s=3, font='Arial'),
               vp.Text2D("N", (0.87 * size[0], 0.16 * size[1]), s=3, font='Arial'),
               vp.Text2D("S", (0.49 * size[0], 0.90 * size[1]), s=3, font='Arial')]

    k = 2
    vecs = np.array([[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],
                     [[k,-k,0,0,0,0],[0,0,k,-k,0,0],[0,0,0,0,k,-k]]])
    varrows = vp.Arrows(vecs[0].T, vecs[1].T, s=1.2, c='k')

    vp.show([vline, vmesh_hns, vmesh_hnsi, vpoints, varrows] + vlabels,
            camera=dict(pos=(16, 13, 20), focalPoint=(0, 0, 1.5), viewup=(0, 0, 1))
    )

    img = vp.screenshot(None, scale=1, returnNumpy=True)
    vp.clear()
    vp.closePlotter()

    return img



def get_c_hss(w, q, c1s, t2, t3):
    cs = np.zeros((len(c1s), 3))
    for i, c1 in enumerate(c1s):
        c, t = simprop.propinv(w, q, np.array([0, 1, 1]).astype(bool),
                               np.array([t2, t3]), np.array([c1]))
        cs[i, :] = c

    return cs


def get_c_hhs(w, q, c1s, c2s, t3):
    n1 = len(c1s)
    n2 = len(c2s)

    c1s, c2s = np.meshgrid(c1s, c2s)
    c1s, c2s = c1s.ravel(), c2s.ravel()

    # Vertices
    verts = np.zeros((len(c1s), 3))
    for i, (c1, c2) in enumerate(zip(c1s, c2s)):
        c, t = simprop.propinv(w, q, np.array([0, 0, 1]).astype(bool),
                               np.array([t3]), np.array([c1, c2]))
        verts[i, :] = c

    # Triangles
    triangs = []
    for i in range(n1-1):
        for j in range(n2-1):
            triangs.append([j*n1 + i, j*n1 + i+1, (j+1)*n1 + i+1])
            triangs.append([j*n1 + i, (j+1)*n1 + i+1, (j+1)*n1 + i])
    triangs = np.array(triangs)

    return verts, triangs

def get_c_hhs_adaptive(w, q, c1range, n1, c2range, n2, t3, tol, maxlevel):
    c1 = lambda ind: c1range[0] + ind * (c1range[1] - c1range[0]) / ((n1-1) * 2**maxlevel)
    c2 = lambda ind: c2range[0] + ind * (c2range[1] - c2range[0]) / ((n2-1) * 2**maxlevel)

    # Create initial rectangular grid
    obsmask = np.array([0, 0, 1]).astype(bool)
    vertices = {}
    rectangles = []
    for i in range(n1+1):
        for j in range(n2+1):
            spacing = 2**maxlevel
            ind1 = i * spacing
            ind2 = j * spacing
            c, _ = simprop.propinv(w, q, obsmask, [t3], [c1(ind1), c2(ind2)])
            vertices[ind1, ind2] = (len(vertices), c)
            if i < n1-1 and j < n2-1:
                rectangles.append((ind1, ind2, spacing, spacing))

    # Refine
    nmin = 0
    def refine_rectangle(rect, vertices):
        nonlocal nmin

        indx, indy, spacingx, spacingy = rect

        vs = [(indx, indy), (indx + spacingx, indy), (indx, indy + spacingy), (indx + spacingx, indy + spacingy)]
        cs = [0, 0, 0, 0]
        for i, v in enumerate(vs):
            try:
                cs[i] = vertices[v][1][2]
            except KeyError:
                c, _ = simprop.propinv(w, q, obsmask, [t3], [c1(v[0]), c2(v[1])])
                vertices[v] = (len(vertices), c)
                cs[i] = c[2]

        refinex = ((np.abs(cs[0] - cs[1]) > tol) or (np.abs(cs[2] - cs[3]) > tol))
        refiney = ((np.abs(cs[0] - cs[2]) > tol) or (np.abs(cs[1] - cs[3]) > tol))

        if refinex and spacingx == 1:
            nmin +=1
            refinex = False
        if refiney and spacingy == 1:
            nmin += 1
            refiney = False

        if refinex and refiney:
            rects1 = refine_rectangle((indx,                 indy,                 spacingx // 2, spacingy // 2), vertices)
            rects2 = refine_rectangle((indx + spacingx // 2, indy,                 spacingx // 2, spacingy // 2), vertices)
            rects3 = refine_rectangle((indx,                 indy + spacingy // 2, spacingx // 2, spacingy // 2), vertices)
            rects4 = refine_rectangle((indx + spacingx // 2, indy + spacingy // 2, spacingx // 2, spacingy // 2), vertices)
            return rects1 + rects2 + rects3 + rects4
        elif refinex:
            rects1 = refine_rectangle((indx,                 indy, spacingx // 2, spacingy), vertices)
            rects2 = refine_rectangle((indx + spacingx // 2, indy, spacingx // 2, spacingy), vertices)
            return rects1 + rects2
        elif refiney:
            rects1 = refine_rectangle((indx, indy,                 spacingx, spacingy // 2), vertices)
            rects2 = refine_rectangle((indx, indy + spacingy // 2, spacingx, spacingy // 2), vertices)
            return rects1 + rects2
        else:
            return [rect]

    refined_rectangles = []
    for r in rectangles:
        refined_rectangles.extend(refine_rectangle(r, vertices))

    if nmin > 0:
        print(f"Refinement limit reached: {nmin}x")

    # Create the final vertices and triangles
    verts = np.zeros((len(vertices), 3))
    for index, c in vertices.values():
        verts[index, :] = c

    nr = len(refined_rectangles)
    triangs = np.zeros((2*nr, 3), dtype=int)
    for i, r in enumerate(refined_rectangles):
        va = (r[0],        r[1])
        vb = (r[0] + r[2], r[1])
        vc = (r[0],        r[1] + r[3])
        vd = (r[0] + r[2], r[1] + r[3])
        triangs[2*i]     = [vertices[va][0], vertices[vb][0], vertices[vd][0]]
        triangs[2*i + 1] = [vertices[va][0], vertices[vd][0], vertices[vc][0]]

    return verts, triangs


def build_manifold(w, q, c_range, n, t3, tlim, out_file):
    ADAPT_MAXLEVEL = 2
    ADAPT_TOL = 0.2

    # Hidden-Nonseizing-Seizing boundary
    # c = np.linspace(c_range[0], c_range[1], n)
    c = np.linspace(c_range[0], c_range[1], n * 2**ADAPT_MAXLEVEL)
    boundary_hns = get_c_hss(w, q, c, tlim, t3)

    # Hidden-Hidden-Seizing manifold
    verts_hhs, triangs_hhs = get_c_hhs_adaptive(w, q, c_range, n, c_range, n, t3,
                                                tol=ADAPT_TOL, maxlevel=ADAPT_MAXLEVEL)

    # Cutting mesh
    cut_line = np.copy(boundary_hns)
    cut_line[:, 2] = c_range[0]
    cut_line = vp.Line(cut_line)
    cut_mesh = cut_line.extrude(zshift=c_range[1] - c_range[0])

    # Hidden-Nonseizing-Seizing manifold
    vmesh_hhs = vp.Mesh([verts_hhs, triangs_hhs])
    vmesh_hns = vmesh_hhs.cutWithMesh(cut_mesh, invert=False)
    verts_hns   = vmesh_hns.points()
    triangs_hns = vmesh_hns.faces()

    # Hidden-Nonseizing-Seizing inverted manifold
    vmesh_hhs = vp.Mesh([verts_hhs, triangs_hhs])
    vmesh_hnsi = vmesh_hhs.cutWithMesh(cut_mesh, invert=True)
    verts_hnsi   = vmesh_hnsi.points()
    triangs_hnsi = vmesh_hnsi.faces()

    assert all([len(f) == 3 for f in triangs_hns])
    assert all([len(f) == 3 for f in triangs_hnsi])

    np.savez(out_file, boundary_hns=boundary_hns,
             verts_hns=verts_hns,   triangs_hns=triangs_hns,
             verts_hnsi=verts_hnsi, triangs_hnsi=triangs_hnsi,
             verts_hhs=verts_hhs,   triangs_hhs=triangs_hhs)


def recprob(tsamples, nt, tlim, resected=None):
    nreg = tsamples.shape[1]
    ts = np.linspace(0, tlim, nt, endpoint=True)
    psz = np.zeros((nreg, nt))
    for i in range(nt):
        psz[:, i] = np.mean(tsamples < ts[i], axis=0)
    if resected is not None:
        for reg in resected:
            psz[reg, :] = np.nan
    return psz


def get_results(filename, q, w, clim, nc, nt, nsamples, refine_mesh=False):
    data = np.load(filename)

    # Create and refine mesh
    verts = data['verts_hns']
    triangs = data['triangs_hns']
    mesh = vp.Mesh([verts, triangs])
    if refine_mesh:
        # Subdivision fails for A. Perhaps because it's flat?
        mesh = subdivide_adapt(mesh, max_edge_length=0.1, max_triangle_area=1,
                               max_num_triangles=1000000)
    verts = mesh.points()
    triangs = mesh.faces()

    # Calculate triangle areas
    triangle_areas = np.zeros(len(triangs))
    triangle_centers = np.zeros((len(triangs), 3))
    for i, triang in enumerate(triangs):
        vs = verts[triang, :]
        triangle_areas[i] = 0.5*np.linalg.norm(np.cross(vs[1] - vs[0], vs[2] - vs[0]))
        triangle_centers[i] = np.mean(vs, axis=0)

    prior = np.exp(-np.sum(triangle_centers**2, axis=1)/2.) / np.sqrt(2*np.pi)
    triangle_post = prior * triangle_areas
    triangle_post /= np.sum(triangle_post)

    # Get samples
    csamples = triangle_centers[np.random.choice(len(triangs), size=nsamples,
                                                 p=triangle_post)]

    # Density
    bins = np.linspace(clim[0], clim[1], nc+1)
    cdens = np.zeros((3, nc))
    for i in range(3):
        cdens[i] = np.histogram(csamples[:, i], bins=bins, density=True)[0]

    # Onset time samples
    tsamples = np.zeros_like(csamples)
    for i, c in enumerate(csamples):
        tsamples[i, :] = simprop.prop(c, w, q)

    psz = recprob(tsamples, nt+1, tlim)

    return cdens, psz


def make_figure(manifolds, sketch_file, w, t3, tlim, out_file):
    clim = (-3, 3)
    nc = 30
    nt = 100

    psz = {}
    cdens = {}
    for fun, name, q, filename in manifolds:
        print(fun, flush=True)
        cdens[fun], psz[fun] = get_results(
            filename, q, w, clim=clim, nc=nc, nt=nt, nsamples=100000,
            refine_mesh=(fun != 'A'))
        # cdens[fun] = np.random.uniform(0, 1, (3, nc))
        # psz[fun] = np.random.uniform(0, 1, (3, nt))

    fig = plt.figure(figsize=(12, 6))

    gs = gridspec.GridSpec(2, 5, height_ratios=[1, 1], width_ratios=[1, 0.0, 1, 1, 1],
                           left=0.02, right=0.89, bottom=0.08, top=0.94,
                           hspace=0.3, wspace=0.3)

    # Network sketch
    gsa = gridspec.GridSpecFromSubplotSpec(3, 3, gs[0, 0], hspace=0, wspace=0,
                                           height_ratios=[1, 5, 1], width_ratios=[1, 5, 1])
    axa = plt.subplot(gsa[1, 1])
    img_network = plt.imread(sketch_file)
    plt.imshow(img_network, interpolation='none')
    plt.axis('off')

    # Observation manifold
    axb = plt.subplot(gs[1, 0])
    imb = viz_observation_manifold(t3, tlim, (1000, 1000))
    plt.imshow(imb, interpolation='none')
    plt.xticks([]); plt.yticks([])

    cmax = np.log(np.max([np.max(a) for a in cdens.values()]))

    for i, (fun, name, q, filename) in enumerate(manifolds):
        # Parameter manifold
        ax0 = plt.subplot(gs[0, i+2])
        plt.title(name, fontsize=16)
        im0 = viz_param_manifold(filename, (1000, 1000))
        plt.imshow(im0, interpolation='none')
        plt.xticks([]); plt.yticks([])

        gscd = gridspec.GridSpecFromSubplotSpec(2, 1, gs[1, i+2], hspace=1.3)

        # c density
        ax1 = plt.subplot(gscd[0])
        im1 = plt.imshow(np.log(cdens[fun]+1e-10), aspect='auto', origin='lower',
                         cmap='Reds', vmin=cmax-4, vmax=cmax,
                         extent=[clim[0], clim[-1], -0.5, 2.5])
        plt.xlabel("Excitability $c$")
        plt.yticks([0, 1, 2], ['H', 'N', 'S'])
        ax1.tick_params('y', length=0)

        # Recruitment probability
        ax2 = plt.subplot(gscd[1])
        im2 = plt.imshow(psz[fun], aspect='auto', cmap='bwr', origin='lower',
                         vmin=0, vmax=1,
                         extent=[0, tlim, -0.5, 2.5])
        plt.xlabel("Time [s]")
        plt.xticks([0, 30, 60, 90])
        plt.yticks([0, 1, 2], ['H', 'N', 'S'])
        ax2.tick_params('y', length=0)
        if i == 2:
            ax1cb = ax1.inset_axes([1.15, 0.0, 0.1, 1.0])
            ax2cb = ax2.inset_axes([1.15, 0.0, 0.1, 1.0])
            cb1 = plt.colorbar(im1, cax=ax1cb)
            cb2 = plt.colorbar(im2, cax=ax2cb)
            cb1.set_label("$\log\ p(c)$")
            cb2.set_label("Recruitment\nprobability")

        if i == 0:
            axc, axd, axe = ax0, ax1, ax2

    bg = plot.Background(visible=False, spacing=0.1)
    # bg.vline(x=0.22)
    plot.add_panel_letters(fig, [axa, axb, axc, axd, axe], fontsize=22,
                           xpos=[-0.16, -0.0, -0.2, -0.2, -0.2],
                           ypos=[1.24, 1.1, 1.0, 1.1, 1.1])

    plt.savefig(out_file, dpi=300)


if __name__ == "__main__":
    FUNS = {
        'A': ('Uncoupled',       [ -5.1174474, -5.1174474, 1.9464059,  1.9464059]),
        'C': ('Weak coupling',   [-10.,         2.,        5.5,       33.       ]),
        'D': ('Strong coupling', [-12.6997222, 15.4796221, 5.5270351, 75.2121565])
    }

    w = 0.1 * np.ones((3, 3), dtype=float)
    np.fill_diagonal(w, 0.0)
    t3 = 50.0
    tlim = 90.0

    if sys.argv[1] == "build":
        fun = sys.argv[2]
        out_file = sys.argv[3]
        build_manifold(w, FUNS[fun][1], (-5, 5), 200, t3, tlim, out_file)
    elif sys.argv[1] == "plot":
        sketch = sys.argv[2]
        funs = sys.argv[3].split(' ')
        manifold_files = sys.argv[4].split(' ')
        out_file = sys.argv[5]
        manifolds = [(fun, FUNS[fun][0], FUNS[fun][1], filename)
                     for fun, filename in zip(funs, manifold_files)]
        make_figure(manifolds, sketch, w, t3, tlim, out_file)
