
import os
import json

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib import gridspec, patches, cm

import numpy as np
import scipy.stats as stats

from . import io, plot3d, plot

def check_chain(csv_file, out_file):
    nsamples, summary = io.parse_summary_csv(csv_file)
    with open(out_file, 'w') as fh:
        if stats.mstats.gmean(summary['c'].StdDev.ravel()) < 1e-5:
            fh.write("0\n")
        else:
            fh.write("1\n")


def plot_hyperparams(res, summary, img_file):
    # Hyper parameters
    plt.figure(figsize=(8, 12))
    gs = gridspec.GridSpec(6, 4)

    params = ['q11', 'q12', 'qa21', 'qa22']
    meanp = [np.mean(res[p]) for p in params]

    for i, paramr in enumerate(params):
        for j, paramc in enumerate(params):
            if j > i:
                continue

            ax = plt.subplot(gs[i, j])
            if i == j:
                plt.hist(res[paramr], bins=20, color='b')
                plt.text(0.05, 0.9, "rhat = %.2f" % (summary[paramr][0][-1]), transform=ax.transAxes)
                plt.yticks([])
                plt.axvline(meanp[i], color='r', lw=1)
            else:
                plt.scatter(res[paramc], res[paramr], color='b', alpha=0.01, s=10)
                plt.scatter(meanp[j], meanp[i], color='r', s=20)

            if i == len(params) - 1:
                plt.xlabel(paramc)
            if j == 0:
                plt.ylabel(paramr)

    # Plot function ----------------------------- #
    ax = plt.subplot(gs[4:6, 0:4])
    cs = np.linspace(-5, 5, 100)
    ys = np.linspace(0, 1, 100)
    css, yss = np.meshgrid(cs, ys)

    # Interpolation points
    c1, c2 = -1, 1
    y1, y2 = 0, 1
    q11, q12, qa21, qa22 = meanp
    q21 = q11 + qa21
    q22 = q12 + qa22

    blint = 1./((c2 - c1)*(y2 - y1)) * (  q11*(c2 - css)*(y2 - yss) + q21*(css - c1)*(y2 - yss)
                                        + q12*(c2 - css)*(yss - y1) + q22*(css - c1)*(yss - y1))
    f = np.exp(blint)
    plt.imshow(f, extent=[cs[0], cs[-1], ys[0], ys[-1]], origin='lower', aspect='auto', vmin=0, vmax=0.1)
    plt.xlabel("c")
    plt.ylabel("y")
    plt.colorbar()
    plt.plot(cs, 0.4*stats.norm().pdf(cs), 'r-', lw=1)

    plt.tight_layout()
    plt.savefig(img_file)




def plot_ct(img_file, cinf, tinf, obs_mask, sz_mask, ctru=None, ttru=None, tlim=90, rhat=None,
            region_names=None, ez_hyp=None):
    EZ_THRESHOLD = 2.0
    PEZ_THRESHOLD = 0.05

    obs_mask = np.array(obs_mask, dtype=bool)
    sz_mask = np.array(sz_mask, dtype=bool)
    # ctru = np.array(ctru, dtype=float)
    # ttru = np.array(ttru, dtype=float)

    nsamples, nreg = cinf.shape

    edgecolors = np.array(['r' if sz else 'b' for sz in sz_mask])
    facecolors = np.array(['r' if sz else 'b' for sz in sz_mask])
    facecolors[~obs_mask] = 'w'

    plt.figure(figsize=(16, nreg/3))

    # Plot c ------------------------------------------------------ #
    ax1 = plt.subplot2grid((1, 2), (0, 0))

    if nsamples > 1:
        plt.violinplot(cinf, positions=np.r_[:nreg], vert=False, widths=1.0, points=100)
    else:
        plt.scatter(c[0, :], np.r_[:nreg], c='blue', zorder=3)
    if ctru is not None:
        plt.scatter(ctru, np.r_[:nreg], s=120, facecolors=facecolors, edgecolors=edgecolors, lw=2, zorder=10)

    plt.xlabel("c")
    plt.xlim([-5, 5])

    if rhat is not None:
        for i in range(nreg):
            if rhat[i] > 0:
                color, weight = ('r', 'bold') if rhat[i] >= 1.05 else ('k', 'normal')
                plt.text(-4.8, i, "%5.2f" % rhat[i], ha='left', va='center', fontsize=14, color=color, weight=weight)

    plt.axvline(EZ_THRESHOLD, color='r')
    for i in range(nreg):
        pez = np.sum(cinf[:, i] > EZ_THRESHOLD)/nsamples
        color, weight = ('r', 'bold') if pez > PEZ_THRESHOLD else ('k', 'normal')
        plt.text(4.8, i, "%5.2f" % pez, ha='right', va='center', fontsize=14, color=color, weight=weight)
    # ------------------------------------------------------------- #

    # Plot t ------------------------------------------------------ #
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    subsample_factor = max(int(nsamples/500), 1)

    if nsamples > 1:
        for i in np.r_[:nsamples:subsample_factor]:
            mask = tinf[i, :] < tlim
            plt.scatter(tinf[i, mask], np.r_[:nreg][mask], s=80, c='k', alpha=0.015, edgecolor='none')
    else:
        mask = tinf[0, :] < tlim
        plt.scatter(tinf[0, mask], np.r_[:nreg][mask], s=80, c='b', alpha=1.0, edgecolor='none')

    if ttru is not None:
        plt.scatter(ttru, np.r_[:nreg], s=120, facecolors=facecolors, edgecolors=edgecolors, lw=2, zorder=10)

    for i in range(nreg):
        p_ns = np.sum(tinf[:, i] > tlim)/nsamples
        color = 'b' if p_ns > 0.5 else 'r'
        plt.text(1.1*tlim, i, f"{p_ns:4.2f}", ha='center', va='center', fontsize=14, color=color)

    plt.xlabel("t")
    plt.xlim([0, 1.2*tlim])
    # ------------------------------------------------------------- #

    # Axes -------------------------------------------------------- #
    for ax in [ax1, ax2]:
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_yticks(np.r_[:nreg])
        ax.set_ylim([-1, nreg])
        for reg in np.arange(0, nreg, 10):
            ax.axhline(reg, ls='--', lw=0.5, alpha=0.4, color='0.5')

    ax2.set_xticks(np.r_[np.arange(0, tlim, 10.), tlim], minor=False)

    if region_names is not None:
        ax2.set_yticklabels(region_names)
    # ------------------------------------------------------------- #

    # ------------------------------------------------------------- #
    if ez_hyp is not None:
        ax2.scatter(np.repeat(0, np.sum(ez_hyp)), np.r_[:nreg][ez_hyp], color='orange', marker='s', s=80, clip_on=False)
    # ------------------------------------------------------------- #

    plt.tight_layout()
    plt.savefig(img_file)
    plt.close()



def post_learn(data_file, summary_file, res_files, reg_name_file, output_file, ct_direc):
    nsamples, summary = io.parse_summary_csv(summary_file)
    res = io.parse_csv(res_files, merge=True)
    data = io.rload(data_file)

    plot_hyperparams(res, summary, output_file)

    region_names = list(np.genfromtxt(reg_name_file, dtype=str))

    # c-t plots
    os.makedirs(ct_direc)
    nsamples, nreg, nrec = res['t'].shape
    for i in range(nrec):
        sid = int(data['sids'][i])
        rid = int(data['rids'][i])

        img_file = os.path.join(ct_direc, f"ct_id{sid:03d}_{rid:02d}.pdf")

        reg_ns = np.array(data['reg_ns'][i], dtype=int)[:int(data['n_ns'][i])]
        reg_sz = np.array(data['reg_sz'][i], dtype=int)[:int(data['n_sz'][i])]
        t_sz = np.array(data['t_sz'][i], dtype=float)[:int(data['n_sz'][i])]
        rhat = [row[-1] for row in summary['c'][i, :, 0]]

        obs_mask = np.zeros(nreg, dtype=bool)
        obs_mask[reg_ns] = True
        obs_mask[reg_sz] = True

        sz_mask = np.zeros(nreg, dtype=bool)
        sz_mask[reg_sz] = True

        ttru = np.full(nreg, np.nan)
        ttru[reg_sz] = t_sz
        ttru[reg_ns] = data['t_lim']

        plot_ct(img_file, res['c'][:,:,i], res['t'][:,:,i], obs_mask, sz_mask,
                ttru=ttru, tlim=data['t_lim'], rhat=rhat, region_names=region_names)


def post_solo(data_file, summary_file, res_files, reg_name_file, img_file, ez_file=None, ground_truth_file=None):
    nsamples, summary = io.parse_summary_csv(summary_file)
    res = io.parse_csv(res_files, merge=True)
    data = io.rload(data_file)
    region_names = list(np.genfromtxt(reg_name_file, dtype=str))

    reg_ns = np.array(data['reg_ns'], dtype=int)
    reg_sz = np.array(data['reg_sz'], dtype=int)
    t_sz = np.array(data['t_sz'], dtype=float)

    nsamples, nreg = res['t'].shape
    obs_mask = np.zeros(nreg, dtype=bool)
    obs_mask[reg_ns] = True
    obs_mask[reg_sz] = True

    sz_mask = np.zeros(nreg, dtype=bool)
    sz_mask[reg_sz] = True

    ttru = np.full(nreg, np.nan)
    ttru[reg_sz] = t_sz
    ttru[reg_ns] = data['t_lim']

    # Leftout region
    leftout = int(data['leftout'])
    if leftout != -1:
        ttru[leftout] = max(0, min(data['tleftout'], data['t_lim']))
        sz_mask[leftout] = data['tleftout'] < data['t_lim']

    rhat = [row.R_hat for row in summary['c'][:, 0]]

    ez_mask = None
    if ez_file is not None:
        with open(ez_file) as fh:
            ez_orig = json.load(fh)
            ez_all = {int(subj[2:5]): data['i_ez'] for subj, data in ez_orig.items()}

        sid = int(data['sid'])
        ez_regs = ez_all.get(sid, [])
        ez_mask = np.zeros(nreg, dtype=bool)
        ez_mask[ez_regs] = True


    # Ground truth
    t_lim = data['t_lim']
    ctru = None
    if ground_truth_file is not None:
        t_first = np.min(t_sz)

        with open(ground_truth_file) as fh:
            ground_truth = json.load(fh)
        ttru = np.array(ground_truth['t'])
        ttru -= np.min(ttru[obs_mask]) - t_first
        ttru = np.minimum(ttru, t_lim)
        ctru = np.array(ground_truth['c'])
        sz_mask = ttru < t_lim

    # Plot
    plot_ct(img_file, res['c'], res['t'], obs_mask, sz_mask, ttru=ttru, ctru=ctru,
            tlim=data['t_lim'], rhat=rhat, region_names=region_names, ez_hyp=ez_mask)

    
    
def get_szobs(data, nt, tlim):
    ts = np.linspace(0, tlim, nt, endpoint=True)
    nreg = data['tinf'].shape[1]
    sz = np.full((nreg, nt), np.nan, dtype=float)
    for reg in data['reg_ns']:
        sz[int(reg)] = 0.
    for reg, tsz in zip(data['reg_sz'], data['t_sz']):
        sz[int(reg)] = ts >= tsz
    return sz

def get_szinf(data, nt, tlim):
    tinf = data['tinf']
    nreg = tinf.shape[1]
    ts = np.linspace(0, tlim, nt, endpoint=True)
    psz = np.zeros((nreg, nt))
    for i in range(nt):
        psz[:, i] = np.mean(tinf < ts[i], axis=0)
    return psz

def get_cdens(data):
    cinf = data['cinf']
    nsamples, nreg = cinf.shape
    nbins = 60
    binlims = [-3, 3]
    binsize = (binlims[-1] - binlims[0])/nbins
    cdens = np.zeros((nreg, nbins))
    for i in range(nreg):
        cdens[i] = np.histogram(cinf[:, i], bins=np.linspace(binlims[0], binlims[1], nbins+1))[0]
    cdens = cdens/(nsamples * binsize)
    return cdens, binlims    

    
def seizure_fig(sid, datafile, resfiles, surgery_file, subject_dir, region_name_file,
                variant='full', times=(40, 60, 80), views=('topdown', 'righleft'), label=True):
    
    nt = 1000
    tlim = 90.
    ez_threshold = 2.0
    
    data = plot3d.read_data(sid, datafile, resfiles, subject_dir)
    nreg = data['tinf'].shape[1]
    szobs = get_szobs(data, nt, tlim)
    szinf = get_szinf(data, nt, tlim)
    cdens, binlims = get_cdens(data)
    
    resection = np.full(nreg, np.nan, dtype=float)
    region_names = list(np.genfromtxt(region_name_file, usecols=(0,), dtype=str))
    with open(surgery_file) as fh:
        surgeries = json.load(fh)['data']
        surgeries = {int(subject[2:5]): data for subject, data in surgeries.items()}
        surgery = surgeries.get(sid, None)
        if surgery is not None:
            resection = np.zeros(nreg, dtype=float)
            for name, frac in surgery['resection'].items():
                index = region_names.index(name)
                resection[index] = frac
    
    # Create 3D images -----------------------------------------------------------------------------------
    imgs = {(t, view): plot3d.viz_network_scalar(data['pos'],
                                                 np.mean(data['tinf'] < t, axis=0),
                                                 data['w'], data['obsmask'], data['surf'],
                                                 view=view, cmap='bwr', vmin=0, vmax=1)
            for t in times for view in views}
    
    imgs_pez = {view: plot3d.viz_network_scalar(data['pos'], np.mean(data['cinf'] > ez_threshold, axis=0),
                                                data['w'], data['obsmask'], data['surf'],
                                                view=view, cmap='Reds', vmin=0, vmax=0.5)
                for view in views}
    
    imgs_res = {view: None if surgery is None else
                      plot3d.viz_network_scalar(data['pos'], resection,
                                                data['w'], data['obsmask'], data['surf'],
                                                view=view, cmap='Reds', vmin=0, vmax=1.)
                for view in views}
    # -----------------------------------------------------------------------------------------------------

    if variant == 'full':
        fig = plt.figure(figsize=(18, 7.3))
        gs = gridspec.GridSpec(2, 5, width_ratios=[0.9, 0.9, 1, 0.8, 1.2], height_ratios=[1, 0.04], wspace=0.2, hspace=0.22,
                               left=0.04, right=0.98, bottom=0.07, top=0.89)
    elif variant == 'paper': 
        fig = plt.figure(figsize=(15, 7.3))
        gs = gridspec.GridSpec(2, 4, width_ratios=[0.9, 0.9, 1, 0.7], height_ratios=[1, 0.04], wspace=0.2, hspace=0.22,
                               left=0.06, right=1.0, bottom=0.07, top=0.89)
    else:
        raise ValueError(f"Unexpected variant {variant}")
        
    # Input data ------------------------------------------------------------------------------------------
    ax0 = plt.subplot(gs[0, 0])
    cmap = cm.get_cmap('bwr')
    cmap.set_bad(color='0.7')
    plt.imshow(szobs, aspect='auto', cmap=cmap, vmin=0, vmax=1, origin='upper', extent=[0, tlim, nreg, 0])
    plt.xlim(0, tlim)
    plt.xlabel("Time [s]")
    plt.title("Observations")
    plt.ylabel("Region\n\n", fontsize=12)
    plot.add_brain_regions(ax0, pad=5, width=7, coord='display')
    plot.add_mask(ax0, data['obsmask'], width=8, pad=0, coord='display')

    ax0leg = plt.subplot(gs[1, 0])
    legend_elements = [patches.Patch(fc='b',   ec='k', label="Non-seizing"),
                       patches.Patch(fc='r',   ec='k', label="Seizing"),
                       patches.Patch(fc='0.7', ec='k', label="Hidden")]
    ax0leg.legend(handles=legend_elements, loc='center', ncol=3)
    plt.axis('off')

    # Inference results ----------------------------------------------------------------------------------
    ax1 = plt.subplot(gs[0, 1])
    im = plt.imshow(szinf, aspect='auto', cmap='bwr', origin='upper', extent=[0, tlim, nreg, 0])
    plt.xlim(0, tlim)
    plt.xlabel("Time [s]")
    plt.title("Inference")
    plot.add_brain_regions(ax1, pad=5, width=5, coord='display', labels=False)
    plot.add_mask(ax1, data['obsmask'], width=8, pad=0, coord='display')

    gs0 = gridspec.GridSpecFromSubplotSpec(1, 3, gs[1, 1:3], wspace=0., hspace=0., width_ratios=[1, 2, 1])
    ax1cb = plt.subplot(gs0[1])
    plt.colorbar(im, cax=ax1cb, orientation='horizontal', label="Recruitment probability")

    # 3D plots -------------------------------------------------------------------------------------------
    gs1 = gridspec.GridSpecFromSubplotSpec(3, 2, gs[0, 2], wspace=0.05, hspace=0.05)
    axes3dA = []
    for i, t in enumerate(times):    
        for j, view in enumerate(views):
            axes3dA.append(plt.subplot(gs1[i, j]))
            plt.imshow(imgs[(t, view)], interpolation='none')
            plt.xticks([]), plt.yticks([])
            plot3d.add_orientation(plt.gca(), view)
            if j == 0:
                plt.ylabel(f"t = {t:2.0f} s")

    # Excitability ----------------------------------------------------------------------------------------
    gs2top = gridspec.GridSpecFromSubplotSpec(1, 4, gs[0, 3], width_ratios=[8, 0.35, 1, 1], wspace=0.0)

    # Inferred excitability
    ax3 = plt.subplot(gs2top[0])
    im_cinf = plt.imshow(cdens, aspect='auto', origin='upper', extent=[binlims[0], binlims[1], nreg, 0],
                    cmap='Reds', vmin=0, vmax=2)
    plt.axvline(2.0, color='k', lw=0.5, ls=(0, (5, 10)))
    plt.xlim(binlims[0], binlims[1]); 
    plt.title("Inferred $c$ (density)", fontsize=10)
    plot.add_brain_regions(ax3, pad=5, width=5, coord='display', labels=False)
    
    # High excitability
    ax4 = plt.subplot(gs2top[2])
    im_pez = plt.imshow(np.mean(data['cinf'] > 2., axis=0)[:, None], aspect='auto', origin='upper', extent=[0, 1, nreg, 0],
                    cmap='Reds', vmin=0, vmax=0.5)
    plt.xlim(0, 1); plt.xticks([])
    plt.yticks([])
    plt.title("$p(c > c_h)$", fontsize=10, rotation='vertical', ha='center', va='bottom')
    
    plot.add_mask(ax3, data['obsmask'], width=8, pad=0, coord='display')
    
    # Resection
    if variant == 'full':
        ax5 = plt.subplot(gs2top[3])
        im_res = plt.imshow(resection[:, None], cmap='Reds', vmin=0, vmax=1, aspect='auto', origin='upper', extent=[0, 1, nreg, 0])
        plt.xlim(0, 1);
        plt.xticks([]); plt.yticks([])
        plt.title("Resect.", fontsize=10, rotation='vertical', ha='center', va='bottom')
    
    # Colorbars
    if variant == 'full':
        gs2bot = gridspec.GridSpecFromSubplotSpec(1, 7, gs[1, 3:5], width_ratios=[1, 3, 2, 3, 2, 3, 1], wspace=0.0)
        args = dict(orientation='horizontal')
        cb1 = plt.colorbar(im_cinf, cax=plt.subplot(gs2bot[1]), ticks=[0., 1., 2.], **args)  
        cb2 = plt.colorbar(im_pez,  cax=plt.subplot(gs2bot[3]), ticks=[0., 0.25, 0.5], **args)
        cb3 = plt.colorbar(im_res,  cax=plt.subplot(gs2bot[5]), ticks=[0., 0.5, 1.], **args)
        cb1.set_label("Inferred $c$")
        cb2.set_label("$p(c > c_h)$")
        cb3.set_label("Resection")
    elif variant == 'paper':
        gs2bot = gridspec.GridSpecFromSubplotSpec(1, 6, gs[1, 3], width_ratios=[2, 4, 2, 0.35, 1, 1], wspace=0.0)
        args = dict(orientation='horizontal')
        axcb1 = plt.subplot(gs2bot[1])
        axcb2 = plt.subplot(gs2bot[4])
        cb1 = plt.colorbar(im_cinf, cax=axcb1, ticks=[0., 1., 2.], **args)  
        cb2 = plt.colorbar(im_pez,  cax=axcb2, ticks=[0, 0.5], **args)
        axcb2.set_xticklabels(["0", "0.5"])    
    
    # 3D plots of epileptogenicity and resection
    if variant == 'full':
        gs3 = gridspec.GridSpecFromSubplotSpec(4, 2, gs[0, 4], wspace=0.05, hspace=0.05, height_ratios=[0.3,1,1,0.3])
        axes3dB = []
        for i, (title, imgs) in enumerate([("$p(c > c_h)$", imgs_pez), ("Resection", imgs_res)]):
            for j, view in enumerate(views):
                if imgs[view] is None:
                    continue
                axes3dB.append(plt.subplot(gs3[i+1, j]))
                plt.imshow(imgs[view], interpolation='none')
                plt.xticks([]), plt.yticks([])
                plot3d.add_orientation(plt.gca(), view)
                if j == 0:
                    plt.ylabel(title)
     
    if variant == 'full':
        return fig, [ax0, ax1, axes3dA, ax3, ax4, ax5, axes3dB]
    elif variant == 'paper':
        if label:
            plot.add_panel_letters(fig, [ax0, ax1, axes3dA[0], ax3], fontsize=22, 
                               xpos=[-0.05, -0.05, -0.1, -0.1], ypos=[1.02, 1.02, 1.06, 1.02])
        return fig, [ax0, ax1, axes3dA, ax3, ax4]


def plot_solo_compact(sid, datafile, resfiles, statuses, surgery_file, subject_dir, region_name_file, img_file):
    resfiles = [r for r, s in zip(resfiles, statuses) if int(open(s).read().strip())]
    if len(resfiles) == 0:
        plt.figure()
    else:
        fig, axes = seizure_fig(sid, datafile, resfiles, surgery_file, subject_dir, region_name_file)
    plt.savefig(img_file)
