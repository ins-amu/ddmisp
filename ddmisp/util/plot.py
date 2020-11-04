import itertools

import matplotlib.pyplot as plt
from matplotlib import lines
import numpy as np


BRAIN_REGIONS = [
    ('LFr',     ( 0, 26), 'tab:blue'),
    (' LCi',    (27, 31), 'tab:orange'),
    ('LIn    ', (32, 33), 'tab:green'),
    ('LTe',     (34, 53), 'tab:red'),
    ('LPa',     (54, 63), 'tab:purple'),
    ('LOc',     (64, 71), 'tab:olive'),
    ('LSc',     (72, 80), 'tab:cyan'),

    ('RFr',     ( 81, 107), 'tab:blue'),
    (' RCi',    (108, 112), 'tab:orange'),
    ('RIn    ', (113, 114), 'tab:green'),
    ('RTe',     (115, 134), 'tab:red'),
    ('RPa',     (135, 144), 'tab:purple'),
    ('ROc',     (145, 152), 'tab:olive'),
    ('RSc',     (153, 161), 'tab:cyan'),
]

def add_brain_regions(ax, labels=True, width=0.02, pad=0.02, coord='axes', fontsize=7):
    """
        width: in 'coord' coordinates
        pad:   in 'coord' coordinates
        coord: either 'data', axes', or 'display'
    """

    plt.sca(ax)
    plt.yticks([])
    plt.ylim(162, 0)

    if coord == 'data':
        x1, x2 = -width-pad, -pad
    elif coord == 'axes':
        axis_to_data = ax.transAxes + ax.transData.inverted()
        (x1, _), (x2, _) = axis_to_data.transform([(-pad-width, 0), (-pad, 0)])
    elif coord == 'display':
        display_to_data = ax.transData.inverted()
        (x0, y0), (x1, y1), (x2, y2) = display_to_data.transform([(0, 0), (-pad-width, 0), (-pad, 0)])
        x1 -= x0
        x2 -= x0
    else:
        raise ValueError(f"Unknown coordinate system: {coord}")

    xleft, xright = ax.get_xlim()
    x1 += xleft
    x2 += xleft

    for label, regrange, color in BRAIN_REGIONS:
        plt.fill_between([x1, x2], [regrange[0], regrange[0]], [regrange[1]+1, regrange[1]+1],
                         clip_on=False, color=color, lw=0)
        if labels:
            plt.text(x1, np.mean(regrange), label, ha='right', va='center', rotation='vertical',
                     fontsize=fontsize)


class Background():
    def __init__(self, fig=None, visible=False, spacing=0.1, linecolor='0.5', linewidth=1):
        if fig is not None:
            plt.scf(fig)
        ax = plt.axes([0,0,1,1], facecolor=None, zorder=-1000)
        plt.xticks(np.arange(0, 1 + spacing/2., spacing))
        plt.yticks(np.arange(0, 1 + spacing/2., spacing))
        plt.grid()
        if not visible:
            plt.axis('off')
        self.axes = ax
        self.linecolor = linecolor
        self.linewidth = linewidth

    def vline(self, x, y0=0, y1=1, **args):
        defargs = dict(color=self.linecolor, linewidth=self.linewidth)
        defargs.update(args)
        self.axes.add_line(lines.Line2D([x, x], [y0, y1], **defargs))

    def hline(self, y, x0=0, x1=1, **args):
        defargs = dict(color=self.linecolor, linewidth=self.linewidth)
        defargs.update(args)
        self.axes.add_line(lines.Line2D([x0, x1], [y, y], **defargs))
        
    def labels(self, xs, ys, fontsize=30):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        assert len(xs) == len(ys)
        for x, y, letter in zip(xs, ys, letters):
            self.axes.text(x, y, letter, transform=self.axes.transAxes, size=fontsize, 
                           weight='bold', ha='left', va='bottom')


def add_panel_letters(fig, axes=None, fontsize=30, xpos=-0.04, ypos=1.05):
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if axes is None:
        axes = fig.get_axes()

    if type(xpos) == float:
        xpos = itertools.repeat(xpos)
    if type(ypos) == float:
        ypos = itertools.repeat(ypos)

    for i, (ax, x, y) in enumerate(zip(axes, xpos, ypos)):
        ax.text(x, y, labels[i],
                transform=ax.transAxes, size=fontsize, weight='bold')


def axtext(ax, text, **args):
    defargs = {'fontsize': 14, 'ha': 'center', 'va': 'center'}
    defargs.update(args)
    plt.text(0.5, 0.5, text, **defargs)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.axis('off')


def add_mask(ax, mask, width=0.02, pad=0.02, coord='axes'):
    plt.sca(ax)

    if coord == 'data':
        data_to_axis = (ax.transAxes + ax.transData.inverted()).inverted()
        (x1, _), (x2, _) = data_to_axis.transform([(0, 0), (1, 0)])
        scale = x2 - x1
    elif coord == 'axes':
        scale = 1
    elif coord == 'display':
        display_to_axis = ax.transAxes.inverted()
        (x1, _), (x2, _) = display_to_axis.transform([(0, 0), (1, 0)])
        scale = x2 - x1
    else:
        raise ValueError(f"Unknown coordinate system: {coord}")


    ax_to_fig = ax.transAxes + ax.figure.transFigure.inverted()
    c1, c2 = ax_to_fig.transform([(1 + scale*pad, 0), (1 + scale*(pad+width), 1)])
    ax2 = plt.gcf().add_axes([c1[0], c1[1], c2[0]-c1[0], c2[1]-c1[1]])
    plt.imshow(mask[:, None], vmin=0, vmax=1, cmap='Greys', aspect='auto', origin='upper')
    plt.xticks([]); plt.yticks([])
    plt.sca(ax)
