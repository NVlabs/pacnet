"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt


LINE_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
LINE_STYLES = ('solid', 'dashed', 'dashdot', 'dotted')


def smooth_plot(xs, ys, smooth=5.0, axis=None, *args, **kwargs):
    min_window = 3
    if smooth > 0 and len(xs) > min_window * 100 / smooth:
        window = int(len(xs) * smooth // 100)
        f = np.repeat(1.0, window) / window
        ys = np.convolve(ys, f, 'valid')
        xs = xs[(window//2):(window//2)+len(ys)]
    # TODO: fix issue with empty strings and strings starting with '_'
    if 'label' in kwargs and (kwargs['label'].startswith('_') or kwargs['label'] == ''):
        kwargs['label'] = ' ' + kwargs['label']
    if axis is None:
        plt.plot(xs, ys, *args, **kwargs)
    else:
        axis.plot(xs, ys, *args, **kwargs)


def remove_common_prefix_suffix(array_of_str):
    if len(array_of_str) == 1:
        return array_of_str
    remove_head, remove_tail = 0, 0
    while True:
        v = array_of_str[0][remove_head]
        if all(s[remove_head] == v for s in array_of_str):
            remove_head += 1
        else:
            break
    while True:
        v = array_of_str[0][len(array_of_str[0]) - remove_tail - 1]
        if all(s[len(s) - remove_tail - 1] == v for s in array_of_str):
            remove_tail += 1
        else:
            break
    return [s[remove_head:len(s) - remove_tail] for s in array_of_str]


def parse_and_plot(paths, output=None, plots=None, labels=None, reorder=None, trials=1,
                   smooth=5.0, num_col=0, subplot_size=5, start_x=0.0, end_x=-1.0, legend='best'):
    with open(paths[0], 'r') as f:
        r = f.readline()
    xlabel = r.strip().split(',')[0]
    plots_all = r.strip().split(',')[1:]
    if plots:
        plot_dict = {p: i+1 for i, p in enumerate(plots_all)}
        plot_cols = tuple(-1 if p == '-' else plot_dict[p] for p in plots)
    else:
        plots = plots_all
        plot_cols = tuple(range(1, len(plots) + 1))
    if not labels:
        labels = remove_common_prefix_suffix(paths)[::trials]
    assert(len(labels) == len(set(labels)))
    paths = [paths[i*trials:(i+1)*trials] for i in range(len(labels))]
    if reorder:
        if len(reorder) == 1 and reorder[0] == 'str':
            reorder = sorted(labels)
        paths = [paths[labels.index(l)] for l in reorder]
        labels = reorder

    runs_data = []
    for ps in paths:
        tmp = []
        for p in ps:
            data = np.genfromtxt(p, delimiter=',', usecols=(0,)+tuple(filter(lambda v: v != -1, plot_cols)), skip_header=1)
            data = data.reshape(-1, data.shape[-1])
            start = np.where(data[:, 0] >= start_x)[0][0]
            end = None
            if end_x > 0:
                end_idxs = np.where(data[:, 0] > end_x)[0]
                if len(end_idxs) > 0:
                    end = end_idxs[0]
            data = data[start:end]
            tmp.append(data)
        _len = min(len(d) for d in tmp)
        runs_data.append(np.mean([d[:_len] for d in tmp], axis=0))

    if output:
        plt.switch_backend('agg')

    if num_col <= 0:
        num_col = int(np.ceil(np.sqrt(len(plots))))
    num_row = (len(plots) - 1) // num_col + 1
    fig, axes = plt.subplots(num_row, num_col, squeeze=False, figsize=(num_col * subplot_size, num_row * subplot_size))
    p_idx = 0
    for plabel, ax in zip(plots, axes.flat):
        if plabel == '-':
            continue
        p_idx += 1
        # plt.subplot(num_row, num_col, p + 1)
        for r, rdata in enumerate(runs_data):
            valid_mask = ~np.isnan(rdata[:, p_idx])
            if any(valid_mask):
                smooth_plot(rdata[valid_mask, 0], rdata[valid_mask, p_idx], smooth, ax,
                            color=LINE_COLORS[r % len(LINE_COLORS)],
                            linestyle=LINE_STYLES[r // len(LINE_COLORS)],
                            linewidth=1.5, label=labels[r])
        ax.set_xlabel(xlabel)
        ax.set_title(plabel)
        ax.grid(True)
        if legend != 'off' and len(labels) > 1:
            ax.legend(loc=legend)
    for ax in axes.flat[len(plots):]:
        fig.delaxes(ax)
    fig.tight_layout()

    if output:
        plt.savefig(output)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse log files (csv format) and make plots',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('paths', nargs='+', help='one or more log files (each subplot will have #paths curves)')
    parser.add_argument('--output', type=str, default='', help='place to save figure (show figure if blank)')
    parser.add_argument('--plots', nargs='+', default=None, help='use select plots instead of one for each column')
    parser.add_argument('--labels', nargs='+', default=None, help='use labels instead of file names')
    parser.add_argument('--average', type=int, default=1, help='plot average curves')
    parser.add_argument('--reorder', nargs='+', default=None, help='order the label legends')
    parser.add_argument('--smooth', type=float, default=5.0, help='smoothing level (0-100)')
    parser.add_argument('--num-col', type=int, default=0, help='number of columns (0 - auto)')
    parser.add_argument('--subplot-size', type=int, default=5, help='width and height of each subplot')
    parser.add_argument('--legend', type=str, default='best', help='place to put legend')
    parser.add_argument('--xlim',  nargs='+', default=None, help='xmin (xmax)')

    args = parser.parse_args()

    start_x = 0.0 if not args.xlim else float(args.xlim[0])
    end_x = -1.0 if (not args.xlim or len(args.xlim) < 2) else float(args.xlim[1])

    parse_and_plot(args.paths, output=args.output, plots=args.plots, labels=args.labels, reorder=args.reorder,
                   trials=args.average, smooth=args.smooth, num_col=args.num_col, subplot_size=args.subplot_size,
                   start_x=start_x, end_x=end_x, legend=args.legend.replace('_', ' '))
