"""Figure styling. Optimization and gradients remain in the notebook."""

from cycler import cycler
import matplotlib.pyplot as plt

FTE_BLUE = '#30A2DA'
FTE_RED = '#FC4F30'
FTE_YELLOW = '#E5AE38'
FTE_GREEN = '#6D904F'
FTE_GRAY = '#8B8B8B'
FTE_PURPLE = '#810F7C'
FTE_COLORS = [FTE_BLUE, FTE_RED, FTE_YELLOW, FTE_GREEN, FTE_GRAY, FTE_PURPLE]


def configure_plots():
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({
        'axes.prop_cycle': cycler(color=FTE_COLORS),
        'axes.titlesize': 16, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.labelsize': 12,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'figure.titlesize': 16,
        'figure.figsize': (9, 5), 'figure.dpi': 110,
        'lines.linewidth': 1.2,
        'lines.markersize': 6, 'lines.markeredgewidth': 0.8,
    })


import numpy as np
import seaborn as sns

def plot_tips(values, mean=None, median=None, rug_only=False, zoom=False):
    fig, ax = plt.subplots(figsize=(9, 3 if rug_only else 5))
    if rug_only:
        sns.rugplot(x=values, height=0.45, ax=ax, color=FTE_BLUE)
        ax.set(ylim=(0, 1), yticks=[], title='Each tick represents one restaurant bill')
    else:
        sns.histplot(x=values, stat='density', kde=True, ax=ax, color=FTE_BLUE, alpha=0.35)
        sns.rugplot(x=values, ax=ax, color=FTE_GRAY)
        if mean is not None:
            ax.axvline(mean, color=FTE_RED, linestyle='--', label=f'Mean = {mean:.2f}%')
            ax.axvline(median, color=FTE_GREEN, linestyle=':', label=f'Median = {median:.2f}%')
            ax.legend()
        ax.set(ylabel='Density per percentage point', title='Tip percentages cluster around the middle')
    if zoom:
        ax.set_xlim(10, 20)
        ax.set_title('A closer view of the 10–20% interval')
    ax.set_xlabel('Tip percentage (%)')
    fig.tight_layout()
    plt.show()


def plot_line_path(path, losses, x, y, loss_fn, scaled=False, sampled_losses=None):
    # Evaluate the notebook's supplied loss function only to draw its surface.
    lo = np.minimum(path.min(axis=0), [-1, -1] if scaled else [-2, -0.2])
    hi = np.maximum(path.max(axis=0), [1, 1] if scaled else [2, 0.3])
    margin = (hi-lo)*0.08
    a = np.linspace(lo[0]-margin[0], hi[0]+margin[0], 75)
    b = np.linspace(lo[1]-margin[1], hi[1]+margin[1], 75)
    A, B = np.meshgrid(a, b)
    Z = np.array([[loss_fn(t0, t1, x, y) for t0 in a] for t1 in b])
    fig = plt.figure(figsize=(12, 10), layout='constrained')
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax1.scatter(x, y, color=FTE_BLUE, s=18, alpha=0.6)
    order = np.argsort(x)
    ax1.plot(x[order], path[-1, 0]+path[-1, 1]*x[order], color=FTE_RED, label='Current fitted line')
    ax1.set(title=f'Intercept {path[-1,0]:.3f}, slope {path[-1,1]:.3f}', xlabel='Standardized bill' if scaled else 'Total bill (USD)', ylabel='Standardized tip' if scaled else 'Tip (USD)')
    ax1.legend()
    ax2.plot(np.arange(len(losses)), losses, color=FTE_BLUE, label='Full-data MSE')
    if sampled_losses is not None:
        ax2.plot(np.arange(len(sampled_losses)), sampled_losses, color=FTE_GRAY, alpha=0.45, linestyle=':', label='Selected-observation loss before update')
    ax2.set(title='Monitor the same full-data loss', xlabel='Updates completed', ylabel='MSE (standardized units²)' if scaled else 'MSE (USD²)')
    ax2.set_yscale('log')
    ax2.legend()
    levels = np.geomspace(max(Z.min(), 0.001), Z.max(), 16)
    ax3.contour(A, B, Z, levels=np.unique(levels), cmap='viridis', linewidths=1)
    ax3.plot(path[:,0], path[:,1], '-', color=FTE_RED, linewidth=1.2)
    ax3.scatter(path[0,0], path[0,1], s=55, marker='s', color=FTE_GRAY, label='Start', zorder=5)
    ax3.scatter(path[-1,0], path[-1,1], s=110, marker='*', color=FTE_YELLOW, edgecolor='black', label='Current', zorder=6)
    ax3.set(title='Updates across the loss contours', xlabel='θ₀ (intercept)', ylabel='θ₁ (slope)')
    ax3.legend()
    ax4.plot_surface(A, B, Z, cmap='viridis', alpha=0.6, rstride=3, cstride=3)
    # Keep the original surface view while limiting drawing cost for long histories.
    keep = np.unique(np.r_[np.linspace(0, len(path)-1, min(len(path), 300), dtype=int), len(path)-1])
    ax4.plot(path[keep,0], path[keep,1], losses[keep], color=FTE_RED, linewidth=1.2)
    ax4.scatter(path[-1,0], path[-1,1], losses[-1], color=FTE_YELLOW, marker='*', s=100)
    ax4.set(title='The same full-data loss as a surface', xlabel='θ₀', ylabel='θ₁', zlabel='MSE')
    ax4.tick_params(labelsize=11)
    plt.show()


def plot_training_curves(curves, reference, title, xlabel='Updates completed', ylabel='Training MSE (kPa²)', zoom_after=None, log_updates=False):
    fig, axes = plt.subplots(1, 2 if zoom_after is not None else 1,
                             figsize=(12, 4.5) if zoom_after is not None else (9, 4.5), squeeze=False)
    for column, ax in enumerate(axes[0]):
        for (label, values), color, style in zip(curves.items(), [FTE_BLUE, FTE_GREEN, FTE_YELLOW, FTE_RED], ['-', '--', ':', '-.']):
            start = zoom_after if column else 0
            ax.plot(np.arange(start, len(values)), values[start:], label=label, color=color, linestyle=style)
        ax.axhline(reference, color=FTE_GRAY, linestyle=':', label='Least-squares training minimum')
        ax.set(title=f'After epoch {zoom_after}: a closer view' if column else title,
               xlabel=xlabel, ylabel=ylabel)
        if column == 0:
            ax.set_yscale('log')
        if log_updates:
            ax.set_xscale('symlog', linthresh=1)
            ax.set_xlim(left=0)
            ax.set_xticks([0, 1, 10, 100, 1000], ['0', '1', '10', '100', '1000'])
            ax.set_xlabel('Updates completed (log scale after 1)')
        ax.legend()
    fig.tight_layout()
    plt.show()


def plot_gradient_noise(estimates, full_gradient):
    fig, axes = plt.subplots(1, len(estimates), figsize=(12, 4), sharex=True)
    for ax, (size, values) in zip(axes, estimates.items()):
        ax.hist(values, bins=20, color=FTE_BLUE, edgecolor='white')
        ax.axvline(full_gradient, color=FTE_RED, linestyle='--', label='Full-data gradient')
        ax.set(title=f'Batch size {size}', xlabel='Estimated slope gradient', ylabel='Number of sampled batches')
    axes[0].legend()
    fig.tight_layout()
    plt.show()


def plot_nonconvex(grid, values, paths, path_values, chord):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(grid, values, color=FTE_GRAY, label='Objective')
    for start, color, style in zip(paths, [FTE_BLUE, FTE_RED], ['-', '--']):
        axes[0].plot(paths[start], path_values[start], style, marker='o', color=color,
                     markersize=5.5, markeredgecolor='white', markeredgewidth=0.8,
                     zorder=3, label=f'Start {start:g}')
        axes[0].scatter(paths[start][-1], path_values[start][-1], marker='*', s=150, color=color, edgecolor='black', zorder=5)
    axes[0].set(title='Different starts reach different minima', xlabel='Parameter w', ylabel='Objective f(w)')
    axes[0].legend()
    axes[1].plot(grid, values, color=FTE_BLUE, label='Objective')
    axes[1].plot(chord[0], chord[1], 'o--', color=FTE_RED, markeredgecolor='white',
                 markeredgewidth=0.8, label='Chord between two points')
    axes[1].set(title='One chord disproves convexity', xlabel='Parameter w', ylabel='Objective f(w)')
    axes[1].legend()
    fig.tight_layout()
    plt.show()
