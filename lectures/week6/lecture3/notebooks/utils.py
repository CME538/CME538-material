"""Display helpers for CME538 Week 6.

The notebooks retain data preparation, fitting, transformations and evaluation.
These functions only arrange and label supplied data and computed results.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

FTE_BLUE = '#30A2DA'
FTE_RED = '#FC4F30'
FTE_YELLOW = '#E5AE38'
FTE_GREEN = '#6D904F'
FTE_GRAY = '#8B8B8B'
FTE_PURPLE = '#810F7C'
FTE_COLORS = [FTE_BLUE, FTE_RED, FTE_YELLOW, FTE_GREEN, FTE_GRAY, FTE_PURPLE]

# Semi-transparent observations make overlapping points visible.
SCATTER_STYLE = {'s': 25, 'alpha': 0.5, 'edgecolor': 'face', 'linewidths': 0.5}

def configure_notebook():
    """Set course typography and thin lines after applying the base style."""
    plt.style.use('fivethirtyeight')
    sns.set_palette(FTE_COLORS)
    plt.rcParams.update({
        'axes.prop_cycle': cycler(color=FTE_COLORS),
        'axes.titlesize': 16, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'figure.titlesize': 16,
        'figure.figsize': (9, 5), 'figure.dpi': 110,
        'lines.linewidth': 1.2, 'lines.markersize': 6,
        'lines.markeredgewidth': 0.8, 'lines.markeredgecolor': 'white',
        'grid.linewidth': 0.6, 'grid.alpha': 0.45, 'axes.linewidth': 0.7,
        'patch.linewidth': 0.5, 'legend.frameon': False,
    })
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 12)
    pd.set_option('display.width', 110)

def finish(fig):
    fig.tight_layout()
    plt.show()
    plt.close(fig)

def spatial_temperature_map(data):
    """Display the supplied synthetic sensor measurements in local coordinates."""
    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(data['east_km'], data['north_km'], c=data['temperature_c'],
                        cmap='viridis', s=48, edgecolors='white', linewidths=0.35, alpha=0.5)
    ax.set(title='Nearby sensors share a temperature pattern',
           xlabel='East (km)', ylabel='North (km)', xlim=(0, 10), ylim=(0, 10))
    ax.set_aspect('equal')
    fig.colorbar(points, ax=ax, label='Synthetic temperature (°C)', shrink=0.88)
    finish(fig)

def spatial_neighbour_comparison(left, right, shuffled_left, shuffled_right):
    """Compare already calculated neighbouring pairs, using the same axis limits."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, x, y, title, color in zip(
        axes, [left, shuffled_left], [right, shuffled_right],
        ['Original locations', 'Temperatures shuffled'], [FTE_BLUE, FTE_GRAY]):
        ax.scatter(x, y, s=18, alpha=0.48, color=color, edgecolors='white', linewidths=0.25)
        low, high = min(np.min(left), np.min(right)), max(np.max(left), np.max(right))
        ax.plot([low, high], [low, high], '--', color=FTE_GRAY, linewidth=0.8)
        ax.set(title=title, xlabel='Sensor temperature (°C)',
               ylabel='Eastern neighbour temperature (°C)')
    finish(fig)

def spatial_split_maps(data, splits):
    """Show the supplied index assignments without choosing or fitting a split."""
    fig, axes = plt.subplots(1, len(splits), figsize=(11, 5), sharex=True, sharey=True)
    for ax, (name, (train_ids, test_ids)) in zip(np.atleast_1d(axes), splits.items()):
        for ids, color, marker, label in [(train_ids, FTE_BLUE, 'o', 'Training'),
                                          (test_ids, FTE_RED, 's', 'Evaluation')]:
            rows = data.loc[ids]
            ax.scatter(rows['east_km'], rows['north_km'], s=23, color=color,
                       marker=marker, label=label, edgecolors='white', linewidths=0.35, alpha=0.5)
        if name == 'Unobserved eastern area':
            ax.axvline(7, color=FTE_GRAY, linestyle='--', linewidth=0.8)
        ax.set(title=name, xlabel='East (km)', ylabel='North (km)',
               xlim=(0, 10), ylim=(0, 10))
        ax.set_aspect('equal')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    finish(fig)

def scatter_fit(x, y, xlabel, ylabel, title, curves=None, residual_segments=False):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=25, alpha=0.5, color=FTE_BLUE, label='Observations', zorder=3)
    for i, (label, (gx, gy)) in enumerate((curves or {}).items()):
        ax.plot(gx, gy, color=[FTE_RED, FTE_GREEN, FTE_GRAY][i % 3],
                linestyle=['-', '--', ':'][i % 3], label=label)
    if residual_segments and curves:
        gx, gy = next(iter(curves.values()))
        ax.vlines(x, y, gy, color=FTE_GRAY, linewidth=0.7, alpha=0.6)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    finish(fig)

def lines(x, series, xlabel, ylabel, title, markers=True, log=False):
    fig, ax = plt.subplots()
    for i, (label, y) in enumerate(series.items()):
        ax.plot(x, y, marker=['o', 's', '^', 'D'][i % 4] if markers else None,
                linestyle=['-', '--', ':', '-.'][i % 4], label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if log: ax.set_yscale('symlog', linthresh=1)
    ax.legend()
    finish(fig)

def residual_plot(fitted, residuals, title, unit, smooth=False):
    fig, ax = plt.subplots()
    ax.scatter(fitted, residuals, color=FTE_BLUE, s=25, alpha=0.5)
    ax.axhline(0, color=FTE_GRAY, linestyle='--', linewidth=0.9)
    if smooth:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.6)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=FTE_RED, label='Local smooth')
        ax.legend()
    ax.set(title=title, xlabel=f'Fitted value ({unit})', ylabel=f'Observed − fitted ({unit})')
    finish(fig)

def heatmap(correlation, title, labels=None):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('course_diverging', [FTE_BLUE, '#F3F3F3', FTE_RED])
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap=cmap, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                xticklabels=labels if labels is not None else 'auto',
                yticklabels=labels if labels is not None else 'auto',
                cbar_kws={'label': 'Pearson correlation', 'shrink': 0.82})
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=40)
    ax.tick_params(axis='y', rotation=0)
    finish(fig)

def bars(values, title, ylabel, baseline=None):
    fig, ax = plt.subplots()
    b = ax.bar(list(values), list(values.values()), color=FTE_COLORS[:len(values)])
    ax.bar_label(b, fmt='%.2f', padding=4)
    ax.set(title=title, ylabel=ylabel)
    ax.set_ylim(0, max(values.values()) * 1.2)
    if baseline is not None:
        ax.axhline(baseline, color=FTE_GRAY, linestyle='--', label='Expected chance accuracy')
        ax.set_ylim(0, 1)
        ax.legend()
    finish(fig)

def polynomial_panels(x, y, grid, predictions, xlabel, ylabel, truth=None):
    count = len(predictions)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4.5), squeeze=False, sharey=True)
    for ax, (label, pred) in zip(axes.flat, predictions.items()):
        ax.scatter(x, y, s=22, alpha=0.5, color=FTE_BLUE)
        ax.plot(grid, pred, color=FTE_RED, label='Fitted curve')
        if truth is not None: ax.plot(grid, truth, '--', color=FTE_GRAY, label='Known signal')
        ax.set(title=label, xlabel=xlabel)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, -1].legend()
    span = max(float(np.max(y) - np.min(y)), 1)
    for ax in axes.flat: ax.set_ylim(np.min(y) - 0.25 * span, np.max(y) + 0.25 * span)
    fig.text(0.5, -0.015, 'Shared axes focus on the observed response range. Extreme fitted curves continue beyond the axes.',
             ha='center', fontsize=10)
    finish(fig)

def binned_scatter(x, y, centers, means, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=24, alpha=0.4, color=FTE_BLUE, label='Observations')
    ax.plot(centers, means, 'o-', color=FTE_RED, label='Local mean', zorder=4)
    ax.set(xlabel=xlabel, ylabel=ylabel, title='Averages reveal the trend across observations')
    ax.legend()
    finish(fig)

def compare_outlier(x, y, x_out, y_out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, xx, yy, title in zip(axes, [x, x_out], [y, y_out], ['Original cloud', 'One added outlier']):
        ax.scatter(xx, yy, s=25, alpha=0.55)
        ax.set(xlabel='Constructed x', ylabel='Constructed y', title=title)
    axes[1].scatter(x_out[-1], y_out[-1], color=FTE_RED, marker='D', s=65, zorder=4, alpha=0.5)
    finish(fig)

def two_histograms(a, b, names, xlabel, title):
    fig, ax = plt.subplots()
    bins = np.linspace(min(np.min(a), np.min(b)), max(np.max(a), np.max(b)), 25)
    ax.hist(a, bins=bins, alpha=0.5, label=names[0], color=FTE_BLUE, density=True)
    ax.hist(b, bins=bins, alpha=0.4, label=names[1], color=FTE_YELLOW, hatch='//', density=True)
    ax.set(title=title, xlabel=xlabel, ylabel='Density')
    ax.legend()
    finish(fig)

def split_scatter(train_x, train_y, other_x, other_y, other_label='Validation'):
    fig, ax = plt.subplots()
    ax.scatter(train_x, train_y, s=25, alpha=0.5, label='Train', color=FTE_BLUE)
    ax.scatter(other_x, other_y, s=35, alpha=0.6, label=other_label, marker='x', color=FTE_RED)
    ax.set(title='Different rows serve different evaluation roles',
           xlabel='Assists per game', ylabel='Points per game')
    ax.legend()
    finish(fig)

def forward_split(train, test):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(train['t'], train['y'], label='Train (past)', color=FTE_BLUE, linewidth=0.8)
    ax.plot(test['t'], test['y'], '--', label='Test (future)', color=FTE_RED, linewidth=0.8)
    ax.axvline(test['t'].iloc[0], color=FTE_GRAY, linestyle=':')
    ax.set(title='Training ends before the forecast period begins', xlabel='Time step', ylabel='Constructed response')
    ax.legend()
    finish(fig)

def time_folds(folds):
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, (train, validation) in enumerate(folds):
        ax.plot(train, np.full(len(train), k), '.', color=FTE_BLUE, markersize=5, markeredgewidth=0)
        ax.plot(validation, np.full(len(validation), k), 'x', color=FTE_RED, markersize=5,
                markeredgecolor=FTE_RED, markeredgewidth=0.9)
    ax.set_yticks(range(len(folds)), [f'Fold {k + 1}' for k in range(len(folds))])
    ax.set(xlabel='Row index in time order', title='Past training rows precede each validation block')
    ax.plot([], [], '.', color=FTE_BLUE, markersize=7, markeredgewidth=0, label='Training')
    ax.plot([], [], 'x', color=FTE_RED, markeredgecolor=FTE_RED, markeredgewidth=0.9, label='Validation')
    ax.legend()
    finish(fig)


def population_shift(guards, centers, grid, prediction):
    fig, ax = plt.subplots()
    ax.scatter(guards['3PA'], guards['PTS'], alpha=0.4, s=25, label='Point guards', color=FTE_BLUE)
    ax.scatter(centers['3PA'], centers['PTS'], alpha=0.5, s=35, label='Centers', color=FTE_YELLOW, marker='^')
    ax.plot(grid['3PA'], prediction, color=FTE_RED, label='Fit using point guards')
    ax.set(title='The scoring relationship also differs by position', xlabel='Three-point attempts per game', ylabel='Points per game')
    ax.legend()
    finish(fig)

# Display routines retained from the canonical lecture examples.

def original_l3_plot_7(X_test, X_train, y_test, y_train):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(x = X_train['AST'], y = y_train['PTS'], label='Train', **SCATTER_STYLE)
    sns.scatterplot(x = X_test['AST'], y = y_test['PTS'], label='Test', ax=ax, **SCATTER_STYLE)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Average Assists Per Game, AST', fontsize=12)
    ax.set_ylabel('Average Points Per Game, PTS', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('Training players and held-out test players', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")
