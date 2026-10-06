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


def plot_point_losses(thetas, observation, absolute=None, squared=None, wide=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    if absolute is not None:
        ax.plot(thetas, absolute, color=FTE_BLUE, label='Absolute loss', linestyle='-')
    if squared is not None:
        ax.plot(thetas, squared, color=FTE_RED, label='Squared loss', linestyle='--')
    ax.axvline(observation, color=FTE_GRAY, linestyle=':', label=f'Observation = {observation}%')
    both = absolute is not None and squared is not None
    title = 'Squared loss gives large errors more weight' if both else ('Absolute loss grows linearly' if squared is None else 'Squared loss grows quadratically')
    ax.set(title=title, xlabel='Constant prediction θ (%)', ylabel='Loss value (units differ)')
    if both:
        ax.set_ylim(0, 1200 if wide else 35)
    else:
        ax.set_ylabel('Absolute error (percentage points)' if squared is None else 'Squared error (percentage points²)')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_average_loss(values, fit, squared=False, optimum=False, coarse=False):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fit['thetas'], fit['losses'], 'o-' if coarse else '-', color=FTE_BLUE,
            markersize=6, markeredgecolor='white', markeredgewidth=0.8, zorder=3)
    sns.rugplot(x=values, height=0.04, color=FTE_GRAY, ax=ax, label='Observed tip percentages')
    if optimum:
        ax.scatter(fit['optimum_theta'], fit['optimum_loss'], marker='*', s=180, color=FTE_YELLOW,
                   edgecolor='black', zorder=5, label=f"Best grid value = {fit['optimum_theta']:.2f}%")
    ax.set(title='Average squared loss across the grid' if squared else 'Average absolute loss across the grid',
           xlabel='Constant prediction θ (%)', ylabel='MSE (percentage points²)' if squared else 'MAE (percentage points)')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_loss_contributions(table):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column, title in zip(axes, ['absolute_error', 'squared_error'], ['Absolute error for each observation', 'Squared error for each observation']):
        ax.bar(np.arange(len(table)), table[column], color=FTE_BLUE)
        ax.set_xticks(np.arange(len(table)), table['observed'])
        ax.set(title=title, xlabel='Observed value', ylabel='Absolute error' if column=='absolute_error' else 'Squared error')
    fig.tight_layout()
    plt.show()


def plot_small_losses(grid, mae, mse, observations, mean, median):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, curve, best, title, ylabel in zip(axes, [mae, mse], [median, mean],
            ['MAE has a corner at each observation', 'MSE has a smooth minimum'], ['MAE (value units)', 'MSE (value units²)']):
        ax.plot(grid, curve, color=FTE_BLUE)
        for value in observations:
            ax.axvline(value, color=FTE_GRAY, alpha=0.2, linewidth=1)
        ax.axvline(best, color=FTE_GREEN, linestyle='--', label=f'Minimizer = {best:g}')
        ax.set(title=title, xlabel='Constant prediction θ', ylabel=ylabel)
        ax.legend()
    fig.tight_layout()
    plt.show()


def plot_flat_minimum(grid, losses, middle):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grid, losses, color=FTE_BLUE)
    ax.axvspan(*middle, alpha=0.25, color=FTE_YELLOW, label=f'Every value from {middle[0]} to {middle[1]} minimizes MAE')
    ax.set(title='An even sample has an interval of MAE minimizers', xlabel='Constant prediction θ', ylabel='MAE (value units)')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_sensitivity(added_values, means, medians, base_mean, base_median):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(added_values, means, color=FTE_BLUE, label='Mean after adding one bill')
    ax.plot(added_values, medians, color=FTE_GREEN, linestyle='--', label='Median after adding one bill')
    ax.axhline(base_mean, color=FTE_BLUE, linestyle=':', alpha=0.5, label='Original mean')
    ax.axhline(base_median, color=FTE_GREEN, linestyle=':', alpha=0.5, label='Original median')
    ax.set(title='One extreme value moves the mean more than the median', xlabel='Added tip percentage (%)', ylabel='Fitted constant (%)')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_planning_losses(durations, grid, symmetric, asymmetric, best_symmetric, best_asymmetric):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    sns.rugplot(x=durations, height=0.45, ax=axes[0], color=FTE_BLUE)
    axes[0].axvline(best_symmetric, color=FTE_GREEN, linestyle='--', label=f'Equal costs: {best_symmetric:g} min')
    axes[0].axvline(best_asymmetric, color=FTE_RED, linestyle=':', label=f'Underestimate costs ×3: {best_asymmetric:g} min')
    axes[0].set(yticks=[], ylim=(0, 1), xlabel='Constructed repair duration (min)', title='The error costs change the planned duration')
    axes[0].legend()
    axes[1].plot(grid, symmetric, label='Equal costs', color=FTE_GREEN, linestyle='--')
    axes[1].plot(grid, asymmetric, label='Underestimate costs ×3', color=FTE_RED, linestyle=':')
    axes[1].set(xlabel='Planned duration θ (min)', ylabel='Mean cost (illustrative cost units)', title='Choose θ by minimizing the stated cost')
    axes[1].legend()
    fig.tight_layout()
    plt.show()
