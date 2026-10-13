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

def smoothie_errors(weight, observed, predicted, residuals):
    """Plot supplied calorie predictions and residuals on matched teaching panels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.scatter(weight, observed, color=FTE_BLUE, s=50, zorder=4, label='Observed', alpha=0.5)
    ax.plot(weight, predicted, color=FTE_GRAY, linewidth=1.1, label='Illustrative model')
    ax.vlines(weight, observed, predicted, color=FTE_RED, linewidth=1.0, label='Prediction error')
    ax.set(title='Calories and ingredient weight', xlabel='Ingredient weight (g)', ylabel='Calories (kcal)')
    ax.legend(fontsize=9)
    ax = axes[1]
    ids = np.arange(1, len(residuals) + 1)
    ax.axhline(0, color=FTE_GRAY, linewidth=0.8)
    ax.vlines(ids, 0, residuals, color=FTE_RED, linewidth=1.0)
    ax.scatter(ids, residuals, color=FTE_RED, s=46, edgecolors='white', linewidths=0.6, zorder=4, alpha=0.5)
    for i, r in zip(ids, residuals):
        ax.annotate(f'{r:+.0f}', (i, r), xytext=(0, 8 if r >= 0 else -16),
                    textcoords='offset points', ha='center', fontsize=11)
    ax.set(title='Each vertical gap is a residual', xlabel='Smoothie', ylabel='Observed − predicted (kcal)',
           xticks=ids, ylim=(min(residuals)-25, max(residuals)+25))
    finish(fig)

def metric_rankings(errors, scores):
    """Display fixed errors and already computed metrics without fitting a model."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ids = np.arange(1, len(next(iter(errors.values()))) + 1)
    for offset, (name, values), color, marker in zip([-0.12, 0.12], errors.items(),
                                                   [FTE_BLUE, FTE_RED], ['o', 's']):
        axes[0].vlines(ids+offset, 0, np.abs(values), color=color, linewidth=1.0)
        axes[0].scatter(ids+offset, np.abs(values), color=color, marker=marker,
                        s=48, edgecolors='white', linewidths=0.6, label=name, zorder=3, alpha=0.5)
    axes[0].set(title='Two different error patterns', xlabel='Smoothie', ylabel='Absolute error (kcal)',
                xticks=ids, ylim=(-3, 62))
    axes[0].legend()
    for row, metric in enumerate(['MAE', 'RMSE']):
        for offset, name, color, marker in zip([-0.10, 0.10], errors,
                                              [FTE_BLUE, FTE_RED], ['o', 's']):
            value=scores.loc[name, metric]
            axes[1].scatter(value, row+offset, color=color, marker=marker, s=70, zorder=3, alpha=0.5)
            axes[1].annotate(f'{name}: {value:.1f}', (value, row+offset), xytext=(8,0),
                             textcoords='offset points', va='center', fontsize=10)
    axes[1].set(title='The preferred model changes', xlabel='Error (kcal), lower is better',
                yticks=[0,1], yticklabels=['MAE','RMSE'], xlim=(0,34), ylim=(-0.45,1.45))
    axes[1].invert_yaxis()
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


def relationship_patterns(noise_x, noise_y, grid, linear_y, fan_y, curved_y):
    def show_relationship(ax, x, y, title):
        ax.scatter(x, y, alpha=0.6, s=15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    show_relationship(axes[0, 0], noise_x, noise_y, 'Noise: no form, no direction')
    show_relationship(axes[0, 1], grid, linear_y, 'Strong, positive, linear')
    show_relationship(axes[1, 0], grid, fan_y, 'Linear but spread grows with x')
    show_relationship(axes[1, 1], grid, curved_y, 'Strong but not linear')
    plt.tight_layout()
    plt.show()


def quartet(data, fits):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for j, (name, group) in enumerate(data.groupby('dataset')):
        fitted = fits[name]
        order = np.argsort(group['x'].to_numpy())
        axes[0, j].scatter(group['x'], group['y'], s=25, alpha=0.5)
        axes[0, j].plot(group['x'].to_numpy()[order], fitted[order], color=FTE_RED)
        axes[0, j].set(title=f'Dataset {name}', xlabel='x', ylabel='y', xlim=(0, 20), ylim=(0, 14))
        axes[1, j].scatter(fitted, group['y'] - fitted, s=25, alpha=0.5)
        axes[1, j].axhline(0, color=FTE_GRAY, linestyle='--')
        axes[1, j].set(title='Residuals', xlabel='Fitted y', ylabel='Observed − fitted', ylim=(-5, 5))
    finish(fig)

import plotly.graph_objects as go

# Display routines retained from the canonical lecture examples.

def original_l1_plot_9(tips):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = tips['total_bill'], y = tips['tip'], **SCATTER_STYLE)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Total bill (USD)', fontsize=12)
    ax.set_ylabel('Tip (USD)', fontsize=12)

    plt.gca().set_title('Larger bills tend to have larger tips', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_21(x, y):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(5, 5))
    ax = sns.scatterplot(x = x, y = y, label='Observations', **SCATTER_STYLE)
    sns.scatterplot(x = [20], y = [20], label='Outlier', **SCATTER_STYLE)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlim([-4, 25])
    ax.set_ylim([-4, 25])
    ax.legend(fontsize=12, loc=2)

    plt.gca().set_title('One distant point can dominate correlation', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_15(tips):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = tips['total_bill'], y = tips['tip'], label='Observations', **SCATTER_STYLE)
    sns.scatterplot(x = tips['total_bill'], y = tips['tip_predicted'], ax=ax, label='Averages', **SCATTER_STYLE)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Total bill (USD)', fontsize=12)
    ax.set_ylabel('Tip (USD)', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('Local averages clarify the bill–tip trend', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_18(tips, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = tips['total_bill'], y = tips['tip'], label='Observations', **SCATTER_STYLE)
    sns.scatterplot(x = tips['total_bill'], y= tips['tip_predicted'], ax=ax, label='Averages', **SCATTER_STYLE)
    sns.lineplot(x = tips['total_bill'], y= y_fitted.flatten(), color='#6d904f', ax=ax, label='Linear Model', lw=1.2)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Total bill (USD)', fontsize=12)
    ax.set_ylabel('Tip (USD)', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('The constrained line summarizes the local averages', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_29(height_weight, linear_model, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = height_weight['weight'],y = height_weight['height'], label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = height_weight['weight'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax, 
                 label='Linear Model\nheight = {:.2f} + {:.2f} * weight'.format(linear_model.intercept_[0], linear_model.coef_[0][0]), lw=1.2)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Weight (lb)', fontsize=12)
    ax.set_ylabel('Height (in)', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('Slope describes an association between different people', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_32(nba):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(x = nba['AST'], y = nba['PTS'], label='Observations', **SCATTER_STYLE)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Average Assists Per Game, AST', fontsize=12)
    ax.set_ylabel('Average Points Per Game, PTS', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('Assists and scoring are associated', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_35(linear_model, nba, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(x = nba['AST'], y = nba['PTS'], label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = nba['AST'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax, 
                 label='Linear Model\nPTS = {:.2f} + {:.2f} * AST'.format(linear_model.intercept_[0], linear_model.coef_[0][0]), lw=1.2)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Average Assists Per Game, AST', fontsize=12)
    ax.set_ylabel('Average Points Per Game, PTS', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('A one-feature NBA regression model', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_nba_plane(nba, u, v, predicted):
    fig = go.Figure()
    fig.add_surface(x=u, y=v, z=predicted, colorscale='Blues', opacity=0.5, showscale=False)
    fig.add_scatter3d(x=nba['AST'], y=nba['3PA'], z=nba['PTS'], mode='markers',
                      marker=dict(color=FTE_RED, size=3), name='Observed points', opacity=0.5)
    fig.update_layout(title='Two features define a regression plane', height=550,
                      scene=dict(xaxis_title='Assists per game', yaxis_title='Three-point attempts per game',
                                 zaxis_title='Points per game'), margin=dict(l=0, r=0, b=0, t=50))
    fig.show(renderer='plotly_mimetype')

def original_l1_plot_58(data):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['years of experience'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Years of experience', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)


    ax.legend(fontsize=12)

    plt.gca().set_title('Constructed salary increases with experience', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_59(data):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['years of experience'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = data['years of experience'], y = np.ones(len(data['salary']))*data['salary'].mean(), color='#6d904f', 
                 label='Mean Salary = {:.2f}'.format(data['salary'].mean()), lw=1.2)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Years of experience', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)


    ax.legend(fontsize=12)

    plt.gca().set_title('The mean gives everyone the same estimate', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_60(data):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['years of experience'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = data['years of experience'], y = np.ones(len(data['salary']))*data['salary'].mean(), color='#6d904f', 
                 label='Mean Salary = {:.2f}'.format(data['salary'].mean()), lw=1.2)
    ax.vlines(data['years of experience'], ymin=data['salary'], 
              ymax=np.ones(len(data['salary']))*data['salary'].mean(), 
              linestyle='dashed', color='#810f7c',alpha=0.3, zorder=0, 
              label='y - mean(y)')
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Years of experience', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)


    ax.legend(fontsize=12)

    plt.gca().set_title('Squared deviations measure variation around the mean', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_62(data, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['years of experience'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = data['years of experience'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax, 
                 label='Linear Model', lw=1.2)
    sns.lineplot(x = data['years of experience'], 
                 y = np.ones(len(data['salary']))*data['salary'].mean(), 
                 color='#6d904f', label='Mean Salary', lw=1.2)
    ax.vlines(data['years of experience'], ymin=data['salary'], 
              ymax=np.ones(len(data['salary']))*data['salary'].mean(), 
              linestyle='dashed', color='#810f7c',alpha=0.3, zorder=0, 
              label='y - mean(y)')
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Years of experience', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)


    ax.legend(fontsize=12)

    plt.gca().set_title('The fitted line follows the experience trend', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_63(data, linear_model, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['years of experience'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = data['years of experience'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax, 
                 label='Linear Model', lw=1.2)
    sns.lineplot(x = data['years of experience'], 
                 y = np.ones(len(data['salary']))*data['salary'].mean(), 
                 color='#6d904f', label='Mean Salary', lw=1.2)
    ax.vlines(data['years of experience'], ymin=data['salary'], 
              ymax=linear_model.intercept_[0] + data['years of experience']*linear_model.coef_[0], 
              linestyle='dashed', color='#810f7c',alpha=0.3, zorder=0, 
              label='Observed minus fitted')
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Years of experience', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)


    ax.legend(fontsize=12)

    plt.gca().set_title('Residuals measure variation left after fitting', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_69(data, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['height'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = data['height'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax, 
                 label='Linear Model', lw=1.2)
    sns.lineplot(x = data['height'], 
                 y = np.ones(len(data['salary']))*data['salary'].mean(), 
                 color='#6d904f', label='Mean Salary', lw=1.2)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Height', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)


    ax.legend(fontsize=12)

    plt.gca().set_title('Height has no designed connection with salary', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_70(data, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['height'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = data['height'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax, 
                 label='Linear Model', lw=1.2)
    sns.lineplot(x = data['height'], 
                 y = np.ones(len(data['salary']))*data['salary'].mean(), 
                 color='#6d904f', label='Mean Salary', lw=1.2)
    ax.vlines(data['height'], ymin=data['salary'], 
              ymax=np.ones(len(data['salary']))*data['salary'].mean(), 
              linestyle='dashed', color='#810f7c',alpha=0.3, zorder=0, 
              label='y - mean(y)')
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Height', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)


    ax.legend(fontsize=12)

    plt.gca().set_title('The mean remains the reference for R²', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_71(data, linear_model, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['height'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = data['height'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax, 
                 label='Linear Model', lw=1.2)
    sns.lineplot(x = data['height'], 
                 y = np.ones(len(data['salary']))*data['salary'].mean(), 
                 color='#6d904f', label='Mean Salary', lw=1.2)
    ax.vlines(data['height'], ymin=data['salary'], 
              ymax=linear_model.intercept_[0] + data['height']*linear_model.coef_[0], 
              linestyle='dashed', color='#810f7c',alpha=0.3, zorder=0, 
              label='Observed minus fitted')
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Height', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)


    ax.legend(fontsize=12)

    plt.gca().set_title('The height fit leaves much of the variation unexplained', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_34(nba, nba_one_predictions):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(x = nba_one_predictions, y = nba['PTS'] - nba_one_predictions, **SCATTER_STYLE)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Fitted points per game', fontsize=12)
    ax.set_ylabel('Observed minus fitted', fontsize=12)
    ax.set_ylim([-15, 15])

    plt.gca().set_title('NBA residuals: assists only', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_39(nba, nba_two_predictions):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(x = nba_two_predictions, y = nba['PTS'] - nba_two_predictions, **SCATTER_STYLE)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Fitted points per game', fontsize=12)
    ax.set_ylabel('Observed minus fitted', fontsize=12)
    ax.set_ylim([-15, 15])

    plt.gca().set_title('NBA residuals: assists and shot attempts', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_47(air_quality, linear_model, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(x = air_quality['Wind'], y = air_quality['Ozone'], label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = air_quality['Wind'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax,
                 label='Linear Model\nozone = {:.2f} + {:.2f} * wind'.format(linear_model.intercept_[0], linear_model.coef_[0][0]), lw=1.2)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Wind', fontsize=12)
    ax.set_ylabel('Ozone', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('Ozone tends to decrease as wind increases', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_48(air_quality, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.residplot(x = y_fitted.flatten(), 
                  y = air_quality['Ozone'], 
                  lowess=True, scatter_kws=SCATTER_STYLE)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Fitted ozone (ppb)', fontsize=12)
    ax.set_ylabel('Observed minus fitted', fontsize=12)
    plt.gca().set_title('Wind-model residuals can reveal curvature', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_50(air_quality, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = air_quality['Temp'], y = air_quality['Ozone'], label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = air_quality['Temp'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Temperature (°F)', fontsize=12)
    ax.set_ylabel('Ozone (ppb)', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('Ozone and temperature show a positive association', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_51(air_quality, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.residplot(x = y_fitted.flatten(), 
                  y = air_quality['Ozone'], 
                  lowess=True, scatter_kws=SCATTER_STYLE)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Fitted ozone (ppb)', fontsize=12)
    ax.set_ylabel('Observed minus fitted', fontsize=12)
    plt.gca().set_title('Temperature-model residuals show remaining variation', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_53(air_quality, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.residplot(x = y_fitted.flatten(), 
                  y = air_quality['Ozone'], 
                  lowess=True, scatter_kws=SCATTER_STYLE)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Fitted ozone (ppb)', fontsize=12)
    ax.set_ylabel('Observed minus fitted', fontsize=12)
    plt.gca().set_title('Several features still leave unexplained variation', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_78(data):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = data['years of experience'], y = data['salary'], 
                         label='Observations', **SCATTER_STYLE)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Years of experience', fontsize=12)
    ax.set_ylabel('Salary (illustrative units)', fontsize=12)
    ax.legend(fontsize=12)

    plt.gca().set_title('The API example reuses the salary data', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l1_plot_91(data, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(x = data['years of experience'], 
                         y = data['salary'], **SCATTER_STYLE)
    sns.lineplot(x = data['years of experience'], 
                 y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax, lw=1.2)
    plt.gca().set_title('The fitted estimator reproduces the salary trend', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")


def metric_error_comparison(errors, scores):
    """Draw supplied player errors and metrics; all calculations stay in the notebook."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    players = np.arange(1, len(next(iter(errors.values()))) + 1)
    styles = zip(errors.items(), [-0.12, 0.12], [FTE_BLUE, FTE_RED], ['o', 's'])
    for (name, values), offset, color, marker in styles:
        axes[0].vlines(players + offset, 0, np.abs(values), color=color, linewidth=1.0)
        axes[0].scatter(players + offset, np.abs(values), color=color, marker=marker,
                        s=50, alpha=0.5, linewidths=0.5, edgecolors='face', label=name, zorder=3)
    axes[0].set(title='One large miss or four smaller misses?', xlabel='Illustrative player',
                ylabel='Absolute error (points per game)', xticks=players, ylim=(-0.5, 9))
    axes[0].legend()
    for row, metric in enumerate(['MAE', 'RMSE']):
        for name, offset, color, marker in zip(errors, [-0.12, 0.12], [FTE_BLUE, FTE_RED], ['o', 's']):
            value = scores.loc[name, metric]
            axes[1].scatter(value, row + offset, color=color, marker=marker, s=65,
                            alpha=0.5, edgecolors='face', linewidths=0.5, zorder=3)
            axes[1].annotate(f'{name}: {value:.0f}', (value, row + offset), xytext=(8, 0),
                             textcoords='offset points', va='center', fontsize=10)
    axes[1].set(title='The metric changes the preferred model',
                xlabel='Error (points per game), lower is better',
                yticks=[0, 1], yticklabels=['MAE', 'RMSE'], xlim=(0, 5.5), ylim=(-0.4, 1.4))
    axes[1].invert_yaxis()
    finish(fig)
