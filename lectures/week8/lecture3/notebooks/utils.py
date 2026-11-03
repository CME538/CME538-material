"""Display-only figures for Lecture 8.3. No fitting or file writes."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from cycler import cycler

BLUE = '#30A2DA'
RED = '#FC4F30'
GREEN = '#6D904F'
GRAY = '#8B8B8B'


def plot_observations(train, x_plot, true_curve):
    fig, ax = plt.subplots()
    ax.scatter(train['x'], train['y'], s=42, color=BLUE, label='Training observations', zorder=3)
    ax.plot(x_plot['x'], true_curve, color=GREEN, linestyle='--', label='True function')
    ax.set(title='Twenty noisy observations from a cubic function', xlabel='x', ylabel='y', xlim=(-10, 10))
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig


def plot_single_fit(train, x_plot, true_curve, fitted_curve):
    fig, ax = plt.subplots()
    ax.scatter(train['x'], train['y'], s=42, color=BLUE, label='Training observations', zorder=3)
    ax.plot(x_plot['x'], true_curve, color=GREEN, linestyle='--', label='True function')
    ax.plot(x_plot['x'], fitted_curve, color=RED, label='Degree 3 fit')
    ax.set(title='A fitted cubic follows the main bend', xlabel='x', ylabel='y', xlim=(-10, 10))
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig


def plot_repeated_models(x_plot, true_curve, predictions):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
    for ax, degree in zip(axes, [1, 3, 10]):
        for index, curve in enumerate(predictions[degree]):
            ax.plot(x_plot['x'], curve, color=BLUE, alpha=0.65, linewidth=1,
                    label='Fitted samples' if index == 0 else None)
        ax.plot(x_plot['x'], true_curve, color=GREEN, linestyle='--', linewidth=1.8, label='True function')
        ax.axvline(8, color=GRAY, linestyle=':', linewidth=1)
        ax.set(title=f'Degree {degree}', xlabel='x', xlim=(-8, 8.4), ylim=(-2, 7))
        ax.set_xticks([-8, -4, 0, 4, 8])
    axes[0].set_ylabel('Predicted y')
    axes[0].legend(loc='upper left', fontsize=9)
    fig.suptitle('The same eight training samples, three model degrees', x=0.07, ha='left', fontsize=16)
    fig.tight_layout()
    return fig


def plot_degree_errors(results, noise_sd):
    fig, ax = plt.subplots()
    ax.plot(results['Degree'], results['Training RMSE'], 'o-', color=BLUE, label='Training RMSE')
    ax.plot(results['Degree'], results['Validation RMSE'], 's-', color=RED, label='Validation RMSE')
    ax.axhline(noise_sd, color=GRAY, linestyle='--', linewidth=1, label='Noise standard deviation (0.3)')
    for degree, offset in [(3, (8, 24)), (10, (-95, 18))]:
        row = results.loc[results['Degree'] == degree].iloc[0]
        ax.annotate(f"Degree {degree}: {row['Validation RMSE']:.3f}",
                    xy=(degree, row['Validation RMSE']), xytext=offset,
                    textcoords='offset points', fontsize=10,
                    arrowprops={'arrowstyle':'-', 'color':GRAY, 'linewidth':0.8})
    ax.set(title='Increasing degree eventually raises validation error',
           xlabel='Polynomial degree', ylabel='Root mean squared error (RMSE)', ylim=(0, None))
    ax.set_xticks(range(1, 13))
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig


def plot_underfitting(x_plot, true_curve, predictions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
    for ax, degree in zip(axes, [1, 3]):
        ax.plot(x_plot['x'], true_curve, color=GREEN, linestyle='--', label='True function')
        ax.plot(x_plot['x'], predictions[(degree, 20)], color=BLUE, label='20 training rows')
        ax.plot(x_plot['x'], predictions[(degree, 300)], color=RED, linestyle=':', label='300 training rows')
        ax.set(title=f'Degree {degree}', xlabel='x', xlim=(-10, 10), ylim=(-2, 8))
        ax.legend(loc='upper left', fontsize=9)
    axes[0].set_ylabel('Predicted y')
    fig.suptitle('More observations do not give a straight line a curved shape', x=0.06, ha='left', fontsize=16)
    fig.tight_layout()
    return fig


def plot_remedies(train, x_plot, true_curve, curves, results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True, sharey=True)
    for index, ax in enumerate(axes.flat):
        ax.scatter(train['x'], train['y'], color=GRAY, s=22, alpha=0.75, label='Original 20 rows', zorder=3)
        ax.plot(x_plot['x'], true_curve, color=GREEN, linestyle='--', label='True function')
        ax.plot(x_plot['x'], curves[index], color=RED if index == 0 else BLUE, label='Fitted model')
        row = results.iloc[index]
        ax.set(title=f"{row['Model']}\nValidation RMSE {row['Validation RMSE']:.3f}",
               xlabel='x', ylabel='Predicted y', xlim=(-10, 10), ylim=(-12, 19))
        ax.tick_params(labelbottom=True, labelleft=True)
    axes[0, 0].legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


def configure_plotting():
    """Apply the shared CME538 figure colours and typography."""
    FTE_BLUE = '#30A2DA'
    FTE_RED = '#FC4F30'
    FTE_YELLOW = '#E5AE38'
    FTE_GREEN = '#6D904F'
    FTE_GRAY = '#8B8B8B'
    FTE_PURPLE = '#810F7C'
    FTE_COLORS = [FTE_BLUE, FTE_RED, FTE_YELLOW, FTE_GREEN, FTE_GRAY, FTE_PURPLE]

    plt.style.use('fivethirtyeight')
    plt.rcParams.update({
        'axes.prop_cycle': cycler(color=FTE_COLORS),
        'axes.titlesize': 16, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'figure.titlesize': 16,
        'figure.figsize': (9, 5), 'figure.dpi': 110, 'lines.linewidth': 1.5,
    })


def plot_bias_variance(results):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for column, color, marker in [('Squared Bias', BLUE, 'o'),
                                  ('Variance', RED, 's'),
                                  ('Estimated MSE', '#444445', 'D')]:
        ax.plot(results['Degree'], results[column], marker=marker, color=color, label=column)
    ax.axhline(results['Noise'].iloc[0], color=GRAY, linestyle='--', linewidth=1,
               label='Noise variance (0.09)')
    ax.set(title='Bias and variance contribute differently as degree increases',
           xlabel='Polynomial degree', ylabel='Squared error at x = 8', ylim=(0, None))
    ax.set_xticks(results['Degree'])
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


def plot_learning_curves(results, noise_sd):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
    for ax, degree in zip(axes, [1, 3, 10]):
        rows = results.loc[results['Degree'] == degree]
        ax.plot(rows['Training observations'], rows['Training RMSE'], 'o-', color=BLUE,
                label='Training RMSE')
        ax.plot(rows['Training observations'], rows['Validation RMSE'], 's-', color=RED,
                label='Validation RMSE')
        ax.axhline(noise_sd, color=GRAY, linestyle='--', linewidth=1,
                   label='Noise standard deviation')
        ax.set(title=f'Degree {degree}', xlabel='Training observations', ylim=(0, 2.7))
        ax.set_xticks([20, 80, 150, 300])
    axes[0].set_ylabel('Root mean squared error (RMSE)')
    axes[0].legend(loc='upper right', fontsize=9)
    fig.suptitle('The same validation rows at every training size', x=0.06, ha='left', fontsize=16)
    fig.tight_layout()
    return fig


def plot_forest_capacity_learning_curves(results):
    """Display already-calculated forest learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
    styles = [('3', BLUE, 'o'), ('6', RED, 's'), ('Unlimited', '#E5AE38', '^')]
    sizes = [20, 50, 100, 200, 500, 1000, 2000, 4000]
    for ax, column, title in zip(axes,
            ['Training accuracy', 'Validation accuracy'], ['Training', 'Fixed validation set']):
        for depth, color, marker in styles:
            rows = results.loc[results['Maximum depth'] == depth].sort_values('Training observations')
            label = 'No depth limit' if depth == 'Unlimited' else 'Maximum depth ' + depth
            ax.plot(rows['Training observations'], rows[column], color=color,
                    marker=marker, linewidth=1.4, markersize=6, label=label)
        ax.set_xscale('log')
        ax.set_xticks(sizes, [str(size) for size in sizes], rotation=40)
        ax.set(title=title, xlabel='Training observations (log scale)', ylim=(0.5, 1.01))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Mean accuracy across five training sequences')
    axes[1].legend(loc='lower right', fontsize=10)
    fig.suptitle('More training data and greater depth improve validation accuracy',
                 x=0.07, ha='left', fontsize=15)
    fig.tight_layout()
    return fig


def _classification_axes(ax):
    # Known equal-probability boundary of the simulated overlapping groups.
    x = np.array([-3.5, 4.5])
    ax.plot(x, 0.421875 * x + 0.08125, color='black', linestyle='--',
            linewidth=1.1, label='Equal-probability boundary')
    ax.set(xlim=(-3.5, 4.5), ylim=(-2, 2.5), xlabel='Feature 1', ylabel='Feature 2')
    ax.set_aspect('equal')
    ax.set_xticks([-3, -1, 1, 3])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.grid(False)


def _classification_observations(ax, features, labels, size=28):
    for label, color, marker in [(0, RED, 'o'), (1, BLUE, '^')]:
        selected = labels == label
        ax.scatter(features[selected, 0], features[selected, 1], edgecolors=color,
                   marker=marker, s=size, facecolors='none', linewidths=0.8,
                   label=f'Class {label}', zorder=3, alpha=0.8)


def plot_classification_data(features, labels):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _classification_observations(ax, features, labels, size=36)
    _classification_axes(ax)
    ax.set_title('Two overlapping groups with a diagonal separation')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    return fig


def plot_classifier_boundaries(features, labels, grid_x, grid_y, probabilities,
                               titles, training_errors, validation_errors):
    count = len(titles)
    fig, axes = plt.subplots(1, count, figsize=(5.4 * count, 4.7), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    backgrounds = ListedColormap(['#FCE3DE', '#DDEFF8'])
    for index, (ax, probability, title, train_error, val_error) in enumerate(zip(
            axes, probabilities, titles, training_errors, validation_errors)):
        surface = np.asarray(probability).reshape(grid_x.shape)
        ax.contourf(grid_x, grid_y, surface >= 0.5, levels=[-0.5, 0.5, 1.5],
                    cmap=backgrounds, alpha=0.9)
        if surface.min() < 0.5 < surface.max():
            ax.contour(grid_x, grid_y, surface, levels=[0.5], colors='black', linewidths=1)
        panel_X = features[index] if isinstance(features, list) else features
        panel_y = labels[index] if isinstance(labels, list) else labels
        _classification_observations(ax, panel_X, panel_y, size=14 if len(panel_y) > 200 else 28)
        _classification_axes(ax)
        ax.set_title(f'{title}\nTraining error {train_error:.3f}\nValidation error {val_error:.3f}', fontsize=12)
    handles = [Line2D([], [], color=RED, marker='o', markerfacecolor='none', linestyle='None', label='Class 0'),
               Line2D([], [], color=BLUE, marker='^', markerfacecolor='none', linestyle='None', label='Class 1'),
               Line2D([], [], color='black', linestyle='-', label='Fitted boundary'),
               Line2D([], [], color='black', linestyle='--', label='Equal-probability boundary')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return fig


def plot_classifier_learning_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    styles = [('Depth 1 Tree', RED, 'o'), ('Logistic Regression', BLUE, '^')]
    for ax, column, title in zip(axes, ['Training Error', 'Validation Error'],
                                ['Training', 'Fixed Validation']):
        for name, color, marker in styles:
            rows = results.loc[results['Model'] == name]
            ax.plot(rows['Training observations'], rows[column], color=color,
                    marker=marker, markersize=5, label=name)
        ax.set(title=title, xlabel='Training observations', ylim=(0, 0.35))
        ax.set_xticks([30, 200, 480, 960])
    axes[0].set_ylabel('Mean classification error')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10)
    fig.suptitle('More data does not give a one-split tree a diagonal boundary',
                 x=0.06, ha='left', fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    return fig


def plot_classifier_stability(grid_x, grid_y, probabilities):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, name, color in zip(axes, ['Full Tree', 'Random Forest'], [RED, GREEN]):
        for probability in probabilities[name][:8]:
            surface = np.asarray(probability).reshape(grid_x.shape)
            ax.contour(grid_x, grid_y, surface, levels=[0.5], colors=[color],
                       linewidths=0.8, alpha=0.6)
        _classification_axes(ax)
        ax.set_title(name)
    handles = [Line2D([], [], color=RED, label='Retrained full tree'),
               Line2D([], [], color=GREEN, label='Retrained random forest'),
               Line2D([], [], color='black', linestyle='--', label='Equal-probability boundary')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Eight samples of 80 observations give eight fitted boundaries',
                 x=0.06, ha='left', fontsize=15)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    return fig
