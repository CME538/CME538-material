"""Helper functions for CME538 Lecture 8.2 (Regularization).

Kept out of the notebook to keep it clean. Import what you need, e.g.:

    from utils import plot_decision_regions
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_regions(ax, model, X, y, colors, alpha=0.2, resolution=300, pad=0.5):
    """Shade the decision regions of a fitted classifier and overlay the points.

    The region behind each class is filled with that class's own colour at a
    lighter alpha, so the shading and the dots always match.

    ax         : matplotlib axis to draw on
    model      : a fitted classifier with a .predict method taking a 2-column array
    X          : (n, 2) array of the two features that were used to fit `model`
    y          : (n,) integer class labels 0, 1, 2, ...
    colors     : list of hex colours, one per class, indexed by label
    alpha      : opacity of the shaded regions
    resolution : grid points per axis for the background
    pad        : how far past the data the grid extends
    """
    X = np.asarray(X)
    y = np.asarray(y)
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    classes = np.unique(y)
    levels = np.arange(classes.min() - 0.5, classes.max() + 1.5, 1.0)
    region_colors = [colors[c] for c in classes]
    ax.contourf(xx, yy, zz, levels=levels, colors=region_colors, alpha=alpha)

    for c in classes:
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], color=colors[c], s=35,
                   edgecolor='white', linewidth=0.6, zorder=3, label=f'class {c}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def plot_ridge_errors(results):
    """Display already calculated training and validation errors."""
    positive = results.loc[results['Alpha'] > 0]
    baseline = results.loc[results['Alpha'] == 0].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(positive['Alpha'], positive['Training RMSE'], 'o-', color='#30A2DA', label='Training RMSE')
    ax.plot(positive['Alpha'], positive['Validation RMSE'], '^-', color='#E5AE38', label='Validation RMSE')
    ax.axhline(baseline['Training RMSE'], color='#30A2DA', linestyle='--', linewidth=1,
               label='Unregularized training RMSE')
    ax.axhline(baseline['Validation RMSE'], color='#E5AE38', linestyle=':', linewidth=1.2,
               label='Unregularized validation RMSE')
    ax.set(xscale='log', xlabel='Ridge alpha', ylabel='RMSE (trillion L)', ylim=(0, 0.34),
           title='The strongest penalty gives the largest validation error')
    ax.set_xticks(positive['Alpha'], ['0.01', '0.1', '1', '10', '100'])
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


def plot_ridge_curves(training, validation, x_grid, curves, selected_alpha):
    """Display predictions already produced by the notebook models."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(training['water_level_change'], training['water_flow'] / 1e12,
               s=38, color='#30A2DA', marker='o', label='Training rows', zorder=4)
    ax.scatter(validation['water_level_change'], validation['water_flow'] / 1e12,
               s=60, facecolors='none', edgecolors='#E5AE38', marker='^',
               label='Validation rows', zorder=4)
    ax.plot(x_grid['water_level_change'], curves[0.0], color='#8B8B8B', linestyle='--',
            linewidth=1.5, label='Unregularized')
    ax.plot(x_grid['water_level_change'], curves[selected_alpha], color='#30A2DA',
            linewidth=1.5, label=f'Selected alpha {selected_alpha:g}')
    ax.plot(x_grid['water_level_change'], curves[100.0], color='#FC4F30', linestyle=':',
            linewidth=1.5, label='Alpha 100')
    ax.set(title='A strong penalty misses the largest water flows',
           xlabel='Water level change (m)', ylabel='Water flow (trillion L)', ylim=(-0.1, 1.65))
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig


def plot_unit_effects(training, grid, curves):
    """Compare predictions supplied by the notebook's four unit fits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, standardized in zip(axes, [False, True]):
        ax.scatter(training['water_level_change'], training['water_flow'] / 1e12,
                   color='#8B8B8B', s=30, label='Training rows', zorder=3)
        ax.plot(grid['water_level_change'], curves[(standardized, 'Metres')],
                color='#30A2DA', linewidth=1.7, label='Feature in metres')
        ax.plot(grid['water_level_change'], curves[(standardized, 'Kilometres')],
                color='#FC4F30', linestyle='--', linewidth=1.3, label='Feature in kilometres')
        ax.set(title='After standardization' if standardized else 'Before standardization',
               xlabel='Water level change (m)', ylim=(-0.2, 1.65))
        ax.legend(loc='upper left', fontsize=9)
    axes[0].set_ylabel('Predicted water flow (trillion L)')
    fig.tight_layout()
    return fig


def plot_coefficient_paths(path, title, x_label):
    """Plot existing standardized-feature coefficients across penalty strengths."""
    colours = ['#30A2DA', '#FC4F30', '#E5AE38', '#6D904F', '#8B8B8B', '#810F7C']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, feature in enumerate(path.columns):
        ax.plot(path.index, path[feature], marker='o', markersize=4,
                color=colours[i % len(colours)], linestyle='-' if i < 6 else '--',
                linewidth=1.2, label=feature)
    ax.axhline(0, color='#444445', linewidth=0.7)
    ax.set(xscale='log', xlabel=x_label, ylabel='Standardized-feature coefficient', title=title)
    ax.set_xticks(path.index, [f'{alpha:g}' for alpha in path.index])
    ax.legend(ncol=4, loc='upper right', fontsize=9, title='Feature')
    fig.tight_layout()
    return fig


def plot_lasso_comparison(training, validation, grid, ridge_curve, lasso_curve, coefficients):
    """Show already fitted curves and coefficients for the selected models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    ax = axes[0]
    ax.scatter(training['water_level_change'], training['water_flow'] / 1e12,
               s=30, color='#30A2DA', marker='o', label='Training rows', zorder=3)
    ax.scatter(validation['water_level_change'], validation['water_flow'] / 1e12,
               s=45, facecolors='none', edgecolors='#E5AE38', marker='^', label='Validation rows', zorder=3)
    ax.plot(grid['water_level_change'], ridge_curve, color='#30A2DA', linewidth=1.5, label='Selected Ridge')
    ax.plot(grid['water_level_change'], lasso_curve, color='#FC4F30', linewidth=1.3,
            linestyle='--', label='Selected Lasso')
    ax.set(title='Similar fitted curves', xlabel='Water level change (m)',
           ylabel='Water flow (trillion L)', ylim=(-0.1, 1.65))
    ax.legend(loc='upper left', fontsize=9)
    ax = axes[1]
    positions = np.arange(len(coefficients))
    ax.bar(positions - 0.18, coefficients['Ridge'], width=0.36, color='#30A2DA', label='Ridge')
    ax.bar(positions + 0.18, coefficients['Lasso'], width=0.36, color='#FC4F30', label='Lasso')
    ax.axhline(0, color='#444445', linewidth=0.8)
    ax.set_xticks(positions, coefficients.index, rotation=30)
    ax.set(title='Different coefficient patterns', xlabel='Standardized polynomial feature',
           ylabel='Coefficient')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    return fig


def plot_logistic_regularization(results, grid, curves):
    """Display reported probability losses and precomputed probability curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].plot(results['C'], results['Training cross-entropy'], 'o-', color='#30A2DA', label='Training')
    axes[0].plot(results['C'], results['Validation cross-entropy'], '^-', color='#E5AE38', label='Validation')
    axes[0].set(xscale='log', xlabel='C (larger values weaken the penalty)', ylabel='Cross-entropy',
                title='Compare predicted probabilities', ylim=(0.4, 0.62))
    axes[0].set_xticks(results['C'], [f'{c:g}' for c in results['C']])
    axes[0].legend(loc='upper right', fontsize=9)
    for c, colour, style in [(0.001, '#FC4F30', ':'), (0.1, '#30A2DA', '-'), (10.0, '#8B8B8B', '--')]:
        axes[1].plot(grid['FG_PCT_DIFF'], curves[c], color=colour, linestyle=style,
                     linewidth=1.5, label=f'C = {c:g}')
    axes[1].axhline(0.5, color='#8B8B8B', linestyle=':', linewidth=0.8)
    axes[1].set(title='Stronger penalties flatten the curve', xlabel='Field-goal percentage difference',
                ylabel='Predicted probability of a win', ylim=(-0.02, 1.02))
    axes[1].legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig
