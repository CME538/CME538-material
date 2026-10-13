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


def bin_comparison(values, width_edges, quantile_edges):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, edges, title in zip(axes, [width_edges, quantile_edges], ['Equal numerical widths', 'Similar observation counts']):
        ax.hist(values, bins=20, color=FTE_BLUE, edgecolor='white')
        for edge in edges: ax.axvline(edge, color=FTE_RED, linestyle='--', linewidth=1)
        ax.set(title=title, xlabel='Fuel economy (mpg)', ylabel='Vehicles')
    finish(fig)


def ozone_transforms(air):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, (column, label) in enumerate([('Ozone', 'Ozone (ppb)'), ('log_ozone', 'Log ozone')]):
        axes[0, j].hist(air[column], bins=20, color=FTE_COLORS[j], edgecolor='white')
        axes[0, j].set(title=f'Distribution of {label.lower()}', xlabel=label, ylabel='Days')
        axes[1, j].scatter(air['Temp'], air[column], alpha=0.55, s=25, color=FTE_COLORS[j])
        axes[1, j].set(title='Relationship with temperature', xlabel='Temperature (°F)', ylabel=label)
    finish(fig)


def transformation_fits(air, predictions):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, pred) in zip(axes, predictions.items()):
        xx = air['Temp'] ** 2 if name == 'Squared temperature' else air['Temp']
        yy = air['log_ozone'] if name == 'Log target' else air['Ozone']
        order = np.argsort(xx.to_numpy())
        ax.scatter(xx, yy, s=25, alpha=0.5)
        ax.plot(xx.iloc[order], pred[order], color=FTE_RED)
        ax.set(title=name, xlabel='Temperature² (°F²)' if name == 'Squared temperature' else 'Temperature (°F)',
               ylabel='Log ozone' if name == 'Log target' else 'Ozone (ppb)')
    finish(fig)


def scaling_comparison(raw, standardized, normalized):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, frame, title in zip(axes, [raw, standardized, normalized], ['Original units', 'Standardized', 'Min-max scaled']):
        frame.boxplot(ax=ax, grid=False)
        ax.set(title=title, ylabel='Feature value')
        ax.tick_params(axis='x', rotation=40)
    finish(fig)


def carrier_rates(carrier_mix, outcome_order):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    carrier_mix[outcome_order].plot.barh(
        stacked=True,
        ax=ax,
        color=[FTE_BLUE, FTE_YELLOW, FTE_RED],
        width=0.82,
    )
    ax.set_title('Outcome rates for carriers with at least 50 flights')
    ax.set_xlabel('Share of scheduled flights (%)')
    ax.set_ylabel('Carrier')
    ax.set_xlim(0, 100)
    ax.legend(
        ['On time', 'Delayed at least 15 min', 'Cancelled'],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def wind_summary(development_features):
    WIND_ORDER = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    recorded_wind = development_features.loc[
        development_features['wind_direction_deg'].notna(),
        'wind_cardinal',
    ]
    missing_wind_share = 100 * development_features['wind_direction_deg'].isna().mean()
    direction_share = (
        100 * recorded_wind
        .value_counts(normalize=True)
        .reindex(WIND_ORDER, fill_value=0)
    )
    angles = np.deg2rad(np.arange(0, 360, 45))
    dominant_direction = direction_share.idxmax()
    direction_colors = [
        FTE_RED if direction == dominant_direction else FTE_BLUE
        for direction in WIND_ORDER
    ]

    fig = plt.figure(figsize=(13, 5.2))
    ax_polar = fig.add_subplot(1, 2, 1, projection='polar')
    ax_bar = fig.add_subplot(1, 2, 2)

    ax_polar.bar(
        angles,
        direction_share.values,
        width=np.deg2rad(40),
        color=direction_colors,
        edgecolor='white',
        alpha=0.9,
    )
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_xticks(angles, WIND_ORDER)
    ax_polar.set_title('Recorded wind directions form a circular pattern', loc='left', pad=18)

    bars = ax_bar.bar(WIND_ORDER, direction_share.values, color=direction_colors)
    ax_bar.bar_label(bars, fmt='%.1f%%', padding=3)
    ax_bar.set_title('Recorded compass bins preserve the pattern')
    ax_bar.set_xlabel('Engineered wind category')
    ax_bar.set_ylabel('Recorded directions (%)')
    ax_bar.set_ylim(0, direction_share.max() * 1.18)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    print(f'Missing directions retained as a separate category: {missing_wind_share:.1f}%')


def weather_missingness(development_features):
    weather_columns = [
        'temperature_f', 'dew_point_f', 'relative_humidity_pct',
        'wind_direction_deg', 'wind_speed_mph', 'wind_gust_mph',
        'precipitation_in', 'sea_level_pressure_mb', 'visibility_miles',
    ]
    missing_percent = (
        100 * development_features[weather_columns].isna().mean()
    ).sort_values()

    gust_plot = development_features.dropna(subset=['wind_speed_mph']).copy()
    gust_plot['gust_status'] = np.where(
        gust_plot['gust_recorded'].eq(1),
        'Gust recorded',
        'Gust not recorded',
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    missing_labels = [name.replace('_', ' ') for name in missing_percent.index]
    missing_colors = [
        FTE_RED if name == 'wind_gust_mph' else FTE_BLUE
        for name in missing_percent.index
    ]
    bars = axes[0].barh(missing_labels, missing_percent.values, color=missing_colors)
    axes[0].bar_label(bars, fmt='%.1f%%', padding=3)
    axes[0].set_title('Gust magnitude is usually absent')
    axes[0].set_xlabel('Missing values (%)')
    axes[0].set_xlim(0, 85)

    sns.violinplot(
        data=gust_plot,
        x='gust_status',
        y='wind_speed_mph',
        hue='gust_status',
        order=['Gust not recorded', 'Gust recorded'],
        palette={'Gust not recorded': FTE_GRAY, 'Gust recorded': FTE_RED},
        legend=False,
        inner='quartile',
        cut=0,
        ax=axes[1],
    )
    axes[1].set_title('Recorded gusts occur at higher wind speeds')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Wind speed (mph)')

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def clock_encoding():
    clock = pd.DataFrame({'hour': np.arange(24)})
    clock['hour_sin'] = np.sin(2 * np.pi * clock['hour'] / 24)
    clock['hour_cos'] = np.cos(2 * np.pi * clock['hour'] / 24)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(clock['hour'], clock['hour_sin'], marker='o', color=FTE_BLUE, label='sine')
    axes[0].plot(clock['hour'], clock['hour_cos'], marker='o', color=FTE_RED, label='cosine')
    axes[0].axhline(0, color=FTE_GRAY, linewidth=1)
    axes[0].set_title('One hour needs two coordinates')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('Encoded value')
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend(frameon=False)

    axes[1].plot(clock['hour_cos'], clock['hour_sin'], color=FTE_GRAY, linewidth=1.2)
    axes[1].scatter(clock['hour_cos'], clock['hour_sin'], color=FTE_YELLOW, s=55, zorder=3, alpha=0.5)
    for hour in [0, 6, 12, 18]:
        row = clock.loc[clock['hour'].eq(hour)].iloc[0]
        axes[1].annotate(
            f'{hour}:00',
            (row['hour_cos'], row['hour_sin']),
            xytext=(6, 6),
            textcoords='offset points',
        )
    axes[1].set_title('Nearby hours remain nearby')
    axes[1].set_xlabel('cos(hour)')
    axes[1].set_ylabel('sin(hour)')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(-1.2, 1.2)
    axes[1].set_ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def calendar_rates(monthly_disruption, hourly_disruption):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].plot(monthly_disruption.index, monthly_disruption.values, marker='o', color=FTE_BLUE)
    axes[0].set_title('Disruption rates vary through the year')
    axes[0].set_xlabel('Scheduled departure month')
    axes[0].set_ylabel('Delayed at least 15 min or cancelled (%)')
    axes[0].set_xticks(range(1, 13))

    axes[1].plot(hourly_disruption.index, hourly_disruption.values, marker='o', color=FTE_RED)
    axes[1].set_title('Disruptions become more common later in the day')
    axes[1].set_xlabel('Scheduled departure hour')
    axes[1].set_ylabel('Delayed at least 15 min or cancelled (%)')
    axes[1].set_xticks(range(5, 24, 3))

    plt.tight_layout()
    plt.show()
    plt.close(fig)

# Display routines retained from the canonical lecture examples.

def original_l2_plot_20(mpg):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    sns.histplot(mpg['mpg'], bins=3, edgecolor='white')
    sns.rugplot(mpg['mpg'], height=0.04)
    plt.xlabel('Fuel economy (mpg)')
    plt.ylabel('Number of vehicles')
    plt.gca().set_title('Equal-width bins can contain different counts', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_mpg_quantiles(values, quantiles):
    fig, ax = plt.subplots()
    ax.hist(values, bins=15, color=FTE_BLUE, edgecolor='white')
    for q in quantiles: ax.axvline(q, color=FTE_RED, linewidth=1.1, linestyle='--')
    ax.set(title='Quantile boundaries adapt to the observed distribution', xlabel='Fuel economy (mpg)', ylabel='Number of vehicles')
    finish(fig)

def original_l2_plot_29(air_quality):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = air_quality['Temp'], y = air_quality['Ozone'], label='Observations', **SCATTER_STYLE)
    # sns.lineplot(x = air_quality['Temp'], y = y_fitted.flatten(), 
    #              color='#fc4f30', ax=ax)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Temperature (°F)', fontsize=12)
    ax.set_ylabel('Ozone', fontsize=12)

    ax.legend(fontsize=12)

    plt.gca().set_title('Ozone and temperature in their original units', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l2_plot_30(air_quality, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    ax = sns.residplot(x = y_fitted.flatten(), 
                  y = air_quality['Ozone'], 
                  lowess=True, scatter_kws=SCATTER_STYLE)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Fitted ozone (ppb)', fontsize=12)
    ax.set_ylabel('Observed minus fitted', fontsize=12)

    plt.gca().set_title('The raw-scale line leaves structure in its residuals', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l2_plot_32(air_quality, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = air_quality['Temp'], y = np.log(air_quality['Ozone']), label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = air_quality['Temp'], y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Temperature (°F)', fontsize=12)
    ax.set_ylabel('log Ozone', fontsize=12)

    ax.legend(fontsize=12)

    plt.gca().set_title('A logarithmic target compresses large ozone values', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l2_plot_34(air_quality, y_fitted):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(10, 5))
    ax = sns.scatterplot(x = air_quality['Temp']**2, y = air_quality['Ozone'], label='Observations', **SCATTER_STYLE)
    sns.lineplot(x = air_quality['Temp']**2, y = y_fitted.flatten(), 
                 color='#fc4f30', ax=ax)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Temperature $^2$', fontsize=12)
    ax.set_ylabel('Ozone', fontsize=12)
    #
    ax.legend(fontsize=12)

    plt.gca().set_title('Squared temperature changes the input representation', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l2_plot_37(ice_cream):
    """Render the original lecture comparison with shared typography."""
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x='sweetness', y='overall', data=ice_cream, **SCATTER_STYLE)
    plt.title('Overall taste rating vs. sweetness')
    plt.gca().set_title('Taste can rise and fall with sweetness', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l2_plot_38(ice_cream):
    """Render the original lecture comparison with shared typography."""
    sns.lmplot(x='sweetness', y='overall', data=ice_cream, scatter_kws=SCATTER_STYLE)
    plt.title('Overall taste rating vs. sweetness')
    plt.gca().set_title('A straight line misses the rise and fall', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l2_plot_41(clf, ice_cream, transformer):
    """Render the original lecture comparison with shared typography."""
    sns.lmplot(x='sweetness', y='overall', data=ice_cream, fit_reg=False, scatter_kws=SCATTER_STYLE)
    xs = np.linspace(3.5, 12.5, 1000).reshape(-1, 1)
    ys = clf.predict(transformer.transform(pd.DataFrame(xs, columns=['sweetness'])))
    plt.plot(xs, ys)
    plt.title('Degree 2 polynomial fit')
    plt.gca().set_title('A quadratic can represent one turning point', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l2_plot_43(clf_five, ice_cream, trans_five):
    """Render the original lecture comparison with shared typography."""
    sns.lmplot(x='sweetness', y='overall', data=ice_cream, fit_reg=False, scatter_kws=SCATTER_STYLE)
    xs = np.linspace(3.5, 12.5, 1000).reshape(-1, 1)
    ys = clf_five.predict(trans_five.transform(pd.DataFrame(xs, columns=['sweetness'])))
    plt.plot(xs, ys)
    plt.title('Degree 5 polynomial fit')
    plt.gca().set_title('A degree-five curve is more flexible', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_l2_plot_44(clf_ten, ice_cream, trans_ten):
    """Render the original lecture comparison with shared typography."""
    sns.lmplot(x='sweetness', y='overall', data=ice_cream, fit_reg=False, scatter_kws=SCATTER_STYLE)
    xs = np.linspace(3.5, 12.5, 1000).reshape(-1, 1)
    ys = clf_ten.predict(trans_ten.transform(pd.DataFrame(xs, columns=['sweetness'])))
    plt.plot(xs, ys)
    plt.title('Degree 10 polynomial fit')
    plt.gca().set_title('A degree-ten curve can bend sharply between observations', loc="left")
    plt.tight_layout()
    plt.show()
    plt.close("all")

def original_month_cycle(months, sine, cosine):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.r_[sine, sine[0]], np.r_[cosine, cosine[0]], color=FTE_GRAY, linewidth=1)
    ax.scatter(sine, cosine, s=55, zorder=3, alpha=0.5)
    for m,x,y in zip(months,sine,cosine):ax.annotate(str(m), (x,y), xytext=(7,7), textcoords='offset points')
    ax.set(title='Month 12 sits next to month 1', xlabel='Sine of month angle', ylabel='Cosine of month angle', xlim=(-1.3,1.3), ylim=(-1.3,1.3))
    ax.set_aspect('equal');finish(fig)

def original_interval_plot(intervals):
    fig, ax = plt.subplots()
    for (name, values), marker in zip(intervals.items(), ['o','s']):ax.plot(np.arange(1,len(values)+1), values, marker=marker, linewidth=1, label=name)
    ax.set(title='Peak intervals reveal timing variability', xlabel='Interval number', ylabel='Time between peaks (s)', ylim=(0,1.3));ax.legend();finish(fig)
