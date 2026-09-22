"""Display and plotting helpers for CME538 Lecture 3.1.

Data loading, grouping, resampling, and missing-value decisions stay visible in
the notebook. These helpers configure the display or draw supplied data.
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.spatial import cKDTree

FTE_BLUE = '#30A2DA'
FTE_RED = '#FC4F30'
FTE_YELLOW = '#E5AE38'
FTE_GREEN = '#6D904F'
FTE_GRAY = '#8B8B8B'
FTE_PURPLE = '#810F7C'
FTE_COLORS = [FTE_BLUE, FTE_RED, FTE_YELLOW, FTE_GREEN, FTE_GRAY, FTE_PURPLE]


def configure_notebook():
    """Apply the shared CME538 palette, typography, and table display."""
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({
        'axes.prop_cycle': cycler(color=FTE_COLORS),
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.figsize': (9, 5),
        'figure.dpi': 110,
    })
    pd.set_option('display.max_columns', 8)
    pd.set_option('display.max_rows', 12)


def plot_structure(passenger_class_counts, recorded_ages, class_survival):
    """Show a category, a distribution, and an observed association."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    passenger_class_counts.plot.bar(
        ax=axes[0], color=FTE_BLUE, rot=0,
    )
    axes[0].set(title='Most records are third class',
                xlabel='Passenger class', ylabel='Recorded passengers')
    axes[1].hist(recorded_ages, bins=30, color=FTE_BLUE,
                 edgecolor='white', linewidth=0.5)
    axes[1].set(title='Ages span infancy to 80',
                xlabel='Age (years)', ylabel='Recorded passengers')
    class_survival.plot.bar(
        ax=axes[2], color=FTE_RED, rot=0,
    )
    axes[2].set(title='Survival share differs by class',
                xlabel='Passenger class', ylabel='Observed survival proportion',
                ylim=(0, 1))
    fig.tight_layout()
    plt.show()


def plot_counter_counts(counts, interval_minutes):
    """Keep the two-hour window and sidewalk channels fixed as grain coarsens."""
    start = counts.index.min()
    offsets = (counts.index - start).total_seconds() / 60
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(offsets, counts['West'], width=interval_minutes, align='edge',
           color=FTE_BLUE, edgecolor='white', label='West sidewalk')
    ax.bar(offsets, counts['East'], bottom=counts['West'],
           width=interval_minutes, align='edge', color=FTE_YELLOW,
           edgecolor='white', hatch='//', label='East sidewalk')
    interval_totals = counts[['West', 'East']].sum(axis=1)
    ax.set_ylim(0, interval_totals.max() * 1.30)
    ax.set_xlim(0, 120)
    ticks = pd.date_range(start, periods=9, freq='15min')
    ax.set_xticks(np.arange(0, 121, 15), ticks.strftime('%H:%M'))
    grain = '5-minute' if interval_minutes == 5 else 'Hourly'
    total = int(interval_totals.sum())
    ax.set_title(f'{grain} counts: same {total:,} crossings, less timing detail')
    ax.set_xlabel('Time on 3 October 2012 (source-file clock labels)')
    ax.set_ylabel(f'Crossings per {interval_minutes}-minute interval')
    ax.legend(loc='upper right', ncols=2)
    if interval_minutes == 60:
        for offset, total in zip(offsets, interval_totals):
            ax.text(offset + 30, total + interval_totals.max() * 0.03,
                    f'{int(total):,}', ha='center', fontsize=12)
    fig.tight_layout()
    plt.show()


def plot_bridge_month_progression(hourly, daily, monthly):
    """Compare resolutions of the identical complete November window."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    start = hourly.index.min().normalize()
    end = start + pd.offsets.MonthBegin(1)
    titles = ['Hourly: within-day peaks remain',
              'Daily: busy days remain; within-day peaks disappear',
              'Monthly: only the two sidewalk totals remain']
    for ax, counts, unit, title in zip(
        axes, [hourly, daily, monthly], ['hour', 'day', 'month'], titles,
    ):
        if unit == 'month':
            west, east = counts.iloc[0][['West', 'East']]
            width = (end - start).days
            ax.bar(start, west, width=width, align='edge', color=FTE_BLUE,
                   edgecolor='white', label='West sidewalk')
            ax.bar(start, east, width=width, align='edge', bottom=west,
                   color=FTE_YELLOW, edgecolor='white', hatch='//', label='East sidewalk')
            ax.text(start + (end - start) / 2, (west + east) * 1.05,
                    f'{int(west + east):,} total = {int(west):,} West + {int(east):,} East',
                    ha='center', fontsize=11)
            ax.set_ylim(0, (west + east) * 1.3)
        else:
            for sidewalk, color, linestyle in zip(
                ['West', 'East'], [FTE_BLUE, FTE_YELLOW], ['-', '--'],
            ):
                ax.plot(counts.index, counts[sidewalk], color=color,
                        linestyle=linestyle, linewidth=1.2 if unit == 'hour' else 2,
                        label=f'{sidewalk} sidewalk')
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.set_ylabel(f'Crossings per {unit}')
        ax.set_xlim(start, end)
    axes[0].legend(loc='upper right', ncols=2)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axes[-1].set_xlabel('Same observation window: 1–30 November 2012')
    fig.tight_layout(h_pad=1.5)
    plt.show()


def plot_hourly_profile(by_hour):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(by_hour.index, by_hour.values, color=FTE_BLUE, marker='o', markersize=4)
    ax.set(title='The all-days average has morning and evening peaks',
           xlabel='Hour of day', ylabel='Mean crossings per observed hour',
           xticks=[0, 6, 12, 18, 23], ylim=(0, None))
    fig.tight_layout()
    plt.show()


def plot_weekday_weekend_profiles(profiles):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, day_type, color, marker in zip(
        axes, ['Weekday', 'Weekend'], [FTE_BLUE, FTE_YELLOW], ['o', 's'],
    ):
        profile = profiles.loc[day_type].sort_index()
        ax.plot(profile.index, profile.values, color=color, marker=marker, markersize=4)
        ax.set(title=f'{day_type}s', xlabel='Hour of day',
               xticks=[0, 6, 12, 18, 23], ylim=(0, None))
    axes[0].set_ylabel('Mean crossings per observed hour')
    fig.tight_layout()
    plt.show()


def plot_weekday_profile(by_dow):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(by_dow.index, by_dow.values, color=FTE_BLUE, marker='o')
    ax.set(title='Observed hourly traffic is higher on weekdays', xlabel='Day of week',
           ylabel='Mean crossings per observed hour', ylim=(0, None))
    ax.set_xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    fig.tight_layout()
    plt.show()


def plot_seasonal_profile(monthly_daily_mean):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(monthly_daily_mean.index, monthly_daily_mean.values, color=FTE_BLUE)
    ax.set(title='Daily traffic follows a seasonal cycle', xlabel='Date',
           ylabel='Mean crossings per complete recorded day', ylim=(0, None))
    fig.tight_layout()
    plt.show()


def plot_co2_raw(co2):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(co2['DecDate'], co2['Avg'], color=FTE_BLUE)
    ax.set(title='Negative spikes expose a missing-value code', xlabel='Year',
           ylabel='Monthly mean CO₂ (ppm)')
    fig.tight_layout()
    plt.show()


def plot_co2_day_counts(known_day_counts):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(known_day_counts['Yr'], known_day_counts['Days'], s=18, color=FTE_BLUE)
    ax.set(title='Reported daily means: documented counts only', xlabel='Year',
           ylabel='Daily means contributing to a month', ylim=(0, None))
    fig.tight_layout()
    plt.show()


def plot_co2_treatments(co2_drop, co2_na, co2_impute, missing_avg):
    """Show the consequences of the notebook's three explicit treatments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    titles = ['Drop: line bridges gaps', 'NaN: gaps stay visible', 'Fill: estimates are marked']
    for ax, frame, title in zip(axes, [co2_drop, co2_na, co2_impute], titles):
        focus = frame['Yr'].between(1958, 1965)
        ax.plot(frame.loc[focus, 'DecDate'], frame.loc[focus, 'Avg'], color=FTE_BLUE)
        ax.set(title=title, xlabel='Year')
    filled = co2_impute['Yr'].between(1958, 1965) & missing_avg
    axes[2].scatter(co2_impute.loc[filled, 'DecDate'], co2_impute.loc[filled, 'Avg'],
                    marker='X', s=70, color=FTE_RED, label='Interpolated month', zorder=3)
    axes[2].legend()
    axes[0].set_ylabel('Monthly mean CO₂ (ppm)')
    fig.tight_layout()
    plt.show()


def _finish_axes(ax):
    """Keep new EDA figures legible without changing earlier figures."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.8)
    ax.tick_params(labelsize=11)


def plot_housing_maps(locations, random_state=538):
    """Keep coordinates and the value distribution fixed while shuffling values."""
    values = locations['median_house_value'].to_numpy()
    shuffled = np.random.default_rng(random_state).permutation(values)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.7), sharex=True, sharey=True,
                             layout='constrained')
    for ax, colors, title in zip(
        axes, [values, shuffled],
        ['Observed locations', 'Same values, shuffled locations'],
    ):
        points = ax.scatter(locations['longitude'], locations['latitude'],
                            c=colors, cmap='viridis', vmin=0, vmax=500001,
                            s=7, edgecolors='none', rasterized=True)
        ax.set(title=title, xlabel='Longitude', xlim=(-124.7, -114.0),
               ylim=(32.3, 42.2), xticks=[-124, -122, -120, -118, -116, -114])
        ax.set_aspect(1 / np.cos(np.deg2rad(37)))
        _finish_axes(ax)
    axes[0].set_ylabel('Latitude')
    fig.suptitle('House values form geographic clusters',
                 fontsize=18, fontweight='bold')
    colorbar = fig.colorbar(points, ax=axes, orientation='horizontal',
                           fraction=0.07, pad=0.02, shrink=0.78, aspect=40)
    colorbar.set_ticks([0, 100000, 200000, 300000, 400000, 500001],
                      labels=['$0', '$100k', '$200k', '$300k', '$400k', '$500k+'])
    colorbar.set_label('Mean of block-group median home values (1990 USD)', fontsize=12)
    plt.show()


def nearest_location_pairs(locations, random_state=538):
    """Find nearest distinct coordinates on a sphere, excluding each point itself.

    Input rows must have unique latitude/longitude pairs. Euclidean distance on
    the unit sphere has the same nearest-neighbour ordering as great-circle
    distance. Reuse the identical directed pairs after one seeded permutation.
    Each location contributes one pair, so reciprocal pairs appear twice.
    """
    if locations.duplicated(['latitude', 'longitude']).any():
        raise ValueError('Aggregate repeated coordinates before finding neighbours.')
    if len(locations) < 2:
        raise ValueError('At least two distinct locations are required.')
    lat, lon = np.deg2rad(locations[['latitude', 'longitude']].to_numpy()).T
    xyz = np.column_stack([np.cos(lat) * np.cos(lon),
                           np.cos(lat) * np.sin(lon), np.sin(lat)])
    distance, indices = cKDTree(xyz).query(xyz, k=2)
    nearest = indices[:, 1]
    values = locations['median_house_value'].to_numpy()
    shuffled = np.random.default_rng(random_state).permutation(values)
    return pd.DataFrame({
        'location_value': values,
        'neighbour_value': values[nearest],
        'observed_gap': np.abs(values - values[nearest]),
        'shuffled_gap': np.abs(shuffled - shuffled[nearest]),
        'distance_km': 2 * 6371.0088 * np.arcsin(np.clip(distance[:, 1] / 2, 0, 1)),
        'neighbour_row': nearest,
    })


def plot_neighbour_differences(pairs):
    """Compare cumulative absolute value gaps for the same nearest-location pairs."""
    fig, ax = plt.subplots(figsize=(11, 5.1), layout='constrained')
    for key, label, color, linestyle, offset in [
        ('observed_gap', 'Observed locations', FTE_BLUE, '-', (18, 70)),
        ('shuffled_gap', 'Shuffled values', FTE_YELLOW, '--', (22, -80)),
    ]:
        values = np.sort(pairs[key].to_numpy())
        cumulative = np.arange(1, len(values) + 1) / len(values)
        median = np.median(values)
        ax.plot(values, cumulative, color=color, linestyle=linestyle,
                linewidth=2.7, label=label)
        ax.vlines(median, 0, 0.5, color=color, linewidth=1.5, linestyle=':')
        ax.scatter([median], [0.5], s=70, color=color, edgecolor='white', zorder=4)
        ax.annotate(f'{label}\nMedian gap: ${median:,.0f}',
                    xy=(median, 0.5), xytext=offset, textcoords='offset points',
                    fontsize=11, color='#333333',
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': 1.4},
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.92})
    ax.axhline(0.5, color='#777777', linestyle=':', linewidth=1, zorder=0)
    ax.set(title='Nearby locations have smaller value differences',
           xlabel='Absolute difference between nearest-location values (1990 USD)',
           ylabel='Share of location pairs at or below this difference',
           xlim=(0, 500001), ylim=(0, 1.02))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'${value / 1000:.0f}k'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.0%}'))
    ax.legend(loc='lower right', frameon=False)
    _finish_axes(ax)
    plt.show()


def plot_earthquake_magnitudes(earthquakes):
    """Show reviewed local magnitudes with values below zero highlighted."""
    magnitudes = earthquakes['mag']
    negative = magnitudes.lt(0)
    bins = np.round(np.arange(-0.6, 4.61, 0.2), 1)
    fig, ax = plt.subplots(figsize=(11, 4.9), layout='constrained')
    ax.hist([magnitudes[negative], magnitudes[~negative]], bins=bins, stacked=True,
            color=[FTE_YELLOW, FTE_BLUE], edgecolor='white', linewidth=0.7,
            label=['Magnitude below zero', 'Magnitude zero or above'])
    ax.axvline(0, color='#333333', linewidth=1.4, linestyle='--')
    ax.set(title='Should the negative earthquake magnitudes stay?',
           xlabel='Local magnitude (ML)', ylabel='Reviewed earthquakes', xlim=(-0.7, 4.6))
    ax.annotate(f'{negative.sum()} events below zero', xy=(-0.2, 10),
                xytext=(0.07, 0.83), textcoords='axes fraction',
                arrowprops={'arrowstyle': '->', 'color': '#555555'},
                fontsize=12, bbox={'facecolor': 'white', 'edgecolor': 'none'})
    ax.legend(frameon=False, loc='upper right')
    _finish_axes(ax)
    plt.show()


def plot_magnitude_context(earthquakes, nonnegative_only):
    """Pair a schematic local-magnitude scale with the effect of a wrong filter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1.4]})
    ax = axes[0]
    ratio = np.logspace(-1.3, 2.2, 200)
    ax.plot(ratio, np.log10(ratio), color=FTE_BLUE, linewidth=2.5)
    for amplitude, magnitude in zip([0.1, 1, 10, 100], [-1, 0, 1, 2]):
        ax.scatter(amplitude, magnitude, s=80,
                   color=FTE_YELLOW if magnitude < 0 else FTE_BLUE,
                   edgecolor='white', zorder=4)
    ax.axhline(0, color='#777777', linestyle='--', linewidth=1.1)
    ax.set_xscale('log')
    ax.set(title='A logarithmic magnitude scale',
           xlabel='Calibrated amplitude ratio A/A₀\n(schematic scale)',
           ylabel='Local magnitude (ML)', yticks=[-1, 0, 1, 2],
           xlim=(0.05, 160), ylim=(-1.4, 2.3))
    ax.set_xticks([0.1, 1, 10, 100], labels=['0.1', '1', '10', '100'])
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.annotate('0.1 times the reference\ngives magnitude −1',
                xy=(0.1, -1), xytext=(0.38, -0.9), fontsize=10,
                arrowprops={'arrowstyle': '-', 'color': '#777777'})
    ax = axes[1]
    bins = np.round(np.arange(-0.6, 4.61, 0.2), 1)
    ax.hist(earthquakes['mag'], bins=bins, color=FTE_YELLOW, alpha=0.75,
            edgecolor='white', linewidth=0.7, label=f'Keep reviewed events: {len(earthquakes):,}')
    ax.hist(nonnegative_only['mag'], bins=bins, histtype='step', color=FTE_RED,
            linewidth=2, label=f'Drop negatives: {len(nonnegative_only):,}')
    ax.axvline(0, color='#777777', linestyle='--', linewidth=1)
    removed = len(earthquakes) - len(nonnegative_only)
    ax.set(title=f'Dropping negatives removes {removed} events',
           xlabel='Local magnitude (ML)', ylabel='Reviewed earthquakes', xlim=(-0.7, 4.6))
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    for ax in axes:
        _finish_axes(ax)
    plt.show()
