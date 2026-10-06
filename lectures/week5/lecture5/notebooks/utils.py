"""Shared figure styling for the Lecture 5.5 demonstrations."""

from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# Figure helpers draw supplied data. Calculations stay in the notebook.
import datetime
import numpy as np
import pandas as pd
import seaborn as sns


def plot_programming_questions(stacked_overflow):
    plt.figure(figsize=(10, 5))
    plt.title('Monthly Stack Overflow questions by programming tag')
    ax = sns.lineplot(data=stacked_overflow, x='month', y='python', label='Python')
    sns.lineplot(ax=ax, data=stacked_overflow, x='month', y='r', label='R', linestyle='--')
    sns.lineplot(ax=ax, data=stacked_overflow, x='month', y='matlab', label='MATLAB', linestyle=':')
    sns.lineplot(ax=ax, data=stacked_overflow, x='month', y='pandas', label='pandas', linestyle='-.')
    ax.legend(fontsize=10)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Questions per month', fontsize=12)
    plt.show()


def plot_matlab_questions(stacked_overflow):
    plt.figure(figsize=(10, 5))
    plt.title('Monthly MATLAB questions change across the record')
    ax = sns.lineplot(data=stacked_overflow, x='month', y='matlab', label='Matlab')
    ax.legend(fontsize=10)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Questions per month', fontsize=12)
    plt.show()


def plot_price_history(sp500):
    plt.figure(figsize=(10, 5))
    plt.title('SPY daily closing price')
    ax = sns.lineplot(data=sp500, x=sp500.index, y='Close', label='SPY', linewidth=0.8)
    ax.legend(fontsize=10)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Closing price (USD per share)', fontsize=12)
    plt.show()


def plot_price_changes(sp500):
    plt.figure(figsize=(10, 5))
    plt.title('SPY change from the previous trading day')
    ax = sns.lineplot(data=sp500, x=sp500.index[0:], y='change', label='SPY', linewidth=0.8)
    ax.legend(fontsize=10)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price change (USD per share)', fontsize=12)
    plt.show()


def plot_price_window(sp500):
    plt.figure(figsize=(10, 5))
    plt.title('SPY daily closing price')
    ax = sns.lineplot(data=sp500, x=sp500.index, y='Close', label='SPY')
    ax.legend(fontsize=10)
    ax.set_xlim([datetime.date(2013, 1, 1), datetime.date(2015, 1, 1)])
    ax.set_ylim([140, 220])
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Closing price (USD per share)', fontsize=12)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.tight_layout()
    plt.show()


def plot_bridge_history(bike_data):
    plt.figure(figsize=(10, 5))
    plt.title('Fremont Bridge hourly bicycle crossings')
    ax = sns.lineplot(data=bike_data, x='Date', y='Fremont Bridge West Sidewalk', linewidth=0.8)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Crossings per hour', fontsize=12)
    plt.show()


def plot_reported_cases(canada_covid):
    plt.figure(figsize=(10, 5))
    plt.title('Reported COVID-19 cases in Canada: seven-day mean')
    ax = sns.lineplot(data=canada_covid, x='date', y='avgtotal_last7')
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Reported cases per day (seven-day mean)', fontsize=12)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plt.tight_layout()
    plt.show()


def plot_earthquake_counts(earthquakes):
    plt.figure(figsize=(10, 5))
    plt.title('Monthly earthquakes in Oklahoma')
    ax = sns.lineplot(data=earthquakes, x=earthquakes.index, y='time')
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Earthquakes per month', fontsize=12)
    plt.show()


def plot_temperature_and_rolling_mean(city_temperatures, city_temperatures_yearly):
    plt.figure(figsize=(10, 5))
    plt.title('Monthly temperature in Melbourne')
    ax = sns.lineplot(data=city_temperatures, x='dt', y='AverageTemperature', label='Monthly', linewidth=0.8)
    ax = sns.lineplot(data=city_temperatures_yearly, x=city_temperatures_yearly.index, y='AverageTemperature', label='Trailing 12-month mean', color=FTE_GREEN, linestyle='--')
    ax.legend(fontsize=10)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    plt.show()


def plot_temperature_rolling_mean(city_temperatures_yearly):
    plt.figure(figsize=(10, 5))
    plt.title('Monthly temperature in Melbourne\n(12 month rolling average)')
    ax = sns.lineplot(data=city_temperatures_yearly, x=city_temperatures_yearly.index, y='AverageTemperature', label='Trailing 12-month mean', color=FTE_GREEN, linestyle='--')
    ax.legend(fontsize=10)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    plt.show()


def plot_lynx_counts(lynx):
    plt.figure(figsize=(10, 5))
    plt.title('Annual lynx trappings in Canada')
    ax = sns.lineplot(data=lynx, x='time', y='value', marker='o',
                     markersize=6, markeredgecolor='white', markeredgewidth=0.8)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual trappings', fontsize=12)
    plt.show()


def plot_lynx_lag_comparison(lynx, lag, corr, lags, corrs):
    fig, axs = plt.subplots(nrows=3, figsize=(8, 12))
    axs[0].set_title('Annual counts with an {}-year shift'.format(lag), fontsize=16)
    axs[0].plot(lynx['time'], lynx['value'], '-o', ms=6,
                markeredgecolor='white', markeredgewidth=0.8, label='Recorded years')
    axs[0].plot(lynx['time']+lag, lynx['value'], '--o', ms=5.5,
                markeredgecolor='white', markeredgewidth=0.8, label=f'Shifted {lag} years')
    axs[0].legend()
    axs[0].set_xlim([1821, 1934])
    axs[0].set_ylabel('Annual trappings', fontsize=12)
    axs[0].set_xlabel('Year', fontsize=12)

    axs[1].set_title('Correlation at this lag: {}'.format(np.round(corr, 2)), fontsize=16)
    axs[1].plot(lynx['value'].iloc[0:-lag], lynx['value'].iloc[lag:], '.')
    axs[1].set_aspect('equal', 'box')
    axs[1].set_xlabel('Annual trappings', fontsize=12)
    axs[1].set_ylabel('Annual trappings (lag: {})'.format(lag), fontsize=12)

    axs[2].set_title('Association across annual lags')
    axs[2].set_xlabel('Lag (years)', fontsize=12)
    axs[2].set_ylabel('Autocorrelation', fontsize=12)
    axs[2].set_xlim([0, 11])
    axs[2].set_ylim([-1, 1])

    axs[2].plot(lags, corrs, '-o', markeredgecolor='white', markeredgewidth=0.8)

    # Narrow the gap between the plots
    plt.subplots_adjust(hspace=0.35)
    plt.show()


def plot_lynx_autocorrelation(lynx):
    pd.plotting.autocorrelation_plot(lynx['value'])
    plt.title('Lynx counts show association across several lags')
    plt.xlabel('Lag (years)')
    plt.ylabel('Autocorrelation')
    plt.tight_layout()
    plt.show()


def plot_brick_production(bricks):
    plt.figure(figsize=(10, 5))
    plt.title('Australian quarterly clay brick production: 1955–1994')
    ax = sns.lineplot(data=bricks, x='date', y='val', marker='o',
                     markersize=6, markeredgecolor='white', markeredgewidth=0.8)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Bricks (million units)', fontsize=12)
    plt.show()


def plot_white_noise(white_noise):
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=white_noise, marker='o',
                     markersize=6, markeredgecolor='white', markeredgewidth=0.8)
    plt.title('Independent random values fluctuate without a trend')
    plt.xlabel('Observation number')
    plt.ylabel('Simulated value')
    plt.tight_layout()
    plt.show()


def plot_white_noise_autocorrelation(white_noise):
    pd.plotting.autocorrelation_plot(white_noise)
    plt.title('White noise has small sample correlations at most lags')
    plt.xlabel('Lag (observations)')
    plt.ylabel('Autocorrelation')
    plt.tight_layout()
    plt.show()


def plot_cattle_counts(cattle):
    plt.figure(figsize=(10, 5))
    plt.title('Monthly cattle slaughtered in Canada')
    ax = sns.lineplot(data=cattle, x=cattle.index, y='NumHd_NmbTetes', marker='o',
                     markersize=6, markeredgecolor='white', markeredgewidth=0.8)
    ax.xaxis.set_tick_params(labelsize=11)
    ax.yaxis.set_tick_params(labelsize=11)
    ax.set_xlim([datetime.date(1998, 1, 1), datetime.date(2003, 1, 1)])
    ax.set_ylim([240000, 380000])
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cattle slaughtered', fontsize=12)
    plt.show()


def plot_cattle_autocorrelation(cattle):
    plt.figure(figsize=(10, 5))
    pd.plotting.autocorrelation_plot(cattle.loc['1998-01-01':'2003-01-01', 'NumHd_NmbTetes'])
    plt.title('Cattle counts show association across reporting periods')
    plt.xlabel('Lag (reporting periods)')
    plt.ylabel('Autocorrelation')
    plt.tight_layout()
    plt.show()


def plot_stable_simulation(time_series):
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=time_series, linewidth=0.8)
    plt.title('The simulated mean and spread stay stable')
    plt.xlabel('Observation number')
    plt.ylabel('Simulated value')
    plt.tight_layout()
    plt.show()


def plot_changing_spread(time_series):
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=time_series, linewidth=0.8)
    plt.title('The simulated spread increases over time')
    plt.xlabel('Observation number')
    plt.ylabel('Simulated value')
    plt.tight_layout()
    plt.show()


def plot_changing_mean(time_series):
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=time_series, linewidth=0.8)
    plt.title('The simulated mean increases over time')
    plt.xlabel('Observation number')
    plt.ylabel('Simulated value')
    plt.tight_layout()
    plt.show()


def plot_random_walk(time_series):
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=time_series, linewidth=0.8)
    plt.title('Accumulated random changes form a random walk')
    plt.xlabel('Observation number')
    plt.ylabel('Simulated level')
    plt.tight_layout()
    plt.show()


def plot_differences(time_series_diff):
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=time_series_diff, linewidth=0.8)
    plt.title('First differences recover the random changes')
    plt.xlabel('Observation number')
    plt.ylabel('Change per observation')
    plt.tight_layout()
    plt.show()


def plot_ordered_and_shuffled(week, shuffled_week):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, sharey=True)
    axes[0].plot(week.index, week, color=FTE_BLUE)
    axes[1].plot(week.index, shuffled_week.to_numpy(),
                 color=FTE_RED, linestyle='--')
    for ax, title in zip(axes, ['Recorded time order', 'Same counts assigned to shuffled times']):
        ax.set_title(title)
        ax.set_ylabel('Crossings per hour')
    axes[1].set_xlabel('Date in May 2016')
    fig.tight_layout()
    plt.show()


def plot_hourly_and_daily(week, daily_crossings):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(week.index, week, color=FTE_BLUE)
    axes[0].set(title='Hourly peaks within one week', xlabel='Date', ylabel='Crossings per hour')
    daily_crossings.loc['2016-05-02':'2016-05-08'].plot.bar(ax=axes[1], color=FTE_BLUE)
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    axes[1].set(title='Daily totals hide the within-day pattern', xlabel='Day', ylabel='Crossings per day')
    fig.tight_layout()
    plt.show()


def plot_weekly_heatmap(profile):
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(profile, cmap='viridis', ax=ax,
                yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                cbar_kws={'label': 'Mean crossings per observed hour'})
    ax.set(title='Weekdays and weekends have different hourly profiles',
           xlabel='Hour of day', ylabel='Day of week')
    fig.tight_layout()
    plt.show()


def plot_temperature_profile(temperature, seasonal_profile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(temperature.index, temperature, linewidth=1)
    axes[0].set(title='The annual temperature pattern repeats', xlabel='Year', ylabel='Temperature (°C)')
    axes[1].plot(range(1, 13), seasonal_profile, marker='o',
                 markeredgecolor='white', markeredgewidth=0.8)
    axes[1].set_xticks(range(1, 13), ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    axes[1].set(title='Melbourne is warmest near January', xlabel='Calendar month', ylabel='Mean temperature (°C)')
    fig.tight_layout()
    plt.show()


def plot_decomposition(temperature, trend, seasonal, remainder):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, values, title, color in zip(axes,
            [temperature, trend, seasonal, remainder],
            ['Observed temperature', 'Estimated slower variation', 'Estimated annual pattern', 'Remainder'],
            [FTE_BLUE, FTE_GREEN, FTE_YELLOW, FTE_GRAY]):
        ax.plot(values.index, values, color=color)
        ax.set_title(title)
        ax.set_ylabel('Temperature (°C)')
    axes[-1].set_xlabel('Year')
    fig.tight_layout()
    plt.show()


def plot_differenced_autocorrelations(correlations, lags):
    titles = ['Raw monthly temperature', 'Difference from previous month', 'Difference from the same month last year']
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    for ax, correlation, title in zip(axes, correlations, titles):
        ax.stem(lags, correlation, basefmt=' ')
        ax.axhline(0, color=FTE_GRAY, linewidth=1)
        ax.set(title=title, ylabel='Correlation', ylim=(-1, 1))
    axes[-1].set(xlabel='Lag (months)', xticks=[1, 6, 12, 18, 24])
    fig.tight_layout()
    plt.show()


def plot_sensor_gap(reservoir, zero_filled_level):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(reservoir.index, reservoir['rainfall_mm'], color=FTE_BLUE)
    axes[0].set(title='Illustrative rainfall arrives in separate events', ylabel='Daily rainfall (mm)')
    axes[1].plot(reservoir.index, zero_filled_level,
                 color=FTE_RED, linestyle='--', label='Incorrect zero fill')
    axes[1].plot(reservoir.index, reservoir['water_level_m'],
                 color=FTE_BLUE, label='Recorded teaching values with a gap')
    axes[1].set(title='Missing readings differ from a zero level', ylabel='Water level (m)', xlabel='Date')
    axes[1].legend()
    fig.tight_layout()
    plt.show()


def plot_reservoir_weekly(weekly_rainfall, weekly_level):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(weekly_rainfall.index, weekly_rainfall, width=5, color=FTE_BLUE)
    axes[0].set(title='Sum rainfall depth within complete weeks', ylabel='Rainfall per week (mm)')
    axes[1].plot(weekly_level.index, weekly_level, marker='o', color=FTE_GREEN,
                 markeredgecolor='white', markeredgewidth=0.8)
    axes[1].set(title='Average level within complete weeks', ylabel='Mean water level (m)', xlabel='Week ending')
    fig.tight_layout()
    plt.show()


def plot_rolling_windows(level, trailing, centred):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(level.index, level, color=FTE_GRAY, linewidth=1, label='Daily level')
    ax.plot(trailing.index, trailing, color=FTE_BLUE, label='Trailing 7-day mean')
    ax.plot(centred.index, centred, color=FTE_RED, linestyle='--', label='Centred 7-day mean')
    ax.set(title='A centred window uses future readings', xlabel='Date', ylabel='Water level (m)')
    ax.legend()
    fig.tight_layout()
    plt.show()
