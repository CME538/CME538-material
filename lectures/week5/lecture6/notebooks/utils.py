"""Shared figure styling for the Lecture 5.6 demonstrations."""

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


def label_map(ax, title, projected=True):
    ax.set_title(title)
    ax.set_xlabel('Easting (m)' if projected else 'Longitude (degrees)')
    ax.set_ylabel('Northing (m)' if projected else 'Latitude (degrees)')
    ax.ticklabel_format(style='plain', useOffset=False)
    if projected:
        ax.set_aspect('equal')
    ax.tick_params(axis='x', rotation=25)

# Figure helpers draw supplied data. Calculations stay in the notebook.
import geopandas as gpd


def plot_original_toronto_layers(regions, bikeshare_stations_gdf, weather_station_gdf, uoft_gdf):
    # Plot to Toronto neighbourhoods shape file 
    ax = regions.plot(figsize=(15, 8), edgecolor='white', color=FTE_GRAY, alpha=0.35)

    # Plot the station location
    bikeshare_stations_gdf.plot(ax=ax, color=FTE_BLUE, edgecolor='k', label='Bike Stations')

    # Plot the weather station location
    weather_station_gdf.plot(ax=ax, marker='X', color=FTE_YELLOW, edgecolor='black', markersize=500, label='Weather Station')

    # Plot the weather station location
    uoft_gdf.plot(ax=ax, marker='P', color=FTE_YELLOW, edgecolor='black', markersize=500, label='UofT')

    plt.legend(fontsize=10, loc=2)
    plt.xlabel('Longitude (degrees)', fontsize=12)
    plt.ylabel('Latitude (degrees)', fontsize=12)
    plt.title('Bike stations, the weather station and UofT in the supplied snapshot')
    plt.tight_layout()
    plt.show()


def plot_downtown(regions, bikeshare_stations_gdf, uoft_gdf):
    fig, ax = plt.subplots(figsize=(9, 6))
    regions.plot(ax=ax, color=FTE_GRAY, edgecolor='white', alpha=0.25)
    bikeshare_stations_gdf.plot(ax=ax, color=FTE_BLUE, markersize=25, label='Bike stations')
    uoft_gdf.plot(ax=ax, color=FTE_YELLOW, marker='P', edgecolor='black', markersize=160, label='UofT')
    ax.set_xlim(-79.425, -79.36)
    ax.set_ylim(43.63, 43.68)
    label_map(ax, 'Downtown stations at a closer scale', projected=False)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_geometry_types(site, route, zone):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, shape, title in zip(axes, [site, route, zone],
            ['A site is a point', 'A route is a line', 'A zone is a polygon']):
        gpd.GeoSeries([shape]).plot(ax=ax, color=FTE_BLUE)
        ax.set(title=title, xlabel='Local x (m)', ylabel='Local y (m)',
               xlim=(0, 600), ylim=(0, 450), aspect='equal')
    fig.tight_layout()
    plt.show()


def plot_count_and_density(area_counts):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, column, title, units in zip(axes,
            ['station_count', 'stations_per_km2'],
            ['Station counts by neighbourhood', 'Station density by neighbourhood'],
            ['Stations', 'Stations per km²']):
        area_counts.plot(column=column, ax=ax, cmap='viridis', edgecolor='white',
                         linewidth=0.3, legend=True, legend_kwds={'label': units, 'shrink': 0.65})
        label_map(ax, title)
    fig.tight_layout()
    plt.show()


def plot_nearby_stations(regions_utm, stations_utm, nearby, uoft_utm, buffer_500m):
    fig, ax = plt.subplots(figsize=(9, 7))
    regions_utm.plot(ax=ax, color=FTE_GRAY, edgecolor='white', alpha=0.25)
    buffer_500m.plot(ax=ax, color=FTE_YELLOW, alpha=0.25, edgecolor='black')
    stations_utm.plot(ax=ax, color=FTE_GRAY, markersize=15, label='Other stations')
    nearby.plot(ax=ax, color=FTE_BLUE, marker='o', markersize=55, label='Within 500 m')
    uoft_utm.plot(ax=ax, color=FTE_YELLOW, marker='P', edgecolor='black', markersize=170, label='UofT point')
    x, y = uoft_utm.geometry.iloc[0].coords[0]
    ax.set_xlim(x - 850, x + 850)
    ax.set_ylim(y - 850, y + 850)
    label_map(ax, 'A 500 m radius selects nearby station locations')
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.show()


def plot_radius_comparison(radii, counts):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(radius) for radius in radii], counts, color=FTE_BLUE)
    ax.set(title='A larger radius includes more station locations',
           xlabel='Straight-line radius (m)', ylabel='Number of stations')
    ax.bar_label(ax.containers[0], padding=3)
    ax.margins(y=0.15)
    fig.tight_layout()
    plt.show()


def plot_library_proximity(regions_utm, stations_utm, nearest):
    fig, ax = plt.subplots(figsize=(11, 7))
    regions_utm.plot(ax=ax, color=FTE_GRAY, edgecolor='white', alpha=0.2)
    stations_utm.plot(ax=ax, color=FTE_GRAY, markersize=6, label='Bike stations')
    nearest.plot(column='nearest_station_m', ax=ax, cmap='viridis', marker='^',
                 markersize=70, edgecolor='black', linewidth=0.4, legend=True,
                 legend_kwds={'label': 'Nearest-station distance (m)', 'shrink': 0.7})
    label_map(ax, 'Library triangles reveal variation in station proximity')
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.show()


def plot_library_connections(regions_utm, stations_utm, farthest, connections):
    fig, ax = plt.subplots(figsize=(11, 7))
    regions_utm.plot(ax=ax, color=FTE_GRAY, edgecolor='white', alpha=0.2)
    stations_utm.plot(ax=ax, color=FTE_GRAY, markersize=5)
    connections.plot(ax=ax, color=FTE_BLUE, linewidth=1.2, label='Straight-line link')
    farthest.plot(ax=ax, color=FTE_YELLOW, marker='^', edgecolor='black', markersize=80, label='Library')
    for _, row in farthest.iterrows():
        ax.annotate(row['BranchName'], (row.geometry.x, row.geometry.y),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    label_map(ax, 'These five branches have the longest nearest-station links')
    ax.legend()
    fig.tight_layout()
    plt.show()
