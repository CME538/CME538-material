"""Display and extraction helpers for the prepared Lecture 3.2 demonstrations."""

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from cycler import cycler

FOLDER = Path(__file__).resolve().parent
FTE_COLORS = ['#30A2DA', '#FC4F30', '#E5AE38', '#6D904F', '#8B8B8B', '#810F7C']


def configure_notebook():
    """Use the explicit shared CME538 plotting and compact table defaults."""
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
    pd.set_option('display.max_rows', 8)
    pd.set_option('display.max_columns', 6)
    pd.set_option('display.max_colwidth', 45)


def load_http_snapshot():
    """Replay saved 2024 response metadata and HTML without making a request.

    The text body was preserved in a notebook, not as the original wire bytes.
    Encoding it as UTF-8 reconstructs the content/text interface for teaching.
    """
    metadata = json.loads((FOLDER / 'civmin_response.json').read_text(encoding='utf-8'))
    response = requests.Response()
    response.status_code = metadata['status_code']
    response.url = metadata['url']
    response.encoding = metadata['encoding']
    response.headers.update(metadata['headers'])
    response._content = (FOLDER / 'civmin_response.html').read_text(encoding='utf-8').encode('utf-8')
    with requests.Session() as session:
        session.trust_env = False
        response.request = session.prepare_request(requests.Request('GET', response.url))
    return response


def make_http_error_example():
    """Construct an explicit 404 example locally; no resource was requested."""
    response = requests.Response()
    response.status_code = 404
    response.reason = 'Not Found'
    response.url = 'https://example.invalid/missing-resource'
    response.encoding = 'utf-8'
    response._content = b'Resource not found (constructed classroom example).'
    return response


def json_print(obj):
    """Format JSON for the reference notebook's inspection step."""
    print(json.dumps(obj, indent=4, ensure_ascii=False))


def plot_bitcoin_history(daily_prices):
    """Plot preserved historical samples without implying intraday coverage."""
    fig, ax = plt.subplots(figsize=(10, 4.6), layout='constrained')
    ax.plot(daily_prices.index, daily_prices['Close'], color=FTE_COLORS[0], linewidth=2)
    ax.set(title='Bitcoin prices from September 2023 to September 2024',
           xlabel='Date', ylabel='Last sampled price per day (USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'${value:,.0f}'))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3)
    plt.show()


def plot_coin_candlesticks(monthly_prices):
    """Display monthly summaries of the original historical chart values."""
    figure = go.Figure(go.Candlestick(
        x=monthly_prices.index.to_pydatetime().tolist(),
        open=monthly_prices['Open'].tolist(),
        high=monthly_prices['High'].tolist(),
        low=monthly_prices['Low'].tolist(),
        close=monthly_prices['Close'].tolist(),
        increasing_line_color=FTE_COLORS[0],
        decreasing_line_color=FTE_COLORS[1],
        name='Sampled BTC prices',
    ))
    figure.update_layout(
        title={'text': 'Monthly Bitcoin price summaries', 'x': 0.06},
        xaxis_title='Month', yaxis_title='Sampled price (USD)',
        template='plotly_white', paper_bgcolor='#F0F0F0', plot_bgcolor='#F0F0F0',
        font={'family': 'Arial', 'size': 14, 'color': '#333333'},
        height=470, margin={'l': 85, 'r': 30, 't': 80, 'b': 90},
        xaxis_rangeslider_visible=False, showlegend=False,
        annotations=[{'text': 'First and last months cover partial periods.',
                      'xref': 'paper', 'yref': 'paper', 'x': 0, 'y': -0.20,
                      'showarrow': False, 'xanchor': 'left', 'font': {'size': 12}}],
    )
    figure.update_xaxes(dtick='M2', tickformat='%b %Y', showgrid=False)
    figure.update_yaxes(tickprefix='$', tickformat=',.0f', gridcolor='#DDDDDD')
    return figure


def plot_hyperspectral_band(filename, band_index=55):
    """Read only one band and apply the dataset's scale/no-data metadata."""
    with h5py.File(filename, 'r') as dataset:
        reflectance = dataset['SJER/Reflectance']
        cube = reflectance['Reflectance_Data']
        band = cube[:, :, band_index].astype(float)
        no_data = float(np.asarray(cube.attrs['Data_Ignore_Value']).item())
        scale = float(np.asarray(cube.attrs['Scale_Factor']).item())
        band[band == no_data] = np.nan
        band /= scale
        wavelength = float(reflectance['Metadata/Spectral_Data/Wavelength'][band_index])
        map_info = np.asarray(reflectance['Metadata/Coordinate_System/Map_Info'][()]).item()
        if isinstance(map_info, bytes):
            map_info = map_info.decode('utf-8')
        fields = [field.strip() for field in str(map_info).split(',')]
        # This course file is a spatial subset. Its dataset extent is authoritative;
        # Map_Info retains the original tile origin and would shift this crop by 500 m.
        extent = tuple(float(value) for value in cube.attrs['Spatial_Extent_meters'])
        coordinate_units = next((f.split('=', 1)[1] for f in fields if f.lower().startswith('units=')), 'map units')
        if fields[0].upper() == 'UTM' and len(fields) >= 9:
            coordinate_label = f'UTM zone {fields[7]} {fields[8]}'
        else:
            coordinate_label = fields[0]

    fig, ax = plt.subplots(figsize=(8.6, 5.2), layout='constrained')
    image = ax.imshow(band, origin='upper', extent=extent, cmap='viridis')
    ax.set_title(f'One spectral band reveals spatial variation ({wavelength:.0f} nm)')
    unit_label = 'm' if coordinate_units.lower() in {'meters', 'metres', 'meter', 'metre'} else coordinate_units
    ax.set_xlabel(f'Easting ({unit_label}; {coordinate_label})')
    ax.set_ylabel(f'Northing ({unit_label})')
    ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    ax.tick_params(axis='x', labelrotation=20)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label='Reflectance (fraction)')
    plt.show()
    return fig
