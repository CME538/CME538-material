"""Display settings and visual comparison layouts for Lecture 4.2."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.patches import Circle

BLUE = '#30A2DA'
RED = '#FC4F30'
GOLD = '#E5AE38'
GREEN = '#6D904F'
GRAY = '#8B8B8B'
PURPLE = '#810F7C'
COLORS = [BLUE, RED, GOLD, GREEN, GRAY, PURPLE]
GENDER_COLORS = {'Men': BLUE, 'Women': RED}


def configure_notebook():
    """Apply the course display style without suppressing warnings."""
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({
        'axes.prop_cycle': cycler(color=COLORS),
        'axes.titlesize': 16, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'axes.labelsize': 12,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'figure.titlesize': 16,
        'figure.figsize': (9, 5), 'figure.dpi': 110,
        'figure.autolayout': True,
    })
    sns.set_palette(COLORS)
    pd.set_option('display.max_columns', 12)


def show_color_scales():
    """Show ordered, centred, circular, and categorical colour mappings."""
    fig, axes = plt.subplots(4, 1, figsize=(9, 6.5))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    specs = [('Sequential: elevation', 'viridis', 0, 1000, 'Elevation (m)'),
             ('Diverging: departure from a reference', 'RdBu_r', -5, 5, 'Difference (°C)'),
             ('Cyclic: direction', 'twilight', 0, 360, 'Direction (degrees)')]
    for ax, (title, cmap, lo, hi, label) in zip(axes[:3], specs):
        ax.imshow(gradient, aspect='auto', cmap=cmap, extent=(lo, hi, 0, 1))
        ax.set(title=title, yticks=[], xlabel=label)
    axes[3].barh([''], [1], left=[0], color=BLUE, label='A')
    for i, color in enumerate(COLORS[1:4], 1):
        axes[3].barh([''], [1], left=[i], color=color, label=chr(65+i))
    axes[3].set(title='Qualitative: separate categories', yticks=[], xticks=[0.5, 1.5, 2.5, 3.5],
                xticklabels=['A', 'B', 'C', 'D'], xlabel='Unordered category')
    axes[3].grid(False)
    plt.show()


def compare_encodings(a, b):
    """Encode the same positive pair as length, angle, and circle area."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(['A', 'B'], [a, b], color=[BLUE, RED])
    axes[0].set(title='Length', ylabel='Illustrative units', ylim=(0, max(a, b)*1.2))
    axes[1].pie([a, b], labels=['A', 'B'], colors=[BLUE, RED], startangle=90)
    axes[1].set_title('Angle')
    for value, center, label, color in [(a, (0.3, 0.5), 'A', BLUE), (b, (0.8, 0.5), 'B', RED)]:
        radius = 0.26 * np.sqrt(value / max(a, b))
        axes[2].add_patch(Circle(center, radius, color=color))
        axes[2].text(center[0], 0.12, label, ha='center')
    axes[2].set(xlim=(0, 1), ylim=(0, 1), aspect='equal', title='Area')
    axes[2].axis('off')
    plt.show()


def plot_colormap_luminance(names):
    """Calculate relative luminance after decoding sRGB's transfer function."""
    values = np.linspace(0, 1, 256)
    for name in names:
        rgb = plt.get_cmap(name)(values)[:, :3]
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
        plt.plot(values, luminance, label=name)
    plt.title('Relative luminance along each colour map')
    plt.xlabel('Position along colour map')
    plt.ylabel('Relative luminance')
    plt.legend()
    plt.show()
