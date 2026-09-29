"""Display settings and repeated figure layouts for Lecture 4.1."""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cycler import cycler

BLUE = '#30A2DA'
RED = '#FC4F30'
GOLD = '#E5AE38'
GREEN = '#6D904F'
GRAY = '#8B8B8B'
PURPLE = '#810F7C'
COLORS = [BLUE, RED, GOLD, GREEN, GRAY, PURPLE]
SMOKER_COLORS = {False: BLUE, True: RED}


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


def plot_distribution_shapes(baby):
    """Compare three units separately, with stable mean and median markers."""
    examples = [
        ('Maternal Pregnancy Weight', 'Pregnancy weight', 'Weight (lb)'),
        ('Gestational Days', 'Gestation', 'Duration (days)'),
        ('Birth Weight', 'Birth weight', 'Weight (oz)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (column, title, label) in zip(axes, examples):
        sns.histplot(data=baby, x=column, bins=25, stat='density', ax=ax, color=BLUE)
        ax.axvline(baby[column].mean(), color=RED, label='Mean')
        ax.axvline(baby[column].median(), color=GRAY, linestyle='--', label='Median')
        ax.set(title=title, xlabel=label, ylabel='Density')
        ax.legend()
    plt.show()
