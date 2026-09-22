"""Display and reusable cleaning helpers for CME538 Lecture 3.3."""

import matplotlib.pyplot as plt
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


def configure_notebook():
    """Apply the lecture guide's plotting and compact table defaults once."""
    plt.style.use('fivethirtyeight')
    sns.set_context('notebook')
    sns.set_palette(FTE_COLORS)
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
    pd.set_option('display.max_rows', 12)
    pd.set_option('display.max_columns', 8)
    pd.set_option('display.max_colwidth', 42)
    pd.set_option('display.precision', 2)


def audit_vintage_years(values, *, reference_year=2026, min_year=1850):
    """Apply the documented classroom retailer policy without guessing centuries."""
    recorded = pd.Series(values, dtype='string').copy(deep=True)
    text = recorded.str.strip()
    four_digits = text.str.fullmatch(r'[0-9]{4}', na=False)
    two_digits = text.str.fullmatch(r'[0-9]{2}', na=False)
    year = pd.to_numeric(text.where(four_digits), errors='coerce').astype('Int64')
    accepted = (four_digits & year.between(min_year, reference_year)).fillna(False)

    decision = pd.Series('invalid format', index=recorded.index, dtype='string')
    decision.loc[text.isna() | text.eq('').fillna(False)] = 'missing value'
    decision.loc[two_digits] = 'ambiguous two-digit year'
    decision.loc[(four_digits & year.lt(min_year)).fillna(False)] = 'before retailer range'
    decision.loc[(four_digits & year.gt(reference_year)).fillna(False)] = 'future year'
    decision.loc[accepted] = 'accepted'

    return pd.DataFrame({
        'recorded_year': recorded,
        'accepted_year': year.where(accepted),
        'age_years': (reference_year - year).where(accepted),
        'decision': decision,
    })


def plot_wine_ages(wine, *, reference_year=2026):
    """Reveal calculated ages; direct labels also distinguish suspicious values."""
    ambiguous = wine['recorded_year'].str.len().ne(4)
    colors = [FTE_RED if flag else FTE_BLUE for flag in ambiguous]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.barh(wine['recorded_year'], wine['naive_age_years'], color=colors)
    for bar, flag in zip(bars, ambiguous):
        if flag:
            bar.set_hatch('//')
            bar.set_edgecolor('white')
    ax.bar_label(
        bars, labels=[f'{age:,.0f} years' for age in wine['naive_age_years']], padding=5,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(1, wine['naive_age_years'].max() * 1.2))
    ax.set_xlabel(f'Naively computed age in {reference_year} (years)')
    ax.set_ylabel('Recorded vintage string')
    ax.set_title('Valid arithmetic can encode an invalid interpretation')
    fig.tight_layout()
    plt.show()


def preview_flagged_claims(claims, flags, *, rows=5):
    """Attach all review reasons without displaying a wide Boolean matrix."""
    selected = flags.loc[flags.any(axis=1)].head(rows)
    preview = claims.loc[selected.index, ['ref_number', 'trip_days', 'total']].copy()
    preview.columns = ['Claim', 'Trip days', 'Total (CAD)']
    short_labels = {
        'negative monetary value': 'Negative amount',
        'date outside 2003-2024 snapshot': 'Date outside snapshot',
        'end before start': 'End before start',
    }
    preview['Review reason'] = selected.rename(columns=short_labels).apply(
        lambda row: '; '.join(row.index[row]), axis=1,
    )
    return preview


def plot_claim_totals(totals):
    """Show the same claim totals at full scale and at a clearly labelled zoom."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axes:
        sns.boxplot(x=totals, color=FTE_BLUE, showfliers=True,
                    fliersize=2, ax=ax)
        ax.set_xlabel('Reported claim total (CAD)')
        ax.set_ylabel('')
        ax.ticklabel_format(axis='x', style='plain')
    axes[0].set_xlim(left=0)
    axes[0].set_title('A long right tail at full scale')
    axes[1].set_xlim(0, totals.quantile(0.99))
    axes[1].set_title('Zoom to the central 99%')
    fig.tight_layout()
    plt.show()
