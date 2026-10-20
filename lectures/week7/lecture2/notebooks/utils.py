"""Visual helpers for CME538 Lecture 7.2 (Classification II)."""

from functools import lru_cache
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageColor, ImageOps


_PHOTO_DIR = Path(__file__).resolve().parent / "animal_photos"

# Colour, pale fill, and a plain-language reading of each outcome.
_OUTCOMES = {
    "TP": ("#1B7837", "#D9F0D3", "actual cat + predicted cat", "correct detection"),
    "FP": ("#E66101", "#FEE8C8", "actual not-cat + predicted cat", "false alarm"),
    "FN": ("#762A83", "#E7D4E8", "actual cat + predicted not-cat", "miss"),
    "TN": ("#2166AC", "#D1E5F0", "actual not-cat + predicted not-cat", "correct rejection"),
}

# These fixed scores are deliberately imperfect teaching data, not the output of a
# trained image model. Every photograph appears once. The order mixes the four outcomes
# at T = 0.50 so that a filtered view is visually meaningful.
_ANIMALS = [
    ("cat_01.jpg", True, 0.93),
    ("dog_02.jpg", False, 0.74),
    ("cat_07.jpg", True, 0.45),
    ("dog_01.jpg", False, 0.47),
    ("cat_02.jpg", True, 0.88),
    ("dog_03.jpg", False, 0.39),
    ("dog_07.jpg", False, 0.66),
    ("cat_03.jpg", True, 0.82),
    ("dog_04.jpg", False, 0.34),
    ("cat_09.jpg", True, 0.31),
    ("cat_04.jpg", True, 0.78),
    ("dog_08.jpg", False, 0.58),
    ("cat_05.jpg", True, 0.72),
    ("dog_05.jpg", False, 0.29),
    ("cat_10.jpg", True, 0.18),
    ("dog_06.jpg", False, 0.24),
    ("cat_06.jpg", True, 0.68),
    ("dog_09.jpg", False, 0.20),
    ("dog_11.jpg", False, 0.51),
    ("cat_08.jpg", True, 0.63),
    ("dog_10.jpg", False, 0.12),
    ("cat_11.jpg", True, 0.57),
    ("dog_12.jpg", False, 0.05),
    ("cat_12.jpg", True, 0.53),
]


def _outcome(is_cat, predicted_cat):
    if is_cat and predicted_cat:
        return "TP"
    if (not is_cat) and predicted_cat:
        return "FP"
    if is_cat and (not predicted_cat):
        return "FN"
    return "TN"


def _is_highlighted(outcome, show):
    if show == "All":
        return True
    if show == "Precision":
        return outcome in {"TP", "FP"}
    if show == "Recall":
        return outcome in {"TP", "FN"}
    return outcome == show


@lru_cache(maxsize=None)
def _photo(filename):
    """Load one photo once and crop it to a consistent 4:3 teaching card."""
    path = _PHOTO_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing animal photograph: {path}. Keep animal_photos beside utils.py."
        )
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (400, 300),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.45),
        )
        return np.asarray(image)


def _faded(photo):
    """Return a light grayscale copy so filtered-out cards recede clearly."""
    luminance = np.dot(photo[..., :3], [0.2126, 0.7152, 0.0722])
    gray = np.repeat(luminance[..., None], 3, axis=2)
    return (0.22 * gray + 0.78 * 255).clip(0, 255).astype(np.uint8)


def _framed(photo, colour, width):
    """Bake a perfectly even, inset pixel border into a photo card."""
    framed = photo.copy()
    rgb = np.asarray(ImageColor.getrgb(colour), dtype=np.uint8)
    framed[:width, :, :] = rgb
    framed[-width:, :, :] = rgb
    framed[:, :width, :] = rgb
    framed[:, -width:, :] = rgb
    return framed


def plot_cat_detector(show="All", threshold=0.50):
    """Plot a filterable photo grid and its matching confusion matrix.

    Parameters
    ----------
    show : {"All", "TP", "FP", "FN", "TN", "Precision", "Recall"}
        A single outcome highlights just that cell. ``Precision`` highlights the
        predicted-positive denominator (TP + FP); ``Recall`` highlights the
        actual-positive denominator (TP + FN).
    threshold : float
        Predict cat when the teaching score is greater than or equal to this value.
    """
    valid = {"All", "TP", "FP", "FN", "TN", "Precision", "Recall"}
    if show not in valid:
        raise ValueError(f"show must be one of {sorted(valid)}")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    records = []
    counts = {key: 0 for key in ("TP", "FP", "FN", "TN")}
    for filename, is_cat, score in _ANIMALS:
        predicted_cat = score >= threshold
        outcome = _outcome(is_cat, predicted_cat)
        counts[outcome] += 1
        records.append((filename, is_cat, predicted_cat, score, outcome))

    # Construct an unmanaged Agg figure.  Unlike ``pyplot.figure``, this figure
    # is never registered with Jupyter's inline backend, so the frontend cannot
    # auto-flush extra copies of it.
    fig = Figure(figsize=(18, 9.5), facecolor="white")
    FigureCanvasAgg(fig)
    outer = fig.add_gridspec(1, 2, width_ratios=[3.7, 1.35], wspace=0.09)
    photo_grid = outer[0].subgridspec(4, 6, wspace=0.07, hspace=0.43)

    for idx, (filename, is_cat, predicted_cat, score, outcome) in enumerate(records):
        ax = fig.add_subplot(photo_grid[idx // 6, idx % 6])
        highlighted = _is_highlighted(outcome, show)
        edge, _, _, _ = _OUTCOMES[outcome]
        image = _photo(filename)
        card = image if highlighted else _faded(image)
        border_colour = edge if highlighted else "#D6D6D6"
        border_width = 7 if highlighted else 2
        ax.imshow(_framed(card, border_colour, border_width))

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        text_colour = "white" if highlighted else "#AAAAAA"
        pill_colour = edge if highlighted else "#E6E6E6"
        ax.text(
            0.04,
            0.94,
            outcome,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=text_colour,
            bbox={"boxstyle": "round,pad=0.28", "fc": pill_colour, "ec": "none", "alpha": 0.96},
        )

        # Keep the predicted class visible even when this card is faded by an
        # outcome filter.  The symbol encodes prediction—not correctness—so it
        # uses one neutral visual system rather than the TP/FP/FN/TN colours.
        prediction_symbol = "+" if predicted_cat else "−"
        prediction_fill = "#243B53" if predicted_cat else "white"
        prediction_text = "white" if predicted_cat else "#243B53"
        ax.text(
            0.93,
            0.92,
            prediction_symbol,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=prediction_text,
            zorder=10,
            bbox={
                "boxstyle": "circle,pad=0.26",
                "fc": prediction_fill,
                "ec": "#243B53",
                "lw": 1.3,
                "alpha": 0.98,
            },
        )

        actual = "cat" if is_cat else "not cat"
        predicted = "cat" if predicted_cat else "not cat"
        ax.text(
            0.5,
            -0.08,
            f"score {score:.2f} · actual {actual}\npredicted {predicted}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            color="#333333" if highlighted else "#BBBBBB",
        )

    view_titles = {
        "All": "All four outcomes",
        "TP": "True positives: correct cat detections",
        "FP": "False positives: false cat alarms",
        "FN": "False negatives: cats the detector missed",
        "TN": "True negatives: correct not-cat rejections",
        "Precision": "Precision denominator: everything predicted cat (TP + FP)",
        "Recall": "Recall denominator: every actual cat (TP + FN)",
    }
    fig.suptitle(
        f'Real-photo “cat detector” at T = {threshold:.2f} — {view_titles[show]}',
        x=0.39,
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.39,
        0.945,
        "TOP-RIGHT PREDICTION BADGE     +  predicted cat (positive)"
        "      −  predicted not-cat (negative)",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#243B53",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.42",
            "fc": "#F3F6F8",
            "ec": "#CBD5E1",
            "lw": 1.0,
        },
    )

    # Confusion matrix in sklearn's ordering: rows are actual [not cat, cat],
    # columns are predicted [not cat, cat]. TP therefore sits at bottom-right.
    axm = fig.add_subplot(outer[1])
    axm.set_xlim(-0.55, 2.08)
    axm.set_ylim(-2.05, 3.0)
    axm.set_aspect("equal")
    axm.axis("off")
    axm.text(0.75, 2.82, f"The same 24 decisions at T = {threshold:.2f}", ha="center",
             fontsize=13.5, fontweight="bold")
    axm.text(0.75, 2.58, "rows = actual, columns = predicted", ha="center", fontsize=9.5, color="#555555")
    axm.text(0.5, 2.18, "not cat", ha="center", fontsize=10)
    axm.text(1.5, 2.18, "cat", ha="center", fontsize=10)
    axm.text(0.98, 2.43, "PREDICTED", ha="center", fontsize=9, color="#666666", fontweight="bold")
    axm.text(-0.40, 0.98, "ACTUAL", ha="center", va="center", rotation=90,
             fontsize=9, color="#666666", fontweight="bold")
    axm.text(-0.10, 1.5, "not cat", ha="right", va="center", fontsize=10)
    axm.text(-0.10, 0.5, "cat", ha="right", va="center", fontsize=10)

    matrix_cells = [
        ("TN", counts["TN"], 0, 1),
        ("FP", counts["FP"], 1, 1),
        ("FN", counts["FN"], 0, 0),
        ("TP", counts["TP"], 1, 0),
    ]
    for outcome, count, x, y in matrix_cells:
        highlighted = _is_highlighted(outcome, show)
        edge, fill, _, _ = _OUTCOMES[outcome]
        axm.add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                facecolor=fill if highlighted else "#F2F2F2",
                edgecolor="white",
                linewidth=1.5,
            )
        )
        # Draw each outline just inside its cell. Adjacent strokes therefore do
        # not overlap or become visually thicker along shared edges.
        inset = 0.018
        axm.add_patch(
            Rectangle(
                (x + inset, y + inset),
                1 - 2 * inset,
                1 - 2 * inset,
                facecolor="none",
                edgecolor=edge if highlighted else "#D6D6D6",
                linewidth=3.2 if highlighted else 1.2,
                joinstyle="miter",
            )
        )
        axm.text(x + 0.5, y + 0.64, str(count), ha="center", va="center",
                 fontsize=22, fontweight="bold", color=edge if highlighted else "#BBBBBB")
        axm.text(x + 0.5, y + 0.28, outcome, ha="center", va="center",
                 fontsize=10, fontweight="bold", color=edge if highlighted else "#BBBBBB")

    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn)

    axm.text(0.0, -0.34, f"Accuracy  = (TP + TN) / all = {tp + tn}/{n} = {accuracy:.2f}",
             ha="left", fontsize=9.5)
    precision_text = f"{precision:.2f}" if np.isfinite(precision) else "undefined"
    axm.text(0.0, -0.63, f"Precision = TP / (TP + FP) = {tp}/{tp + fp} = {precision_text}",
             ha="left", fontsize=9.5, fontweight="bold" if show == "Precision" else "normal")
    axm.text(0.0, -0.92, f"Recall    = TP / (TP + FN) = {tp}/{tp + fn} = {recall:.2f}",
             ha="left", fontsize=9.5, fontweight="bold" if show == "Recall" else "normal")

    if show in _OUTCOMES:
        _, _, reading, consequence = _OUTCOMES[show]
        axm.text(0.0, -1.34, f"{show}: {reading}", ha="left", fontsize=10,
                 color=_OUTCOMES[show][0], fontweight="bold")
        axm.text(0.0, -1.58, consequence, ha="left", fontsize=9.5, color="#444444")
    elif show == "Precision":
        axm.text(0.0, -1.34, "Precision asks: of everything flagged,", ha="left", fontsize=9.5)
        axm.text(0.0, -1.58, "how much was actually a cat?", ha="left", fontsize=9.5, fontweight="bold")
    elif show == "Recall":
        axm.text(0.0, -1.34, "Recall asks: of every cat available,", ha="left", fontsize=9.5)
        axm.text(0.0, -1.58, "how many did the detector find?", ha="left", fontsize=9.5, fontweight="bold")
    else:
        axm.text(0.0, -1.34, "True/false: was the prediction correct?", ha="left", fontsize=9.5)
        axm.text(0.0, -1.58, "Positive/negative: did it predict cat?", ha="left", fontsize=9.5)

    # Rendering is controlled by the notebook's single Image widget.
    return fig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

BLUE, RED, GOLD, GREEN, GRAY, PURPLE = '#30A2DA', '#FC4F30', '#E5AE38', '#6D904F', '#8B8B8B', '#810F7C'

def configure_plots():
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({
        'axes.prop_cycle': cycler(color=[BLUE, RED, GOLD, GREEN, GRAY, PURPLE]),
        'axes.titlesize': 16, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'figure.figsize': (9, 5), 'figure.dpi': 110,
        'lines.linewidth': 1.25, 'lines.markersize': 7,
        'grid.linewidth': 0.6, 'grid.alpha': 0.45,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.bbox': 'tight',
    })

def finish(ax, title, xlabel, ylabel):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.figure.tight_layout()
    plt.show()

def metric_bars(table, columns, title):
    ax=table[columns].plot.bar(figsize=(10,5), width=0.75)
    ax.set_ylim(0,1.05)
    ax.tick_params(axis='x', rotation=15)
    finish(ax,title,'','Score')

from matplotlib.colors import ListedColormap
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

def _matrix_cell_borders(ax, size):
    """Keep equally coloured cells distinct without adding grid lines through labels."""
    ax.grid(False)
    for row in range(size):
        for column in range(size):
            ax.add_patch(Rectangle(
                (column - .5, row - .5), 1, 1, fill=False,
                edgecolor='#D8E1EA', linewidth=1.0, zorder=3, clip_on=False))


def _metric_equations(ax, counts, compact=False):
    """Render the same counts in symbolic and substituted metric equations."""
    tn, fp, fn, tp = (int(value) for value in counts.ravel())
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    ax.set_axis_off()
    if not compact:
        ax.text(0, .96, f'TP = {tp}     FN = {fn}     FP = {fp}     TN = {tn}     Total = {total}',
                transform=ax.transAxes, va='center', fontsize=11, fontweight='bold')
    equations = [
        (r'\mathrm{Accuracy}', r'\dfrac{TP+TN}{TP+TN+FP+FN}',
         rf'\dfrac{{{tp}+{tn}}}{{{total}}}', accuracy, 'no observations'),
        (r'\mathrm{Precision}', r'\dfrac{TP}{TP+FP}',
         rf'\dfrac{{{tp}}}{{{tp}+{fp}}}', precision, 'no positive predictions'),
        (r'\mathrm{Recall}', r'\dfrac{TP}{TP+FN}',
         rf'\dfrac{{{tp}}}{{{tp}+{fn}}}', recall, 'no actual positives'),
    ]
    positions = (.90, .59, .28) if compact else (.69, .37, .05)
    for y, (name, formula, substitution, value, reason) in zip(positions, equations):
        result = f'{value:.2f}' if np.isfinite(value) else r'\mathrm{undefined}'
        if compact:
            ax.text(0, y, f'${name} = {formula}$', transform=ax.transAxes,
                    va='center', fontsize=11.5)
            ax.text(.04, y - .13, f'$= {substitution} = {result}$',
                    transform=ax.transAxes, va='center', fontsize=12)
            if not np.isfinite(value):
                ax.text(.04, y - .22, reason.capitalize(), transform=ax.transAxes,
                        va='center', fontsize=9.5)
            continue
        equation = f'${name} = {formula} = {substitution} = {result}$'
        if not np.isfinite(value):
            equation += f'  ({reason})'
        ax.text(0, y, equation, transform=ax.transAxes, va='center', fontsize=13)
    if compact:
        ax.text(0, .005, f'Total: TP + TN + FP + FN = {total}',
                transform=ax.transAxes, va='bottom', fontsize=9.5)


def probability_threshold(X,y,grid,p,thresholds=(.3,.5,.7)):
    fig,ax=plt.subplots()
    ax.scatter(X,y,alpha=.20,s=20,edgecolors='none',label='Validation outcome')
    ax.plot(grid,p,color=BLUE,label='Fitted probability')
    for T,color in zip(thresholds,[GOLD,GRAY,RED]): ax.axhline(T,color=color,ls='--',lw=1,label=f'T = {T}')
    ax.set_ylim(-.05,1.05);ax.legend()
    finish(ax,'One model, several decision thresholds','Field-goal proportion difference','Probability of a win')

def matrix(y, pred, title, labels=('Not cat (0)', 'Cat (1)')):
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    counts = confusion_matrix(y, pred, labels=np.arange(len(labels)))
    shown = ConfusionMatrixDisplay(counts, display_labels=labels).plot(
        ax=ax, cmap='Blues', colorbar=False,
        im_kw={'vmin': 0, 'vmax': max(1, counts.max())})
    colorbar = fig.colorbar(shown.im_, ax=ax, fraction=0.046, pad=0.06)
    colorbar.set_label('Number of observations')
    colorbar.locator = plt.MaxNLocator(integer=True)
    colorbar.update_ticks()
    ax.set_title(title)
    _matrix_cell_borders(ax, len(labels))
    fig.text(0.5, 0.01, 'Darker blue = more observations, not better performance.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()

def threshold_metrics(table, probabilities=None):
    fig,ax=plt.subplots()
    for col in ['Accuracy','Precision','Recall']:
        values = table[col].copy()
        if col == 'Precision' and probabilities is not None:
            # Keep undefined precision out of the curve. The teaching table
            # still shows its explicit zero_division=0 reporting convention.
            values = values.mask(table.Threshold > np.max(probabilities))
        ax.plot(table.Threshold, values, label=col)
    ax.set_ylim(0,1.04);ax.legend()
    if probabilities is not None:
        ax.text(.02, .03, 'Precision is omitted where no positives are predicted.',
                transform=ax.transAxes, fontsize=10)
    finish(ax,'Validation metrics change with the threshold','Decision threshold','Score')

def pr_roc(precision,recall,fpr,tpr,ap,auc,prevalence,chosen_point=None):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].step(recall,precision,where='post',label=f'Average precision = {ap:.3f}')
    axes[0].axhline(prevalence,color=GRAY,ls='--',lw=1,label=f'Prevalence = {prevalence:.2f}')
    axes[0].set(title='Precision–recall',xlabel='Recall',ylabel='Precision')
    axes[1].plot(fpr,tpr,label=f'ROC AUC = {auc:.3f}')
    axes[1].plot([0,1],[0,1],color=GRAY,ls='--',lw=1,label='Random ranking')
    axes[1].set(title='Receiver operating characteristic',xlabel='False-positive rate',ylabel='True-positive rate (recall)')
    if chosen_point is not None:
        label = f"Chosen T = {chosen_point['Threshold']:.2f}"
        axes[0].scatter(chosen_point['Recall'], chosen_point['Precision'],
                        color=RED, marker='D', s=60, edgecolors='white',
                        linewidths=0.6, zorder=4, label=label)
        axes[1].scatter(chosen_point['False-positive rate'], chosen_point['Recall'],
                        color=RED, marker='D', s=60, edgecolors='white',
                        linewidths=0.6, zorder=4, label=label)
    for ax in axes: ax.set_xlim(0,1.02);ax.set_ylim(0,1.02);ax.legend(loc='lower left')
    fig.tight_layout();plt.show()

def boundaries(model,X,y,thresholds=(.3,.5,.7)):
    names=list(X.columns)
    gx=np.linspace(X.iloc[:,0].min(),X.iloc[:,0].max(),200)
    gy=np.linspace(X.iloc[:,1].min(),X.iloc[:,1].max(),200)
    xx,yy=np.meshgrid(gx,gy)
    grid=pd.DataFrame({names[0]:xx.ravel(),names[1]:yy.ravel()})
    probabilities=model.predict_proba(grid)[:,1].reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(9,6))
    ax.contourf(xx,yy,probabilities>=.5,levels=[-.5,.5,1.5],colors=[BLUE,RED],alpha=.13)
    for label,color,name in [(0,BLUE,'Loss'),(1,RED,'Win')]:
        m=np.asarray(y)==label
        ax.scatter(X.iloc[m,0],X.iloc[m,1],color=color,alpha=.4,s=22,edgecolors='none',label=name)
    cs=ax.contour(xx,yy,probabilities,levels=thresholds,colors=[GRAY,GOLD,PURPLE],linewidths=1.25)
    ax.clabel(cs,fmt=lambda v:f'T = {v:.1f}',fontsize=10)
    ax.legend();finish(ax,'Changing the threshold moves the boundary','Field-goal proportion difference','Personal-foul difference')

def iris_regions(model,X,y,names):
    a=np.linspace(X.iloc[:,0].min()-.3,X.iloc[:,0].max()+.3,200)
    b=np.linspace(X.iloc[:,1].min()-.3,X.iloc[:,1].max()+.3,200)
    xx,yy=np.meshgrid(a,b)
    grid=pd.DataFrame({X.columns[0]:xx.ravel(),X.columns[1]:yy.ravel()})
    region=model.predict(grid).reshape(xx.shape)
    colors=[BLUE,RED,GOLD]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.contourf(xx,yy,region,levels=[-.5,.5,1.5,2.5],colors=colors,alpha=.15)
    for k,name in enumerate(names):
        m=np.asarray(y)==k;ax.scatter(X.iloc[m,0],X.iloc[m,1],color=colors[k],s=35,alpha=.6,edgecolors='none',label=name)
    ax.legend();finish(ax,'Three species, three prediction regions','Petal length (cm)','Petal width (cm)')

def _display_widget_with_fallback(controls, picture):
    """Offer one live widget or its initial PNG, depending on viewer support."""
    import base64
    from IPython.display import display

    bundle = controls._repr_mimebundle_()
    bundle['image/png'] = base64.b64encode(picture.value).decode('ascii')
    display(bundle, raw=True)


def cat_detector_widget(start=0.5, show='All'):
    """Change the threshold and highlight outcomes in the same 24-photo grid."""
    from io import BytesIO
    import ipywidgets as widgets

    options = ['All', 'TP', 'FP', 'TN', 'FN']
    if not np.isfinite(start) or not 0 <= start <= 1:
        raise ValueError('start must be between 0 and 1')
    if show not in options:
        raise ValueError(f'show must be one of {options}')
    threshold = widgets.FloatSlider(
        value=start, min=0, max=1, step=.01, description='Threshold:',
        readout_format='.2f', continuous_update=False,
        layout=widgets.Layout(width='380px', max_width='100%', min_width='0'))
    highlight = widgets.ToggleButtons(
        options=options, value=show, description='Highlight:',
        tooltips=['Show all outcomes', 'True positives', 'False positives',
                  'True negatives', 'False negatives'],
        style={'button_width': '55px'},
        layout=widgets.Layout(max_width='100%', min_width='0'))
    picture = widgets.Image(format='png', width='100%', layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', height='auto',
        flex='0 1 auto', align_self='stretch', object_fit='contain', display='block'))

    def render(change=None):
        figure = plot_cat_detector(show=highlight.value, threshold=threshold.value)
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format='png')
            picture.value = buffer.getvalue()
        finally:
            plt.close(figure)

    threshold.observe(render, names='value')
    highlight.observe(render, names='value')
    render()
    controls = widgets.HBox([threshold, highlight], layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0',
        flex_flow='row wrap', align_items='center'))
    widget = widgets.VBox([controls, picture], layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', height='auto',
        align_items='stretch', overflow='hidden', margin='0', padding='0'))
    _display_widget_with_fallback(widget, picture)
    return widget


def threshold_widget(scores,labels):
    import ipywidgets as widgets
    from io import BytesIO
    slider=widgets.FloatSlider(value=.5,min=0,max=1,step=.05,description='Threshold:',continuous_update=False)
    output=widgets.Image(format='png')
    def render(change=None):
        pred=(np.asarray(scores)>=slider.value).astype(int)
        counts = confusion_matrix(labels, pred, labels=[0, 1])
        fig = Figure(figsize=(10, 6.8), dpi=110)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([.08, .48, .88, .43])
        for k,color,name in [(0,BLUE,'Actual negative'),(1,RED,'Actual positive')]:
            m=np.asarray(labels)==k
            ax.scatter(np.asarray(scores)[m],np.flatnonzero(m),color=color,alpha=.65,s=65,label=name)
        ax.axvline(slider.value,color=GRAY,ls='--',lw=1.2)
        ax.set(xlim=(-.05,1.05),xlabel='Predicted probability',ylabel='Observation',
               title=f'Predict positive when probability is at least {slider.value:.2f}')
        ax.legend()
        _metric_equations(fig.add_axes([.08, .035, .88, .30]), counts)
        buffer=BytesIO();fig.savefig(buffer,format='png');output.value=buffer.getvalue();plt.close(fig)
    slider.observe(render,names='value');render()
    _display_widget_with_fallback(widgets.VBox([slider,output]), output)


def animal_threshold_examples(original=None):
    """Extend the six-image lesson with six already-attributed local photos.

    The original rows and their teaching probabilities are preserved. Added
    photos use the same probabilities as the later 24-photo display.
    """
    if original is None:
        original = pd.DataFrame({
            'Image': list('ABCDEF'),
            'Animal': ['Domestic cat', 'Tiger', 'Raccoon', 'Husky', 'Hamster', 'Hidden cat'],
            'Actual': [1, 0, 0, 0, 0, 1],
            'Probability': [.9, .8, .7, .2, .1, .4],
        })
    examples = original[['Image', 'Animal', 'Actual', 'Probability']].copy()
    if list(examples.Image) != list('ABCDEF'):
        raise ValueError('Pass the original six animal rows A through F.')
    examples['Photo'] = 'classification_animals.png'
    examples['Tile'] = np.arange(6)
    photo_scores = {name: (int(is_cat), score) for name, is_cat, score in _ANIMALS}
    added = []
    for image_id, filename in zip('GHIJKL', [
            'cat_01.jpg', 'cat_03.jpg', 'cat_07.jpg',
            'dog_02.jpg', 'dog_01.jpg', 'dog_12.jpg']):
        actual, probability = photo_scores[filename]
        added.append({'Image': image_id, 'Animal': 'Domestic cat' if actual else 'Dog',
                      'Actual': actual, 'Probability': probability,
                      'Photo': filename, 'Tile': -1})
    examples = pd.concat([examples, pd.DataFrame(added)], ignore_index=True)
    # Invent ratings compatible with the existing teaching probabilities.
    # They are illustrative inputs, not features measured from photographs.
    probability = examples['Probability']
    examples['Cat-feature rating (x)'] = 5 + np.log(probability / (1 - probability))
    return examples


@lru_cache(maxsize=None)
def _animal_thumbnail(filename, tile=-1):
    """Load a small display thumbnail; leave source photographs untouched."""
    path = (Path(__file__).resolve().parent / filename if tile >= 0
            else _PHOTO_DIR / filename)
    with Image.open(path) as source:
        photo = ImageOps.exif_transpose(source).convert('RGB')
        if tile >= 0:
            # The original illustration is a two-row, three-column image.
            width, height = photo.size
            col, row = tile % 3, tile // 3
            photo = photo.crop((col * width // 3, row * height // 2,
                                (col + 1) * width // 3, (row + 1) * height // 2))
        return np.asarray(ImageOps.fit(photo, (96, 96), Image.Resampling.LANCZOS))


def threshold_prediction_figure(threshold=0.5, animals=None):
    """Follow the same animals from assigned probabilities to decisions.

    x is an invented 0–10 cat-feature rating, chosen to preserve the existing
    teaching probabilities under p(x) = 1 / (1 + exp(-(x - 5))). It is not measured
    from the photos. Markers and thumbnail leaders end at (x, actual label);
    the sigmoid alone shows probability, keeping observations and predictions
    distinct.
    Equality predicts cat. Endpoints never require log(0).
    """
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    animals = animal_threshold_examples() if animals is None else animals.copy()
    probability = np.asarray(animals.Probability, dtype=float)
    actual = np.asarray(animals.Actual, dtype=int)
    if len(animals) != 12 or not np.isfinite(probability).all() or np.any((probability <= 0) | (probability >= 1)):
        raise ValueError('Use the twelve animal examples, with probabilities strictly between 0 and 1.')
    if not np.isin(actual, [0, 1]).all():
        raise ValueError('Actual labels must be 0 (not cat) or 1 (cat).')
    x = np.asarray(animals['Cat-feature rating (x)'], dtype=float)
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 10)):
        raise ValueError('Cat-feature ratings must be between 0 and 10.')
    if not np.allclose(1 / (1 + np.exp(-(x - 5))), probability, atol=1e-12, rtol=0):
        raise ValueError('Use the fixed teaching model p(x) = 1 / (1 + exp(-(x - 5))) and its matching probabilities.')
    predicted = (probability >= threshold).astype(int)
    counts = confusion_matrix(actual, predicted, labels=[0, 1])
    fig = Figure(figsize=(14.5, 5.2), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([.055, .33, .405, .44])
    axm = fig.add_axes([.535, .33, .16, .49])
    grid = np.linspace(0, 10, 600)
    ax.plot(grid, 1 / (1 + np.exp(-(grid - 5))), color=BLUE, lw=1.4,
            label='Sigmoid')
    ax.axhline(threshold, color=GRAY, lw=1.0, linestyle='--',
               label=f'T = {threshold:.2f}')
    if 0 < threshold < 1:
        boundary = 5 + np.log(threshold / (1 - threshold))
        ax.axvline(boundary, color=GRAY, lw=1.0, linestyle=':')
        decision_text = rf'Predict cat when $x \geq {boundary:.2f}$'
    else:
        decision_text = 'All 12 predicted cat' if threshold == 0 else 'All 12 predicted not cat'
    shapes = {'TP': 'o', 'FP': '^', 'FN': 's', 'TN': 'D'}
    # Keep actual cats above the plot and non-cats below it. Every leader ends
    # on a binary observation, not on the probability curve.
    thumbnail_positions = {}
    for actual_class, box_y in [(1, 1.25), (0, -.47)]:
        indices = [i for i in np.argsort(probability, kind='stable') if actual[i] == actual_class]
        for box_x, index in zip(np.linspace(.045, .955, len(indices)), indices):
            thumbnail_positions[index] = (box_x, box_y)
    for index in np.argsort(probability, kind='stable'):
        row = animals.iloc[index]
        outcome = _outcome(bool(actual[index]), bool(predicted[index]))
        color = _OUTCOMES[outcome][0]
        ax.scatter(x[index], actual[index], s=50, color=color,
                   marker=shapes[outcome], edgecolors='white', linewidths=.5,
                   zorder=5)
        box_x, box_y = thumbnail_positions[index]
        thumbnail = OffsetImage(_animal_thumbnail(row.Photo, int(row.Tile)), zoom=.36)
        card = AnnotationBbox(thumbnail, (x[index], actual[index]),
                              xybox=(box_x, box_y), xycoords='data',
                              boxcoords=ax.transAxes, pad=.12,
                              bboxprops={'edgecolor': color, 'linewidth': 1.6, 'facecolor': 'white'},
                              arrowprops={'arrowstyle': '-', 'color': '#A5A5A5', 'linewidth': .65},
                              annotation_clip=False, zorder=2)
        ax.add_artist(card)
        ax.annotate(f'{row.Image} · {outcome}', xy=(box_x, box_y),
                    xycoords='axes fraction', xytext=(0, -23), textcoords='offset points',
                    ha='center', va='top', fontsize=8.5, color=color, fontweight='bold',
                    annotation_clip=False)
    ax.set(xlim=(0, 10), ylim=(-.08, 1.08),
           xlabel='Visual cat-feature rating, x', ylabel='Actual label / predicted probability')
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.tick_params(labelsize=10)
    ax.text(.02, .91, r'$p(x)=\frac{1}{1+e^{-(x-5)}}$' + f'\nT = {threshold:.2f}; ' + decision_text,
            transform=ax.transAxes, fontsize=10, linespacing=1.8, va='top', zorder=6,
            bbox={'facecolor': '#F0F0F0', 'edgecolor': 'none', 'alpha': 1})
    ax.xaxis.label.set_bbox({'facecolor': '#F0F0F0', 'edgecolor': 'none', 'alpha': .95, 'pad': 1})

    color_max = int(max(np.bincount(actual, minlength=2)))
    shown = ConfusionMatrixDisplay(counts, display_labels=['Not cat (0)', 'Cat (1)'])
    shown.plot(ax=axm, cmap='Blues', colorbar=False,
               im_kw={'vmin': 0, 'vmax': color_max})
    for row, column, label in [(0, 0, 'TN'), (0, 1, 'FP'), (1, 0, 'FN'), (1, 1, 'TP')]:
        shown.text_[row, column].set_text(f'{label} = {counts[row, column]}')
        shown.text_[row, column].set_fontsize(11)
    axm.set_title('The same 12 animals', fontsize=12)
    axm.tick_params(labelsize=9.5)
    axm.set_xlabel('Predicted label', fontsize=11)
    axm.set_ylabel('True label', fontsize=11)
    _matrix_cell_borders(axm, 2)
    colorbar = fig.colorbar(shown.im_, cax=fig.add_axes([.706, .33, .010, .49]), ticks=range(color_max + 1))
    colorbar.set_label('Observations', fontsize=9.5)
    colorbar.ax.tick_params(labelsize=9)
    axm.text(.5, -.34, f'Darker blue = more animals.\nFixed scale: 0 to {color_max}.',
             transform=axm.transAxes, ha='center', fontsize=9.5)
    metrics = fig.add_axes([.775, .14, .22, .70])
    metrics.set_title(f'Metrics at T = {threshold:.2f}', fontsize=12, loc='left')
    _metric_equations(metrics, counts, compact=True)
    return fig


def threshold_prediction_plot(start=0.5, animals=None):
    """Display an interactive threshold slider with a saved initial PNG."""
    import ipywidgets as widgets
    from io import BytesIO

    if not np.isfinite(start) or not 0 <= start <= 1:
        raise ValueError('start must be between 0 and 1')
    slider = widgets.FloatSlider(value=start, min=0, max=1, step=.01,
                                 description='Threshold:', readout_format='.2f',
                                 continuous_update=False,
                                 layout=widgets.Layout(max_width='100%', min_width='0'))
    # Allow the image to shrink below its intrinsic PNG width in notebook
    # flex layouts, while also filling a wider output cell without a pixel cap.
    output = widgets.Image(format='png', width='100%', layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', height='auto',
        flex='0 1 auto', align_self='stretch', object_fit='contain', display='block'))

    def render(change=None):
        figure = threshold_prediction_figure(slider.value, animals=animals)
        buffer = BytesIO()
        figure.savefig(buffer, format='png')
        output.value = buffer.getvalue()
        plt.close(figure)

    slider.observe(render, names='value')
    render()
    controls = widgets.VBox([slider, output], layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', height='auto',
        align_items='stretch', overflow='hidden', margin='0', padding='0'))
    _display_widget_with_fallback(controls, output)


def threshold_prediction(threshold):
    """Compatibility entry point for the original percentage-based callback."""
    from IPython.display import display
    figure = threshold_prediction_figure(float(threshold) / 100)
    display(figure)
    plt.close(figure)


def prevalence_curves(curves, threshold=0.5, figsize=(13, 5.8)):
    """Plot already-calculated ROC and PR curves for controlled class mixes.

    Each dictionary contains prevalence, fpr, tpr, precision, recall, and a
    point dictionary with the same three rates at the indicated threshold.
    Data generation and metric calculations stay visible in the notebook.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = [BLUE, GOLD, RED]
    styles = ['-', '--', ':']
    for i, (curve, color, style) in enumerate(zip(curves, colors, styles)):
        label = f"{curve['prevalence']:.0%} positive"
        # Different styles reveal coincident curves without moving the data.
        axes[0].plot(curve['fpr'], curve['tpr'], color=color,
                     linestyle=style, linewidth=[2.5, 1.8, 1.1][i], label=label)
        axes[1].step(curve['recall'], curve['precision'], where='post',
                     color=color, linestyle=style, linewidth=1.5, label=label)
        axes[1].axhline(curve['prevalence'], color=color, linestyle='--',
                        linewidth=.8, alpha=.65)
        point = curve['point']
        # Nested outlines keep the three identical ROC operating points visible.
        axes[0].scatter(point['fpr'], point['recall'], s=[125, 65, 20][i],
                        facecolors='none', edgecolors=color, linewidths=1.2,
                        zorder=5)
        axes[1].scatter(point['recall'], point['precision'], s=48,
                        color=color, alpha=.8, edgecolors='white',
                        linewidths=.6, zorder=5)
    axes[0].plot([0, 1], [0, 1], color=GRAY, linestyle='--', linewidth=.8)
    axes[0].set(title='ROC: all three curves coincide',
                xlabel='False-positive rate', ylabel='Recall (true-positive rate)')
    axes[1].set(title='PR: precision falls as positives become rare',
                xlabel='Recall', ylabel='Precision')
    axes[0].text(.43, .25, 'The score distributions within\neach class stay fixed.',
                 transform=axes[0].transAxes, fontsize=11)
    axes[1].text(.04, .26, 'Dashed horizontal lines:\nprevalence baselines',
                 transform=axes[1].transAxes, fontsize=10)
    for ax in axes:
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(-.02, 1.03)
        ax.legend(loc='lower left' if ax is axes[0] else 'upper right',
                  framealpha=.9)
    fig.text(.5, .02,
             f'Markers show the same threshold T = {threshold:.2f}. '
             'Only the proportion of positive and negative observations changes.',
             ha='center', fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, 1))
    return fig


def _prevalence_state(reference, positive_share):
    """Reweight the two fixed classes; preserve every within-class rate."""
    if not np.isfinite(positive_share) or not 0 < positive_share < 1:
        raise ValueError('positive_share must be strictly between 0 and 1')
    base_share = float(reference['prevalence'])
    if not 0 < base_share < 1:
        raise ValueError('The reference must contain both classes')
    # At each score cutoff, scale the positive and negative contributions to
    # the alert list separately. Its ranking and class-conditional rates stay fixed.
    base_precision = np.asarray(reference['precision'], dtype=float)
    positive_weight = positive_share / base_share
    negative_weight = (1 - positive_share) / (1 - base_share)
    precision = positive_weight * base_precision / (
        positive_weight * base_precision + negative_weight * (1 - base_precision))
    recall = float(reference['point']['recall'])
    fpr = float(reference['point']['fpr'])
    positives = 100_000 * positive_share
    negatives = 100_000 - positives
    tp, fn = positives * recall, positives * (1 - recall)
    fp, tn = negatives * fpr, negatives * (1 - fpr)
    point_precision = tp / (tp + fp) if tp + fp else np.nan
    return dict(share=positive_share, precision=precision, recall=recall, fpr=fpr,
                point_precision=point_precision, positives=positives, negatives=negatives,
                tp=tp, fn=fn, fp=fp, tn=tn)


def prevalence_figure(reference, positive_share=.5, roc_auc=None, threshold=.5):
    """Compare a changed class mix with its fixed 50% reference."""
    state = _prevalence_state(reference, positive_share)
    base = _prevalence_state(reference, reference['prevalence'])
    fig = Figure(figsize=(14.6, 6.2), facecolor='#F0F0F0')
    FigureCanvasAgg(fig)
    roc = fig.add_axes([.055, .38, .255, .51])
    pr = fig.add_axes([.395, .38, .255, .51])
    alerts = fig.add_axes([.735, .38, .25, .51])
    reference_label = f"{reference['prevalence']:.0%} positive reference"
    current_label = f'{positive_share:.0%} positive now'

    roc.plot(reference['fpr'], reference['tpr'], color=GRAY, ls='--', lw=2.5,
             alpha=.65, label=reference_label)
    roc.plot(reference['fpr'], reference['tpr'], color=BLUE, lw=1.4,
             label=current_label)
    roc.plot([0, 1], [0, 1], color=GRAY, ls=':', lw=.8)
    roc.scatter(state['fpr'], state['recall'], s=60, color=BLUE,
                edgecolors='white', linewidths=.6, zorder=5)
    roc.text(.34, .13,
             f"Recall = {state['recall']:.3f}\nFalse-positive rate = {state['fpr']:.3f}\n"
             + (f'ROC AUC = {roc_auc:.3f}\n' if roc_auc is not None else '')
             + 'All stay fixed.',
             transform=roc.transAxes, fontsize=9, linespacing=1.4)
    roc.set(title='ROC stays unchanged', xlabel='False-positive rate',
            ylabel='Recall (true-positive rate)')

    pr.step(reference['recall'], reference['precision'], where='post',
            color=GRAY, ls='--', lw=1.2, alpha=.7, label=reference_label)
    pr.step(reference['recall'], state['precision'], where='post',
            color=BLUE, lw=1.4, label=current_label)
    pr.axhline(reference['prevalence'], color=GRAY, ls=':', lw=.8, alpha=.7)
    pr.axhline(positive_share, color=BLUE, ls=':', lw=1)
    pr.scatter(state['recall'], state['point_precision'], s=60, color=BLUE,
               edgecolors='white', linewidths=.6, zorder=5)
    pr.text(.04, .08, 'Dotted lines:\npositive-share baselines',
            transform=pr.transAxes, fontsize=9)
    pr.set(title='PR changes with the class mix', xlabel='Recall', ylabel='Precision')
    for ax in [roc, pr]:
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(-.02, 1.03)
        ax.tick_params(labelsize=10)
        ax.set_title(ax.get_title(loc='left'), fontsize=12, loc='left')
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.legend(loc='upper left' if ax is pr else 'lower right',
                  bbox_to_anchor=(0, 1) if ax is pr else (1, .01),
                  fontsize=9, framealpha=.95)
    # Keep the ROC explanation and legend separate.
    roc.get_legend().set_visible(False)
    from matplotlib.ticker import StrMethodFormatter
    for position, values, alpha in [(0, base, .40), (1, state, 1.0)]:
        alerts.bar(position, values['tp'], width=.60, color=GREEN, alpha=alpha,
                   label='True alerts (TP)' if position == 1 else None)
        alerts.bar(position, values['fp'], bottom=values['tp'], width=.60,
                   color=RED, alpha=alpha, hatch='///', edgecolor='#7F2D20', linewidth=.4,
                   label='False alerts (FP)' if position == 1 else None)
        alerts.text(position, values['tp'] + values['fp'] + 1800,
                    f"Precision {values['point_precision']:.1%}",
                    ha='center', fontsize=9.5)
    alerts.set_title('What is in the alert list?', fontsize=12)
    alerts.set_ylabel('Number of alerts', fontsize=11)
    alerts.set_xticks([0, 1], [f"{reference['prevalence']:.0%} positive\nreference",
                              f'{positive_share:.0%} positive\ncurrent'])
    alerts.set_xlim(-.65, 1.65)
    alerts.set_ylim(0, 65_000)
    alerts.set_yticks([0, 20_000, 40_000, 60_000])
    alerts.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    alerts.tick_params(labelsize=9)
    alerts.grid(axis='x', visible=False)
    alerts.legend(loc='upper center', fontsize=8.5, framealpha=.95)
    fig.text(.5, .96,
             f'Fixed scores within each class · Fixed threshold T = {threshold:.2f}',
             ha='center', fontsize=12)

    def count(value):
        return f'{value:,.0f}' if np.isclose(value, round(value)) else f'{value:,.1f}'

    fig.text(.5, .245,
             f"100,000 illustrative cases: {count(state['positives'])} positive + "
             f"{count(state['negatives'])} negative",
             ha='center', fontsize=11)
    for left, label, formula, numerator, first, second, value in [
        (.065, 'Recall', r'\frac{TP}{TP+FN}', state['tp'], state['tp'], state['fn'], state['recall']),
        (.40, 'False-positive rate', r'\frac{FP}{FP+TN}', state['fp'], state['fp'], state['tn'], state['fpr']),
        (.735, 'Precision', r'\frac{TP}{TP+FP}', state['tp'], state['tp'], state['fp'], state['point_precision']),
    ]:
        fig.text(left, .155, label + r' $= ' + formula + '$', fontsize=12)
        fig.text(left, .055,
                 f'= {count(numerator)} / ({count(first)} + {count(second)}) = {value:.3f}',
                 fontsize=11, color=BLUE if label == 'Precision' else '#303030')
    return fig


def prevalence_widget(reference, roc_auc=None, threshold=.5, start=50):
    """Change the positive percentage while holding class-specific scores fixed."""
    from io import BytesIO
    import ipywidgets as widgets

    if not np.isfinite(start) or int(start) != start or not 1 <= start <= 50:
        raise ValueError('start must be an integer percentage from 1 to 50')
    slider = widgets.IntSlider(
        value=int(start), min=1, max=50, step=1, description='Positive cases (%):',
        continuous_update=False, style={'description_width': '145px'},
        layout=widgets.Layout(width='560px', max_width='100%', min_width='0'))
    picture = widgets.Image(format='png', width='100%', layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', height='auto',
        flex='0 1 auto', align_self='stretch', object_fit='contain', display='block'))

    def render(change=None):
        figure = prevalence_figure(reference, slider.value / 100, roc_auc, threshold)
        try:
            buffer = BytesIO()
            figure.savefig(buffer, format='png', bbox_inches=None)
            picture.value = buffer.getvalue()
        finally:
            plt.close(figure)

    slider.observe(render, names='value')
    render()
    widget = widgets.VBox([slider, picture], layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', height='auto',
        align_items='stretch', overflow='hidden', margin='0', padding='0'))
    _display_widget_with_fallback(widget, picture)
    return widget
