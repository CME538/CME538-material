"""Plotting helpers for Lecture 8.1. Model fitting stays in the notebook."""
import base64
from io import BytesIO
import os
from pathlib import Path
import sys
import warnings

from cycler import cycler
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import get_ipython
from IPython.display import display
from sklearn.tree import plot_tree

FTE_BLUE = '#30A2DA'
FTE_RED = '#FC4F30'
FTE_YELLOW = '#E5AE38'
FTE_GREEN = '#6D904F'
FTE_GRAY = '#8B8B8B'
FTE_PURPLE = '#810F7C'
FTE_COLORS = [FTE_BLUE, FTE_RED, FTE_YELLOW, FTE_GREEN, FTE_GRAY, FTE_PURPLE]
SPECIES_COLORS = dict(zip(['setosa', 'versicolor', 'virginica'], FTE_COLORS[:3]))
SPECIES_MARKERS = {'setosa': 'o', 'versicolor': 's', 'virginica': '^'}

PLOT_TITLE_SIZE = 16
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 11
LEGEND_SIZE = 10


def configure_notebook():
    """Apply the course display settings and locate the Graphviz executable."""
    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic('matplotlib', 'inline')

    # Locate Graphviz in the active environment without a machine-specific path.
    for graphviz_bin in (Path(sys.prefix) / 'Library' / 'bin',
                         Path(sys.prefix) / 'Library' / 'bin' / 'graphviz',
                         Path(sys.prefix) / 'Library' / 'Graphviz' / 'bin'):
        if graphviz_bin.is_dir():
            current_path = os.environ.get('PATH', '')
            if str(graphviz_bin) not in current_path.split(os.pathsep):
                os.environ['PATH'] = str(graphviz_bin) + os.pathsep + current_path

    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 100)

    # Compatibility warnings from seaborn 0.12 calling newer pandas APIs.
    warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated.*',
                            category=FutureWarning, module=r'seaborn\..*')
    warnings.filterwarnings('ignore', message='When grouping with a length-1 list-like.*',
                            category=FutureWarning, module=r'seaborn\..*')

    plt.style.use('fivethirtyeight')
    sns.set_context('notebook')
    sns.set_palette(FTE_COLORS)
    plt.rcParams.update({
        'axes.prop_cycle': cycler(color=FTE_COLORS),
        'axes.titlesize': PLOT_TITLE_SIZE,
        'axes.titleweight': 'bold',
        'axes.titlelocation': 'left',
        'axes.labelsize': AXIS_LABEL_SIZE,
        'xtick.labelsize': TICK_LABEL_SIZE,
        'ytick.labelsize': TICK_LABEL_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'figure.titlesize': PLOT_TITLE_SIZE,
        'figure.figsize': (9, 5),
        'figure.dpi': 110,
        'lines.linewidth': 1.5,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.45,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


COLORS = ['#30A2DA', '#FC4F30', '#E5AE38']
NAMES = ['setosa', 'versicolor', 'virginica']
MARKERS = ['o', '^', 's']


def single_figure_widget(plot_function, rule_id, **plot_kwargs):
    """Replace one image on each slider change instead of appending plot output."""
    rule_id.continuous_update = False
    rule_id.description = 'Split:'
    rule_id.layout = widgets.Layout(width='100%')
    canvas = widgets.Image(
        format='png',
        layout=widgets.Layout(width='100%', max_width='100%', height='auto'),
    )

    def draw(change=None):
        # Disable automatic inline display while constructing this frame.
        with plt.ioff():
            fig = plot_function(rule_id=rule_id.value, **plot_kwargs)
            try:
                with BytesIO() as buffer:
                    fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
                    canvas.value = buffer.getvalue()
            finally:
                plt.close(fig)

    rule_id.observe(draw, names='value')
    draw()
    view = widgets.VBox([rule_id, canvas], layout=widgets.Layout(width='100%', max_width='100%'))
    display(view)
    return view


def multiclass_animal_widget(animals, photo_dir='animal_photos'):
    """Reveal one-versus-rest outcomes for fixed single-label predictions."""
    from PIL import Image, ImageOps
    from matplotlib.patches import Rectangle
    from sklearn.metrics import confusion_matrix

    data = animals.copy().reset_index(drop=True)
    required = {'Image', 'Actual', 'Predicted', 'Photo'}
    if not required.issubset(data.columns) or len(data) != 12:
        raise ValueError('Use the 12-photo table with Image, Actual, Predicted, and Photo columns.')
    classes = ['Cat', 'Dog', 'Rabbit']
    if not data['Actual'].isin(classes).all() or not data['Predicted'].isin(classes).all():
        raise ValueError('Actual and predicted classes must be Cat, Dog, or Rabbit.')

    photos = []
    for filename in data['Photo']:
        with Image.open(Path(photo_dir) / filename) as picture:
            photos.append(np.asarray(ImageOps.fit(picture.convert('RGB'), (320, 250))))

    outcome_colors = {'TP': '#21783C', 'FP': '#D95F02', 'FN': '#7B3294', 'TN': '#2166AC'}
    focus = widgets.Dropdown(options=classes, value='Cat', description='Positive class:',
                             style={'description_width': 'initial'},
                             layout=widgets.Layout(width='240px'))
    reveal = widgets.Checkbox(value=False, description='Show answers', indent=False,
                              layout=widgets.Layout(width='145px'))
    highlight = widgets.Dropdown(options=['All', 'TP', 'FP', 'FN', 'TN'], value='All',
                                 description='Highlight:', disabled=True,
                                 layout=widgets.Layout(width='190px'))
    canvas = widgets.Image(format='png', layout=widgets.Layout(
        width='100%', max_width='100%', height='auto'))

    def draw(change=None):
        highlight.disabled = not reveal.value
        positive = focus.value
        actual_positive = data['Actual'].eq(positive).to_numpy()
        predicted_positive = data['Predicted'].eq(positive).to_numpy()
        outcomes = np.where(actual_positive, np.where(predicted_positive, 'TP', 'FN'),
                            np.where(predicted_positive, 'FP', 'TN'))
        counts = confusion_matrix(actual_positive, predicted_positive, labels=[False, True])

        with plt.ioff():
            fig = plt.figure(figsize=(12.6, 6.5))
            try:
                grid = fig.add_gridspec(3, 6, width_ratios=[1, 1, 1, 1, 1.1, 1.1],
                                       left=0.015, right=0.99, bottom=0.07, top=0.92,
                                       wspace=0.16, hspace=0.42)
                fig.text(0.015, 0.965, f'Positive class: {positive}', fontsize=16, weight='bold')
                for i, row in data.iterrows():
                    ax = fig.add_subplot(grid[i // 4, i % 4])
                    outcome = outcomes[i]
                    selected = not reveal.value or highlight.value == 'All' or highlight.value == outcome
                    color = outcome_colors[outcome] if reveal.value else '#8B8B8B'
                    ax.imshow(photos[i], alpha=1.0 if selected else 0.18)
                    ax.grid(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(2.5)
                        spine.set_color(color if selected else '#C8C8C8')
                    badge = f'{row["Image"]}: {outcome}' if reveal.value else row['Image']
                    ax.text(0.03, 0.97, badge, transform=ax.transAxes, va='top', color='white',
                            fontsize=10, weight='bold', bbox=dict(facecolor=color, edgecolor='none', pad=2))
                    ax.set_xlabel(f'Actual: {row["Actual"]}\nPredicted: {row["Predicted"]}',
                                  fontsize=10, labelpad=4, color='#222222')

                matrix_ax = fig.add_subplot(grid[:, 4:])
                matrix_ax.set_xlim(-0.65, 2.05)
                matrix_ax.set_ylim(-0.7, 2.95)
                matrix_ax.set_aspect('equal')
                matrix_ax.axis('off')
                matrix_ax.text(1, 2.82, f'{positive} against the rest', ha='center', fontsize=14, weight='bold')
                matrix_ax.text(1, 2.50, 'Predicted', ha='center', fontsize=11)
                matrix_ax.text(0.5, 2.25, f'Not {positive}', ha='center', fontsize=11)
                matrix_ax.text(1.5, 2.25, positive, ha='center', fontsize=11)
                matrix_ax.text(-0.58, 1, 'Actual', ha='center', va='center', rotation=90, fontsize=11)
                matrix_ax.text(-0.06, 1.5, f'Not {positive}', ha='right', va='center', fontsize=10)
                matrix_ax.text(-0.06, 0.5, positive, ha='right', va='center', fontsize=10)
                for row, names in enumerate([['TN', 'FP'], ['FN', 'TP']]):
                    for col, outcome in enumerate(names):
                        color = outcome_colors[outcome] if reveal.value else '#8B8B8B'
                        active = not reveal.value or highlight.value in ('All', outcome)
                        x, y = col, 1 - row
                        matrix_ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color,
                                                       alpha=0.14 if active else 0.04, edgecolor='none'))
                        matrix_ax.add_patch(Rectangle((x, y), 1, 1, facecolor='none',
                                                       edgecolor='#E0E0E0', linewidth=2))
                        label = f'{outcome}\n{counts[row, col]}' if reveal.value else '?'
                        matrix_ax.text(x + 0.5, y + 0.5, label, ha='center', va='center',
                                       fontsize=20, weight='bold', color=color if active else '#C8C8C8')
                instruction = ('Counts use all 12 photos. Highlighting changes emphasis only.'
                               if reveal.value else 'Name TP, FP, FN, and TN. Then reveal the answers.')
                matrix_ax.text(1, -0.25, instruction.replace('. ', '.\n'),
                               ha='center', va='top', fontsize=10)
                with BytesIO() as buffer:
                    fig.savefig(buffer, format='png', dpi=110)
                    canvas.value = buffer.getvalue()
            finally:
                plt.close(fig)

    for control in (focus, reveal, highlight):
        control.observe(draw, names='value')
    draw()
    controls = widgets.HBox([focus, reveal, highlight], layout=widgets.Layout(
        width='100%', flex_flow='row wrap', align_items='center'))
    view = widgets.VBox([controls, canvas], layout=widgets.Layout(width='100%', max_width='100%'))
    display(view)
    return view


def scatter(ax, X, y):
    for k, (name, color, marker) in enumerate(zip(NAMES, COLORS, MARKERS)):
        rows = np.asarray(y) == k
        ax.scatter(X.iloc[rows, 0], X.iloc[rows, 1], s=36, alpha=0.65,
                   c=color, marker=marker, edgecolors='white', linewidths=0.4,
                   label=name)
    ax.set_xlabel(X.columns[0].capitalize())
    ax.set_ylabel(X.columns[1].capitalize())
    ax.spines[['top', 'right']].set_visible(False)


def boundary(ax, model, X, y, title, limits=None):
    limits = limits or (X.iloc[:, 0].min() - 0.4, X.iloc[:, 0].max() + 0.4,
                        X.iloc[:, 1].min() - 0.3, X.iloc[:, 1].max() + 0.3)
    xx, yy = np.meshgrid(np.linspace(*limits[:2], 160), np.linspace(*limits[2:], 160))
    grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
    z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5, 2.5],
                cmap=ListedColormap(COLORS), alpha=0.18)
    scatter(ax, X, y)
    ax.set(xlim=limits[:2], ylim=limits[2:])
    ax.set_title(title)


def model_view(model, X, y, title):
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    boundary(ax, model, X, y, title)
    ax.legend(loc='best')
    return fig


def tree_view(model, feature_names, class_names=None, title='A sequence of feature thresholds'):
    fig, ax = plt.subplots(figsize=(10, 4.6), layout='constrained')
    plot_tree(model, feature_names=list(feature_names), class_names=class_names,
              filled=False, rounded=True, precision=2, fontsize=10, ax=ax)
    ax.set_title(title)
    return fig


def pair_view(model, train, valid, features, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    limits = (min(train[features[0]].min(), valid[features[0]].min()) - 0.4,
              max(train[features[0]].max(), valid[features[0]].max()) + 0.4,
              min(train[features[1]].min(), valid[features[1]].min()) - 0.3,
              max(train[features[1]].max(), valid[features[1]].max()) + 0.3)
    for ax, data, label in zip(axes, [train, valid], ['Training', 'Validation']):
        score = model.score(data[features], data.label)
        boundary(ax, model, data[features], data.label,
                 f'{label}: {score:.1%} accuracy', limits)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    return fig


def depth_view(depths, training, validation):
    fig, ax = plt.subplots(figsize=(9, 4.4), layout='constrained')
    ax.plot(depths, training, '-o', label='Training', linewidth=1.5, markersize=5)
    ax.plot(depths, validation, '--s', label='Validation', linewidth=1.5, markersize=5)
    ax.set(xlabel='Maximum tree depth', ylabel='Accuracy', ylim=(0, 1.04))
    ax.set_title('Choose complexity using validation performance')
    ax.legend()
    return fig


def forest_members(model, X, y, title):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout='constrained')
    for i, ax in enumerate(axes):
        # Individual forest estimators use arrays internally.
        member = model.estimators_[i]
        class WithNames:
            def predict(self, frame):
                return member.predict(frame.to_numpy())
        boundary(ax, WithNames(), X, y, f'Tree {i + 1}')
    fig.suptitle(title, fontsize=16, fontweight='bold')
    return fig


def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    return 1 - np.sum((counts / counts.sum()) ** 2) if len(counts) else 0.0


def split_values(X, y, threshold, regression=False):
    left = np.asarray(X) <= threshold
    a, b = np.asarray(y)[left], np.asarray(y)[~left]
    loss = np.var if regression else gini
    la, lb = loss(a), loss(b)
    weighted = (len(a) * la + len(b) * lb) / len(y)
    return left, la, lb, weighted


def split_figure(df, feature, target, index, regression=False):
    values = np.sort(df[feature].unique())
    thresholds = (values[:-1] + values[1:]) / 2
    threshold = thresholds[index]
    y = df[target].to_numpy()
    left, la, lb, weighted = split_values(df[feature], y, threshold, regression)
    losses = [split_values(df[feature], y, t, regression)[-1] for t in thresholds]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), layout='constrained')
    if regression:
        axes[0].scatter(df[feature], y, s=28, alpha=0.6, edgecolors='white', linewidths=0.4)
        for rows in [left, ~left]:
            axes[0].hlines(y[rows].mean(), df[feature][rows].min(), df[feature][rows].max(),
                           color='#810F7C', linewidth=2)
        axes[0].set_ylabel(target.capitalize())
    else:
        for k, (name, color, marker) in enumerate(zip(NAMES, COLORS, MARKERS)):
            rows = y == k
            axes[0].scatter(df.loc[rows, feature], y[rows], c=color, marker=marker,
                            alpha=0.6, s=35, label=name)
        axes[0].set_yticks([0, 1, 2], NAMES)
        axes[0].set_ylabel('Observed species')
    axes[0].axvline(threshold, color='#810F7C', linewidth=2)
    axes[0].set_xlabel(feature.capitalize())
    axes[0].set_title(f'Split at {threshold:.2f}: {left.sum()} left, {(~left).sum()} right')
    axes[1].plot(thresholds, losses, '-o', linewidth=1, markersize=3)
    axes[1].scatter([threshold], [weighted], c='#810F7C', s=65, zorder=4)
    axes[1].set_xlabel(f'{feature.capitalize()} threshold')
    axes[1].set_ylabel('Weighted MSE' if regression else 'Weighted Gini impurity')
    axes[1].set_title(f'Weighted loss = {weighted:.3f}')
    fig.supxlabel(f'{left.sum()}/{len(y)} × {la:.3f} + {(~left).sum()}/{len(y)} × {lb:.3f} = {weighted:.3f}', fontsize=11)
    return fig


def split_widget(df, features, target='label', regression=False):
    feature = widgets.Dropdown(options=features, description='Feature:', layout=widgets.Layout(width='48%'))
    slider = widgets.IntSlider(value=0, min=0, max=1, description='Split:', continuous_update=False,
                               layout=widgets.Layout(width='48%'))
    canvas = widgets.HTML(layout=widgets.Layout(width='100%', overflow='hidden'))
    busy = False
    def draw(change=None):
        nonlocal busy
        if busy:
            return
        busy = True
        try:
            count = df[feature.value].nunique() - 1
            slider.max = count - 1
            slider.value = min(slider.value, slider.max)
            fig = split_figure(df, feature.value, target, slider.value, regression)
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            plt.close(fig)
            payload = base64.b64encode(buf.getvalue()).decode('ascii')
            canvas.value = '<img alt="Candidate split and weighted loss" style="width:100%;height:auto;display:block" src="data:image/png;base64,' + payload + '">'
        finally:
            busy = False
    feature.observe(draw, names='value')
    slider.observe(draw, names='value')
    draw()
    display(widgets.VBox([widgets.HBox([feature, slider]), canvas], layout=widgets.Layout(width='100%')))


def save_tree_leaf_path_assets(model, X, y, output_dir, queries=None, class_names=None, include_overview=False):
    """Save two full dtreeviz diagrams from one fitted model, without refitting.

    The only change in the dtreeviz copy is encoding string class labels as
    integers. Its fitted nodes, thresholds, counts and probabilities stay intact.
    """
    from copy import deepcopy
    import json
    import re
    import shutil
    import subprocess
    import tempfile
    import dtreeviz
    from dtreeviz.utils import inline_svg_images
    import pymupdf

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = list(X.columns) if hasattr(X, 'columns') else ['petal length (cm)', 'petal width (cm)']
    x_values, y_values = np.asarray(X), np.asarray(y)
    if queries is None:
        queries = np.array([[5.5, 2.0], [4.8, 1.8]])
    query_values = np.asarray(queries, dtype=float)
    if query_values.shape != (2, x_values.shape[1]):
        raise ValueError('Provide two input rows with the same columns as X.')
    classes = list(model.classes_)
    names = list(class_names) if class_names is not None else [str(value) for value in classes]
    if all(isinstance(value, (int, np.integer)) for value in classes) and class_names is None and len(classes) == 3:
        names = NAMES.copy()
    label_ids = {value: index for index, value in enumerate(classes)}
    encoded_y = np.array([label_ids[value] for value in y_values])
    visual_model = deepcopy(model)
    visual_model.classes_ = np.arange(len(classes))
    if hasattr(visual_model, 'feature_names_in_'):
        del visual_model.feature_names_in_
    palette = ['#008FD5', '#FC4F30', '#E5AE38']
    background, ink, highlight = '#F0F0F0', '#444445', '#0066FF'
    dot_exe = shutil.which('dot')
    if dot_exe is None:
        for folder in (Path(sys.prefix) / 'Library' / 'bin',
                       Path(sys.prefix) / 'Library' / 'bin' / 'graphviz',
                       Path(sys.prefix) / 'Library' / 'Graphviz' / 'bin'):
            candidate = folder / ('dot.exe' if os.name == 'nt' else 'dot')
            if candidate.is_file():
                dot_exe = str(candidate)
                break
    if dot_exe is None:
        raise RuntimeError('Graphviz is needed to display the decision paths.')
    paths, records, layouts = [], [], []
    with warnings.catch_warnings(), plt.ioff(), plt.rc_context(), tempfile.TemporaryDirectory(prefix='cme538_leaf_paths_') as temp:
        warnings.filterwarnings('ignore', message='X does not have valid feature names.*', category=UserWarning)
        plt.rcParams.update({'figure.facecolor': background, 'axes.facecolor': background,
                             'savefig.facecolor': background, 'text.color': ink})
        leaf_ids = model.apply(x_values)
        stages = list(zip(['pure', 'mixed'], query_values))
        if include_overview:
            stages.append(('overview', query_values[0]))
        for index, (stem, query) in enumerate(stages):
            stage = Path(temp) / stem
            stage.mkdir()
            leaf = int(model.apply(query[None, :])[0])
            decision_path = model.decision_path(query[None, :]).indices.tolist()
            counts = np.bincount(encoded_y[leaf_ids == leaf], minlength=len(classes))
            probabilities = model.predict_proba(query[None, :])[0]
            np.testing.assert_allclose(probabilities, counts / counts.sum())
            view = dtreeviz.model(
                visual_model, x_values, encoded_y, target_name='Species',
                feature_names=features, class_names=names,
            ).view(x=None if stem == 'overview' else query,
                   show_just_path=False, fontname='Arial',
                   label_fontsize=17, ticks_fontsize=12,
                   colors={'classes': [None, None, palette[:2], palette],
                           'highlight': highlight, 'text': ink, 'axis_label': ink,
                           'tick_label': ink, 'leaf_label': ink, 'arrow': '#888888',
                           'larrow': '#888888', 'rarrow': '#888888'}, ai_chat=False)
            dot = view.dot
            # The notebook and slide provide the measurements and probabilities.
            # Remove dtreeviz's duplicate legend and query table, not tree nodes.
            dot = re.sub(r'^[ \t]*subgraph cluster_(?:legend|instance)\s*\{.*?^[ \t]*\}',
                         '', dot, flags=re.S | re.M)
            dot = re.sub(r'^.*?\bleaf\d+ -> X_y.*$', '', dot, flags=re.M)
            dot = dot.replace('style="dashed"', 'style="solid"')
            dot = dot.replace('penwidth=".5" color="' + highlight + '"',
                              'penwidth="2.2" color="' + highlight + '"')
            dot = dot.replace('penwidth=1.2 ', 'penwidth=2.8 ')
            dot = dot.replace('ranksep=.2;', 'ranksep=.08;')
            for number, source in enumerate(re.findall(r'<img src="(.*?)"', dot)):
                svg = Path(source).read_text(encoding='utf-8')
                svg = svg.replace('#ffffff', background).replace('#FFFFFF', background)
                copied = stage / f'node_{number}.svg'
                copied.write_text(svg, encoding='utf-8')
                dot = dot.replace(source, str(copied))
            source_path = stage / 'tree.dot'
            source_path.write_text(dot, encoding='utf-8')
            svg_path = output_dir / f'tree_leaf_paths_{stem}.svg'
            subprocess.run([dot_exe, '-Tsvg', f'-Gbgcolor={background}',
                            str(source_path), '-o', str(svg_path)], check=True, capture_output=True)
            svg_path.write_text(inline_svg_images(svg_path.read_text(encoding='utf-8')), encoding='utf-8')
            layout = subprocess.run([dot_exe, '-Tplain', str(source_path)],
                                    check=True, capture_output=True, text=True).stdout
            layouts.append({line.split()[1]: list(map(float, line.split()[2:6]))
                            for line in layout.splitlines() if line.startswith('node ')})
            raster_dot = dot
            for number, source in enumerate(re.findall(r'<img src="(.*?)"', dot)):
                with pymupdf.open(source) as document:
                    pixels = document[0].get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=True)
                    raster_path = stage / f'node_{number}.png'
                    pixels.save(str(raster_path))
                raster_dot = raster_dot.replace(source, str(raster_path))
            source_path.write_text(raster_dot, encoding='utf-8')
            png_path = output_dir / f'tree_leaf_paths_{stem}.png'
            subprocess.run([dot_exe, '-Tpng', '-Gdpi=144', f'-Gbgcolor={background}',
                            str(source_path), '-o', str(png_path)], check=True, capture_output=True)
            if stem != 'overview':
                paths.append(png_path)
                records.append(dict(name=stem, query=query.tolist(), leaf=leaf, path=decision_path,
                                    counts=counts.tolist(), probabilities=probabilities.tolist(),
                                    prediction=names[int(probabilities.argmax())]))
    if layouts[0].keys() != layouts[1].keys():
        raise AssertionError('Both diagrams must contain the same complete tree.')
    for node in layouts[0]:
        np.testing.assert_allclose(layouts[0][node], layouts[1][node], rtol=0, atol=.015,
                                   err_msg='The two tree layouts must stay aligned.')
    metadata = dict(features=features, classes=names, parameters=model.get_params(),
                    examples=records, node_count=model.tree_.node_count,
                    node_layouts=layouts, colors=palette, background=background,
                    text_color=ink, highlight=highlight)
    (output_dir / 'tree_leaf_paths.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return paths, metadata


def tree_leaf_probability_comparison(model, X, y, queries=None, class_names=None):
    """Show pure and mixed leaf paths side by side for the same fitted tree."""
    import tempfile
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix='cme538_leaf_comparison_') as temp:
        paths, metadata = save_tree_leaf_path_assets(model, X, y, temp, queries, class_names)
        images = [np.asarray(Image.open(path).convert('RGB')) for path in paths]
    with plt.rc_context(), plt.ioff():
        plt.rcParams.update({'figure.facecolor': '#F0F0F0', 'axes.facecolor': '#F0F0F0',
                             'text.color': '#444445', 'font.family': 'DejaVu Sans'})
        fig = plt.figure(figsize=(17, 10), dpi=120)
        for index, (image, example) in enumerate(zip(images, metadata['examples'])):
            left = .025 + index * .50
            ax = fig.add_axes([left, .15, .45, .70])
            ax.imshow(image)
            ax.set_axis_off()
            heading = 'Pure leaf' if index == 0 else 'Mixed leaf'
            fig.text(left + .225, .95, heading, ha='center', weight='bold', fontsize=20)
            length, width = example['query']
            fig.text(left + .225, .905, f'Petal length {length:g} cm, petal width {width:g} cm',
                     ha='center', fontsize=14)
            predicted = example['prediction']
            class_index = metadata['classes'].index(predicted)
            count, total = example['counts'][class_index], sum(example['counts'])
            fig.text(left + .225, .085, f'P({predicted}) = {count}/{total} = {example["probabilities"][class_index]:.3f}',
                     ha='center', weight='bold', fontsize=17)
        return fig


def plot_probability_variance(variances):
    """Show per-flower variance for increasing forest sizes."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    distributions = variances.to_numpy().T
    positions = np.arange(len(distributions))
    means = distributions.mean(axis=1)
    jitter = np.random.default_rng(20261026).uniform(-.13, .13, len(variances))
    colors = [FTE_BLUE] * len(distributions)
    ink = '#444445'
    fig, ax = plt.subplots(figsize=(12, 4.75))
    fig.subplots_adjust(left=.09, right=.985, bottom=.22, top=.88)

    violins = ax.violinplot(
        distributions.T, positions=positions, widths=.70, points=150,
        bw_method=.30, showmeans=False, showmedians=False, showextrema=False,
    )
    for body, color in zip(violins['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(.22)
        body.set_linewidth(1.3)

    for position, color, distribution, mean in zip(positions, colors, distributions, means):
        ax.scatter(position + jitter, distribution, s=26, color=color,
                   alpha=.65, edgecolors='white', linewidths=.45, zorder=3)
        ax.plot([position - .16, position + .16], [mean, mean],
                color=ink, linewidth=1.6, zorder=4)
        ax.scatter([position], [mean], marker='D', s=65, color=FTE_YELLOW,
                   edgecolors=ink, linewidths=.8, zorder=5)

    ax.set_xticks(positions, list(variances.columns))
    ax.set_xlim(-.50, len(distributions) - .50)
    ax.set_ylim(-.006, .185)
    ax.yaxis.set_major_locator(MultipleLocator(.04))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('Probability variance per flower', labelpad=10)
    ax.tick_params(axis='x', length=0, pad=8)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.grid(axis='x', visible=False)
    ax.set_axisbelow(True)
    ax.text((len(distributions) - 1) / 2, -.25, 'Number of trees in the random forest',
            transform=ax.get_xaxis_transform(), ha='center', va='top')
    handles = [
        Line2D([], [], marker='o', linestyle='', color=FTE_BLUE, alpha=.65,
               markeredgecolor='white', markersize=6, label='One held-out flower'),
        Line2D([], [], marker='D', color=ink, markerfacecolor=FTE_YELLOW,
               markeredgewidth=.8, markersize=6, linewidth=1.6,
               label=f'Mean across {len(variances)} flowers'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, 1.01),
               ncol=2, frameon=False, handletextpad=.7, columnspacing=2.5)
    return fig


def plot_forest_probability_growth(prefix_probabilities):
    """Display all fitted forests for the fixed-flower averaging example."""
    probabilities = np.asarray(prefix_probabilities)
    counts = np.arange(1, probabilities.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={'width_ratios': [1.4, 1]})
    for history in probabilities:
        axes[0].plot(counts, history, color=FTE_GRAY, alpha=0.45, linewidth=0.9)
        axes[0].scatter(counts[-1], history[-1], color=FTE_BLUE,
                        s=30, alpha=0.6, zorder=3)
    axes[0].set(xscale='log', xlim=(0.9, 115), ylim=(-0.03, 1.03),
                xlabel='Trees in Each Forest', ylabel='P(Versicolor)',
                title='50 Separately Trained Forests', xticks=[1, 10, 100],
                xticklabels=['1', '10', '100'], yticks=[0, 0.5, 1])
    final_probabilities = probabilities[:, -1]
    violin = axes[1].violinplot(
        final_probabilities, positions=[0], orientation='horizontal', widths=0.5,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in violin['bodies']:
        body.set_facecolor(FTE_BLUE)
        body.set_edgecolor(FTE_BLUE)
        body.set_alpha(0.18)
        body.set_linewidth(1.3)
    offsets = np.random.default_rng(538).permutation(
        np.linspace(-0.13, 0.13, len(final_probabilities))
    )
    axes[1].scatter(final_probabilities, offsets, color=FTE_BLUE, s=35,
                    alpha=0.72, zorder=4)
    median = np.median(final_probabilities)
    axes[1].plot([median, median], [-0.27, 0.27], color='#444445',
                 linewidth=3, zorder=5)
    axes[1].set(xlim=(-0.03, 1.03), ylim=(-0.62, 0.62), xlabel='P(versicolor)',
                title='One Prediction per Forest', yticks=[], xticks=[0, 0.5, 1])
    axes[1].text(0.5, 0.88, f'Median: {median:.3f}',
                 transform=axes[1].transAxes, ha='center', va='center', fontsize=12)
    axes[1].text(0.5, 0.10,
                 f'P(versicolor) SD: {final_probabilities.std(ddof=0):.3f}',
                 transform=axes[1].transAxes, ha='center', va='center', fontsize=12)
    axes[1].grid(axis='y', visible=False)
    fig.suptitle('100 Trees in Each Forest', y=1.01, fontsize=16)
    fig.tight_layout()
    return fig
