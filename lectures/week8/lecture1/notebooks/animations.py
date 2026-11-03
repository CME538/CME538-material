"""Rebuild the Lecture 8.1 teaching videos from scikit-learn models.

Run with Anaconda base, for example::

    python animations.py --output ../slides/animations
    python animations.py --only tree_growth --output ../slides/animations

Dependencies: numpy, matplotlib, scikit-learn, Pillow and imageio-ffmpeg.
The growing-tree animation also uses dtreeviz, Graphviz and PyMuPDF.
Its notebook player additionally uses ipywidgets.
All data comes from sklearn.datasets.load_iris; no network access is needed.
The held-out observations are displayed for teaching, not used to select a model.

Visual teaching references (the data and models here are generated locally):
https://r2d3.us/visual-intro-to-machine-learning-part-2/
https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


WIDTH, HEIGHT, FPS = 1280, 600, 24
BACKGROUND = "#F0F0F0"
INK, MUTED = "#252525", "#626262"
# Same FiveThirtyEight palette and plotting defaults as utils.configure_notebook.
# Keep the constants local so rebuilding a video does not import widget helpers.
COLORS = ["#30A2DA", "#FC4F30", "#E5AE38"]
NAMES = ["Setosa", "Versicolor", "Virginica"]
TRAIN_COLOR, TEST_COLOR = "#30A2DA", "#FC4F30"
TREE_DEPTHS = list(range(1, 11)) + [None]
FOREST_COUNTS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100]


def _data():
    iris = load_iris()
    x, y = iris.data[:, :2], iris.target
    train, heldout = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=42
    )
    x1 = np.linspace(4.1, 8.1, 240)
    x2 = np.linspace(1.8, 4.6, 180)
    xx, yy = np.meshgrid(x1, x2)
    return dict(x=x, y=y, train=train, heldout=heldout,
                xx=xx, yy=yy, grid=np.c_[xx.ravel(), yy.ravel()])


def _style():
    plt.style.use("fivethirtyeight")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.titleweight": "bold", "axes.titlesize": 16,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "savefig.facecolor": BACKGROUND,
        "grid.color": "#CBCBCB", "grid.linewidth": .5, "grid.alpha": .45,
    })


def _figure(title, subtitle):
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    fig.text(.055, .955, title, fontsize=23, weight="bold", va="top")
    fig.text(.055, .900, subtitle, fontsize=13.5, color=MUTED, va="top")
    return fig


def _boundary(ax, data, probabilities, title, bold_boundary=False):
    # Tint the class regions lightly; the points retain the actual class colours.
    rgb = probabilities @ np.array([to_rgb(c) for c in COLORS])
    rgb = .23 * rgb + .77 * np.array(to_rgb(BACKGROUND))
    ax.imshow(rgb.reshape(data["xx"].shape + (3,)), origin="lower",
              extent=(4.1, 8.1, 1.8, 4.6), aspect="auto", interpolation="nearest")
    region = probabilities.argmax(axis=1).reshape(data["xx"].shape)
    if len(np.unique(region)) > 1:
        for k in np.unique(region):
            ax.contour(data["xx"], data["yy"], (region == k).astype(float),
                       levels=[.5], colors=INK if bold_boundary else "#777777",
                       linewidths=2.2 if bold_boundary else .65,
                       alpha=1 if bold_boundary else .7, zorder=2)
    for k, color in enumerate(COLORS):
        train = data["train"][data["y"][data["train"]] == k]
        heldout = data["heldout"][data["y"][data["heldout"]] == k]
        ax.scatter(*data["x"][train].T, s=25, color=color, alpha=.55,
                   edgecolors="white", linewidths=.3, zorder=3)
        ax.scatter(*data["x"][heldout].T, s=76, color=color, marker="X",
                   edgecolors="#333333", linewidths=.55, zorder=4)
    ax.set(xlim=(4.1, 8.1), ylim=(1.8, 4.6),
           xlabel="Sepal length (cm)", ylabel="Sepal width (cm)")
    ax.set_title(title, loc="left", pad=11)
    ax.grid(color="#CBCBCB", linewidth=.5, alpha=.45)
    ax.set_axisbelow(True)


def _legend(fig, y=.055):
    handles = [Line2D([], [], color=c, marker="o", linestyle="", markersize=8,
                      label=n) for c, n in zip(COLORS, NAMES)]
    handles += [Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=6,
                       alpha=.55, label="Training flower"),
                Line2D([], [], color=MUTED, marker="X", linestyle="", markersize=9,
                       label="Held-out flower")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.52, y),
               ncol=5, frameon=False, fontsize=12.5, columnspacing=1.7)


def _frame(fig):
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    assert pixels.shape == (HEIGHT, WIDTH, 3)
    return pixels


def _write_video(path, frames, seconds, captions, metadata):
    import imageio_ffmpeg

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path), (WIDTH, HEIGHT), fps=FPS, codec="libx264", quality=8,
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-movflags", "+faststart", "-profile:v", "high", "-level", "3.1"],
    )
    writer.send(None)
    total_frames = 0
    try:
        for frame, hold in zip(frames, seconds):
            repeats = round(hold * FPS)
            for _ in range(repeats):
                writer.send(frame)
            total_frames += repeats
    finally:
        writer.close()
    for suffix, i in [("poster", 0), ("middle", len(frames) // 2), ("final", len(frames) - 1)]:
        Image.fromarray(frames[i]).save(path.with_name(f"{path.stem}_{suffix}.png"))
    metadata.update(width=WIDTH, height=HEIGHT, fps=FPS,
                    duration_seconds=total_frames / FPS,
                    codec="H.264", pixel_format="yuv420p", captions=captions,
                    style_reference="utils.py: configure_notebook; FiveThirtyEight palette",
                    visual_references=[
                        "https://r2d3.us/visual-intro-to-machine-learning-part-2/",
                        "https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html",
                    ])
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def make_tree_complexity(output):
    """Increase depth while preserving the data, split, axes, colours and seed."""
    _style()
    data = _data()
    x, y, tr, te = (data[k] for k in ["x", "y", "train", "heldout"])
    models = [DecisionTreeClassifier(max_depth=d, random_state=42).fit(x[tr], y[tr])
              for d in TREE_DEPTHS]
    train_errors = np.array([1 - m.score(x[tr], y[tr]) for m in models])
    test_errors = np.array([1 - m.score(x[te], y[te]) for m in models])
    # Identical feature values with different labels impose a lower bound on
    # training error. An unrestricted tree reaches that bound on these data.
    unique_x, inverse = np.unique(x[tr], axis=0, return_inverse=True)
    collision_groups = []
    minimum_errors = 0
    for group, values in enumerate(unique_x):
        counts = np.bincount(y[tr][inverse == group], minlength=3)
        unavoidable = int(counts.sum() - counts.max())
        minimum_errors += unavoidable
        if unavoidable:
            collision_groups.append(dict(features=values.tolist(),
                                         class_counts=counts.tolist(),
                                         minimum_errors=unavoidable))
    final_training_errors = int(np.sum(models[-1].predict(x[tr]) != y[tr]))
    assert final_training_errors == minimum_errors
    frames = []
    actual_depths = [model.get_depth() for model in models]
    labels = [str(d) for d in actual_depths]
    for i, (depth, model) in enumerate(zip(TREE_DEPTHS, models)):
        fig = _figure("Tree depth and classification error",
                      "Iris sepal measurements. 112 training flowers, 38 held-out flowers.")
        left = fig.add_axes([.065, .235, .41, .545])
        name = f"Maximum depth {depth}" if depth is not None else f"Unrestricted tree (depth {model.get_depth()})"
        _boundary(left, data, model.predict_proba(data["grid"]), name)
        right = fig.add_axes([.59, .235, .355, .545])
        right.set_title("Training and held-out error", loc="left", pad=11)
        right.plot(np.arange(i + 1), train_errors[:i + 1] * 100,
                   color=TRAIN_COLOR, marker="o", lw=2.5, label="Training error")
        right.plot(np.arange(i + 1), test_errors[:i + 1] * 100,
                   color=TEST_COLOR, marker="s", lw=2.5, label="Held-out error")
        right.scatter([i, i], [train_errors[i] * 100, test_errors[i] * 100],
                      color=[TRAIN_COLOR, TEST_COLOR], s=90, zorder=5)
        right.set(xlim=(-.2, len(models) - .75), ylim=(0, 55), xticks=np.arange(len(models)),
                  xticklabels=labels, xlabel="Maximum tree depth", ylabel="Misclassified flowers (%)")
        right.grid(color="#CBCBCB", linewidth=.5, alpha=.45)
        right.legend(loc="upper left", frameon=False, fontsize=12)
        fig.text(.595, .12, f"Training: {train_errors[i]:.1%}     Held-out: {test_errors[i]:.1%}",
                 fontsize=15, weight="bold")
        _legend(fig)
        frames.append(_frame(fig))
    return _write_video(Path(output) / "tree_complexity.mp4", frames,
                        [1.8] * (len(models) - 1) + [4.5], [],
                        dict(seed=42, features=["sepal length (cm)", "sepal width (cm)"],
                             depths=TREE_DEPTHS, train_error=train_errors.tolist(),
                             heldout_error=test_errors.tolist(), train_indices=tr.tolist(),
                             heldout_indices=te.tolist(), actual_depths=actual_depths,
                             minimum_training_errors=minimum_errors,
                             collision_groups=collision_groups))


def make_forest_aggregation(output):
    """Reveal individual trees and the class-probability average of all seen trees."""
    _style()
    data = _data()
    x, y, tr, te = (data[k] for k in ["x", "y", "train", "heldout"])
    forest = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_split=10,
        max_features="sqrt", bootstrap=True, random_state=42,
    ).fit(x[tr], y[tr])
    grid_predictions = np.array([tree.predict_proba(data["grid"]) for tree in forest.estimators_])
    test_predictions = np.array([tree.predict_proba(x[te]) for tree in forest.estimators_])
    grid_cumulative = grid_predictions.cumsum(axis=0)
    test_cumulative = test_predictions.cumsum(axis=0)
    frames, results = [], []
    for n in FOREST_COUNTS:
        fig = _figure("A single tree versus a random forest",
                      "Bootstrap samples and random feature subsets. Maximum depth 4, minimum split size 10.")
        left, right = (fig.add_axes([v, .235, .405, .545]) for v in [.065, .56])
        _boundary(left, data, grid_predictions[n - 1], f"Single tree {n}")
        _boundary(right, data, grid_cumulative[n - 1] / n,
                  f"Forest with {n} {'tree' if n == 1 else 'trees'}",
                  bold_boundary=n == FOREST_COUNTS[-1])
        individual_wrong = int(np.sum(test_predictions[n - 1].argmax(axis=1) != y[te]))
        ensemble_wrong = int(np.sum(test_cumulative[n - 1].argmax(axis=1) != y[te]))
        fig.text(.065, .12, f"Held-out errors: {individual_wrong}/38 ({individual_wrong/38:.0%})", fontsize=15)
        fig.text(.56, .12, f"Held-out errors: {ensemble_wrong}/38 ({ensemble_wrong/38:.0%})", fontsize=15, weight="bold")
        _legend(fig)
        frames.append(_frame(fig))
        results.append(dict(trees=n, individual_errors=individual_wrong, ensemble_errors=ensemble_wrong))
    assert np.allclose(test_cumulative[-1] / 100, forest.predict_proba(x[te]))
    return _write_video(Path(output) / "forest_aggregation.mp4", frames,
                        [1.45] * 12 + [4.5], [],
                        dict(seed=42, features=["sepal length (cm)", "sepal width (cm)"],
                             results=results, train_indices=tr.tolist(), heldout_indices=te.tolist(),
                             parameters=forest.get_params()))


TREE_GROWTH_ORDER = [0, 2, 3, 12, 4, 7, 9]
TREE_GROWTH_SIZE = (1920, 1000)
TREE_GROWTH_VERSION = 3


def _tree_growth_models():
    """Reveal a fixed fitted tree breadth first, without fitting new models."""
    from copy import deepcopy

    iris = load_iris()
    x, y = iris.data[:, 2:], iris.target
    # Seed 0 selects the petal-width root shown in the lecture. Petal length
    # 2.45 gives the same training partition and is tied with width 0.80.
    full = DecisionTreeClassifier(random_state=0).fit(x, y)
    assert full.tree_.feature[0] == 1
    assert np.isclose(full.tree_.threshold[0], .8)
    assert set(TREE_GROWTH_ORDER) == set(np.flatnonzero(full.tree_.feature >= 0))
    models = []
    for count in range(len(TREE_GROWTH_ORDER) + 1):
        partial = deepcopy(full)
        active = set(TREE_GROWTH_ORDER[:count])
        for node in range(full.tree_.node_count):
            if node not in active:
                partial.tree_.children_left[node] = -1
                partial.tree_.children_right[node] = -1
                partial.tree_.feature[node] = -2
                partial.tree_.threshold[node] = -2
        models.append(partial)
    return x, y, full, models


def _tree_growth_rectangles(model, bounds=(.6, 7.1, 0., 2.7)):
    """Return each reachable node's rectangle in length/width coordinates."""
    rectangles = {}

    def visit(node, rect):
        rectangles[node] = rect
        left, right = model.tree_.children_left[node], model.tree_.children_right[node]
        if left == -1:
            return
        x0, x1, y0, y1 = rect
        threshold = model.tree_.threshold[node]
        if model.tree_.feature[node] == 0:
            visit(left, (x0, threshold, y0, y1))
            visit(right, (threshold, x1, y0, y1))
        else:
            visit(left, (x0, x1, y0, threshold))
            visit(right, (x0, x1, threshold, y1))

    visit(0, bounds)
    return rectangles


def _tree_growth_graphviz():
    """Find Graphviz in the active environment, as configure_notebook does."""
    import os
    import shutil
    import sys

    for folder in (Path(sys.prefix) / "Library" / "bin",
                   Path(sys.prefix) / "Library" / "bin" / "graphviz",
                   Path(sys.prefix) / "Library" / "Graphviz" / "bin"):
        if (folder / "dot.exe").is_file() or (folder / "dot").is_file():
            return str(folder / ("dot.exe" if os.name == "nt" else "dot"))
    executable = shutil.which("dot")
    if executable is None:
        raise RuntimeError("Graphviz is needed to draw the tree. Install Graphviz in the active environment.")
    return executable


def _tree_growth_diagram(model, x, y, new_node, folder):
    """Render real dtreeviz node histograms/pies with the Graphviz layout."""
    import re
    import subprocess
    import dtreeviz
    import pymupdf

    view = dtreeviz.model(
        model, x, y, target_name="Species",
        feature_names=["Petal length (cm)", "Petal width (cm)"],
        class_names=[name.lower() for name in NAMES],
    ).view(fontname="Arial", label_fontsize=15, ticks_fontsize=11,
           highlight_path=[],
           colors={"classes": [None, None, COLORS[:2], COLORS],
                   "highlight": "#810F7C", "text": INK,
                   "axis_label": INK, "tick_label": MUTED}, ai_chat=False)
    dot = view.dot
    if new_node is not None:
        dot = dot.replace(f'node{new_node} [margin="0" shape=none',
                          f'node{new_node} [margin="0" shape=box penwidth="2.5" color="#810F7C"')
    # The shared species legend sits above the scatter plot. Removing this
    # duplicate Graphviz legend gives the tree more room.
    dot = re.sub(r"^[ \t]*subgraph cluster_legend\s*\{.*?^[ \t]*\}", "", dot, flags=re.S | re.M)
    # On Windows Graphviz may lack an SVG rasterizer. Rasterize each dtreeviz
    # SVG node with MuPDF, then let Graphviz draw the genuine node/edge layout.
    for i, source in enumerate(re.findall(r'<img src="(.*?)"', dot)):
        svg = Path(source).read_text(encoding="utf-8")
        svg = svg.replace("#ffffff", BACKGROUND).replace("#FFFFFF", BACKGROUND)
        with pymupdf.open(stream=svg.encode(), filetype="svg") as document:
            pixels = document[0].get_pixmap(matrix=pymupdf.Matrix(3, 3), alpha=True)
            node_path = folder / f"node_{i}.png"
            pixels.save(str(node_path))
        dot = dot.replace(source, str(node_path))
    dot = dot.replace('penwidth=".5"', 'penwidth="2.5"')
    dot = dot.replace('style="dashed"', 'style="solid"')
    source_path, image_path = folder / "tree.dot", folder / "tree.png"
    source_path.write_text(dot, encoding="utf-8")
    subprocess.run([_tree_growth_graphviz(), "-Tpng", "-Gdpi=144",
                    f"-Gbgcolor={BACKGROUND}", str(source_path), "-o", str(image_path)],
                   check=True, capture_output=True)
    return Image.open(image_path).convert("RGB")


def _tree_growth_frame(model, full, x, y, count, diagram):
    """Pair the partial dtreeviz tree with the exact same model's regions."""
    from matplotlib.patches import Rectangle

    width, height = TREE_GROWTH_SIZE
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    count_total = len(TREE_GROWTH_ORDER)
    new_node = TREE_GROWTH_ORDER[count - 1] if count else None
    if count:
        feature = ["petal length", "petal width"][full.tree_.feature[new_node]]
        rule = f"{feature} ≤ {full.tree_.threshold[new_node]:.2f} cm"
        heading = f"Rule {count} of {count_total}   {rule}"
    else:
        heading = "Before the first split"
    fig.text(.04, .955, heading, fontsize=27, weight="bold", va="top")
    tree_box = [.15, .245, .23, .42] if count == 0 else [.015, .035, .50, .84]
    tree_ax = fig.add_axes(tree_box)
    tree_ax.imshow(diagram)
    tree_ax.set_axis_off()
    ax = fig.add_axes([.585, .16, .385, .64])
    rectangles = _tree_growth_rectangles(model)
    for node, (x0, x1, y0, y1) in rectangles.items():
        if model.tree_.children_left[node] != -1:
            continue
        predicted = int(model.tree_.value[node][0].argmax())
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                              facecolor=COLORS[predicted], alpha=.16, linewidth=0))
    for node, (x0, x1, y0, y1) in rectangles.items():
        if model.tree_.children_left[node] == -1:
            continue
        threshold = model.tree_.threshold[node]
        feature = model.tree_.feature[node]
        line_x, line_y = (([threshold, threshold], [y0, y1]) if feature == 0
                          else ([x0, x1], [threshold, threshold]))
        ax.plot(line_x, line_y, color="#810F7C" if node == new_node else INK,
                linewidth=3.4 if node == new_node else 1.1,
                alpha=1 if node == new_node else .6, zorder=3)
    for k, color in enumerate(COLORS):
        ax.scatter(*x[y == k].T, color=color, s=58, alpha=.68,
                   edgecolors="white", linewidths=.55, zorder=4)
    ax.set(xlim=(.6, 7.1), ylim=(0., 2.7),
           xlabel="Petal length (cm)", ylabel="Petal width (cm)")
    ax.tick_params(labelsize=16)
    ax.xaxis.label.set_size(19)
    ax.yaxis.label.set_size(19)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=c, marker="o", linestyle="", markersize=9,
                      label=n) for c, n in zip(COLORS, NAMES)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.045),
              ncol=3, frameon=False, fontsize=16, columnspacing=1.0)
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    assert pixels.shape == (height, width, 3)
    return pixels


def tree_growth_frames(cache_dir=None):
    """Return locally generated PNG paths and metadata for all eight stages.

    Each stage stops traversal at nodes whose rules have not been revealed.
    The final stage is the original fitted tree. No rows or axes move.
    """
    import tempfile
    import sklearn
    import dtreeviz

    folder = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent / ".tree_growth_cache"
    folder.mkdir(parents=True, exist_ok=True)
    cache_key = dict(version=TREE_GROWTH_VERSION, sklearn=sklearn.__version__,
                     dtreeviz=dtreeviz.__version__, size=TREE_GROWTH_SIZE)
    metadata_path = folder / "tree_growth.json"
    paths = [folder / f"step_{i:02d}.png" for i in range(8)]
    if metadata_path.exists() and all(p.exists() for p in paths):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") == json.loads(json.dumps(cache_key)):
            return paths, metadata
    x, y, full, models = _tree_growth_models()
    xx, yy = np.meshgrid(np.linspace(.6, 7.1, 401), np.linspace(0., 2.7, 301))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    results = []
    with matplotlib.rc_context(), plt.ioff(), tempfile.TemporaryDirectory(prefix="cme538_tree_") as temp:
        _style()
        for count, model in enumerate(models):
            active = set(TREE_GROWTH_ORDER[:count])
            # Independently traverse the full tree with a stopping rule and
            # compare every grid point with sklearn's partial-tree predictions.
            reached = np.zeros(len(grid), dtype=int)
            for _ in range(full.get_depth()):
                moving = np.isin(reached, list(active))
                rows = np.flatnonzero(moving)
                if not len(rows):
                    break
                nodes = reached[rows]
                go_left = grid[rows, full.tree_.feature[nodes]] <= full.tree_.threshold[nodes]
                reached[rows] = np.where(go_left, full.tree_.children_left[nodes],
                                        full.tree_.children_right[nodes])
            expected = full.tree_.value[reached, 0]
            expected = expected / expected.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(model.predict_proba(grid), expected)
            new_node = TREE_GROWTH_ORDER[count - 1] if count else None
            stage_folder = Path(temp) / str(count)
            stage_folder.mkdir()
            diagram = _tree_growth_diagram(model, x, y, new_node, stage_folder)
            Image.fromarray(_tree_growth_frame(model, full, x, y, count, diagram)).save(paths[count])
            results.append(dict(step=count, added_node=new_node,
                                active_nodes=TREE_GROWTH_ORDER[:count],
                                leaves=count+1,
                                training_errors=int(np.sum(model.predict(x) != y))))
        np.testing.assert_allclose(models[-1].predict_proba(grid), full.predict_proba(grid))
    metadata = dict(cache_key=cache_key, seed=0,
                    features=["petal length (cm)", "petal width (cm)"],
                    training_rows=150, order=TREE_GROWTH_ORDER, results=results,
                    full_tree_depth=full.get_depth(), grid_points_validated=len(grid),
                    parameters=full.get_params(),
                    note="Breadth-first reveal of one fitted tree, not the optimizer's fitting chronology.")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths, metadata


def tree_growth_widget(cache_dir=None):
    """Play or step through one dtreeviz tree and its matching decision regions."""
    import base64
    import ipywidgets as widgets

    paths, _ = tree_growth_frames(cache_dir)
    data = [p.read_bytes() for p in paths]

    class TreeGrowthDisplay(widgets.VBox):
        def _repr_mimebundle_(self, **kwargs):
            bundle = super()._repr_mimebundle_(**kwargs)
            bundle["image/png"] = base64.b64encode(data[0]).decode("ascii")
            return bundle

    image = widgets.Image(value=data[0], format="png",
                          layout=widgets.Layout(width="100%", height="auto"))
    slider = widgets.IntSlider(value=0, min=0, max=len(data)-1, step=1,
                               description="Rules", continuous_update=False,
                               layout=widgets.Layout(width="85%"))
    play = widgets.Play(value=0, min=0, max=len(data)-1, step=1, interval=2200,
                        repeat=False, description="Play")
    link = widgets.jslink((play, "value"), (slider, "value"))
    slider.observe(lambda change: setattr(image, "value", data[change["new"]]), names="value")
    panel = TreeGrowthDisplay([widgets.HBox([play, slider]), image],
                              layout=widgets.Layout(width="100%"))
    panel._tree_growth_link = link
    return panel


def make_tree_growth(output):
    """Write the growing petal tree movie using the same frames as the widget."""
    import imageio_ffmpeg

    paths, metadata = tree_growth_frames()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    video = output / "tree_growth.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(video), TREE_GROWTH_SIZE, fps=FPS, codec="libx264", quality=8,
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-movflags", "+faststart", "-profile:v", "high", "-level", "4.1"],
    )
    seconds = [2.5] + [3.0] * 6 + [5.0]
    writer.send(None)
    try:
        for path, hold in zip(paths, seconds):
            frame = np.asarray(Image.open(path).convert("RGB"))
            for _ in range(round(FPS * hold)):
                writer.send(frame)
    finally:
        writer.close()
    for suffix, index in [("poster", 0), ("middle", 4), ("final", 7)]:
        Image.open(paths[index]).save(output / f"tree_growth_{suffix}.png")
    metadata.update(width=TREE_GROWTH_SIZE[0], height=TREE_GROWTH_SIZE[1], fps=FPS,
                    duration_seconds=sum(seconds), codec="H.264", pixel_format="yuv420p",
                    style_reference="utils.py: configure_notebook; FiveThirtyEight palette")
    video.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "slides" / "animations")
    parser.add_argument("--only", choices=["complexity", "forest", "both", "tree_growth", "all"], default="both")
    args = parser.parse_args()
    if args.only in ["complexity", "both", "all"]:
        info = make_tree_complexity(args.output)
        print(f"Saved tree_complexity.mp4 ({info['duration_seconds']:.2f} seconds)")
    if args.only in ["forest", "both", "all"]:
        info = make_forest_aggregation(args.output)
        print(f"Saved forest_aggregation.mp4 ({info['duration_seconds']:.2f} seconds)")
    if args.only in ["tree_growth", "all"]:
        info = make_tree_growth(args.output)
        print(f"Saved tree_growth.mp4 ({info['duration_seconds']:.2f} seconds)")


if __name__ == "__main__":
    main()
