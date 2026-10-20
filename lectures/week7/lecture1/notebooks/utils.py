
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

def _shot_make_rates(distance, outcome):
    """Summarize recorded outcomes in fixed five-foot bins for display."""
    shots = pd.DataFrame({'distance': np.asarray(distance), 'made': np.asarray(outcome)})
    shots['bin'] = np.floor(shots['distance'] / 5).astype(int)
    return shots.groupby('bin', sort=True).agg(
        distance=('distance', 'mean'), rate=('made', 'mean'), count=('made', 'size'))


def _plot_make_rates(ax, rates):
    ax.scatter(rates.distance, rates.rate, marker='D', s=48, color=GREEN,
               edgecolor='black', linewidth=0.6, zorder=6, label='Observed make rate (5-ft groups)')
    for row in rates.itertuples():
        ax.annotate(f'n={row.count}', (row.distance, row.rate),
                    xytext=(0, -15 if row.rate > 0.85 else 8), textcoords='offset points',
                    ha='center', fontsize=8, color='#303030', zorder=7,
                    bbox=dict(facecolor=ax.get_facecolor(), edgecolor='none', pad=0.5, alpha=0.9))


def shot_plot(distance, outcome, x_grid=None, curves=None, title='LeBron’s shot outcomes',
              probability=True, observed_rates=False):
    rng=np.random.default_rng(538)
    fig,ax=plt.subplots()
    jitter = np.zeros(len(outcome)) if observed_rates else rng.uniform(-0.04,0.04,len(outcome))
    ax.scatter(distance, np.asarray(outcome)+jitter,
               alpha=0.22 if observed_rates else 0.38, s=25, edgecolors='none',
               color=GRAY if observed_rates else BLUE, label='Recorded shot (0 or 1)')
    for label, values in (curves or {}).items():
        ax.plot(x_grid, values, label=label)
    if probability: ax.set_ylim(-0.09,1.09)
    ax.set_yticks([0,0.5,1])
    ax.axhline(0,color=GRAY,lw=0.7);ax.axhline(1,color=GRAY,lw=0.7)
    if observed_rates:
        _plot_make_rates(ax, _shot_make_rates(distance, outcome))
        ax.set_xlim(min(distance)-0.8, max(distance)+0.8)
        ax.legend(loc='upper left', bbox_to_anchor=(0, -0.17), ncol=2, fontsize=9)
    elif curves:
        ax.legend()
    finish(ax,title,'Shot distance (ft)','Outcome / predicted probability')

def sigmoid_parameters(x):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for slope in [-1,0.3,1,3]:
        axes[0].plot(x,1/(1+np.exp(-slope*x)),label=f'Slope = {slope}')
    for intercept in [-4,0,4]:
        axes[1].plot(x,1/(1+np.exp(-(intercept+x))),label=f'Intercept = {intercept}')
    for ax,title in zip(axes,['Slope changes direction and steepness','Intercept shifts the curve']):
        ax.set(title=title,xlabel='Feature value, x',ylabel='Predicted probability');ax.legend()
    fig.tight_layout();plt.show()

def loss_curves(p, loss_positive, loss_negative, mse_positive):
    fig,ax=plt.subplots()
    ax.plot(p,loss_positive,label='Cross-entropy: actual class 1')
    ax.plot(p,loss_negative,label='Cross-entropy: actual class 0')
    ax.plot(p,mse_positive,ls='--',label='Squared error: actual class 1')
    ax.legend();finish(ax,'Confident errors receive large log loss','Predicted probability of class 1','Loss per observation')

def loss_landscape(slopes,mse,log_loss):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,values,title in zip(axes,[mse,log_loss],['Squared error after a sigmoid','Cross-entropy after a sigmoid']):
        ax.plot(slopes,values)
        ax.set(title=title,xlabel='Slope parameter',ylabel='Mean loss')
    fig.tight_layout();plt.show()

def training_history(history):
    fig,ax=plt.subplots()
    ax.plot(np.arange(len(history)),history)
    finish(ax,'Gradient descent reduces the training loss','Parameter update','Mean cross-entropy loss')

def nba_probability(games,grid,p, label='Fitted probability', show_threshold=True):
    fig,ax=plt.subplots()
    rng=np.random.default_rng(538)
    ax.scatter(games.FG_PCT_DIFF,games.WON+rng.uniform(-.035,.035,len(games)),alpha=.22,s=18,edgecolors='none')
    ax.plot(grid,p,color=RED,label=label)
    if show_threshold:
        ax.axhline(.5,color=GRAY,ls='--',lw=1,label='Decision threshold = 0.5')
    ax.legend();ax.set_ylim(-.08,1.08)
    finish(ax,'Shooting difference and game outcome','Field-goal proportion difference','Outcome / probability of a win')


def _logistic_animation_data(feature, outcome, theta_history, learning_rate, x_mean, x_scale,
                             *, xlabel='Feature value', outcome_label='Recorded outcome (0 or 1)',
                             curve_title='Current probability curve'):
    """Evaluate recorded notebook states for display; do not refit the model."""
    feature = np.asarray(feature, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    path = np.asarray(theta_history, dtype=float)
    if (feature.ndim != 1 or outcome.shape != feature.shape or len(feature) == 0
            or path.ndim != 2 or path.shape[1] != 2 or len(path) < 2):
        raise ValueError('Provide one feature value and label per observation, and a two-column parameter history.')
    if (not all(np.isfinite(a).all() for a in [feature, outcome, path])
            or not np.isin(outcome, [0, 1]).all() or not np.isfinite(x_mean)
            or not np.isfinite(x_scale) or x_scale <= 0
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Use finite inputs, binary labels, and positive scale and learning rate.')
    design = np.column_stack([np.ones(len(outcome)), (feature - x_mean) / x_scale])
    scores = design @ path.T
    probabilities = np.exp(-np.logaddexp(0, -scores))
    losses = (np.logaddexp(0, scores) - outcome[:, None] * scores).mean(axis=0)
    gradients = ((probabilities - outcome[:, None]).T @ design) / len(outcome)
    if not np.allclose(path[1:], path[:-1] - learning_rate * gradients[:-1], atol=1e-10, rtol=1e-9):
        raise ValueError('The recorded updates do not match this data, scale and learning rate.')

    # Compute a fixed loss contour grid once, rather than on each animation frame.
    lo = np.minimum(path.min(axis=0), [-0.5, -0.15])
    hi = np.maximum(path.max(axis=0), [0.5, 0.15])
    margin = (hi - lo) * 0.12
    intercepts = np.linspace(lo[0] - margin[0], hi[0] + margin[0], 65)
    slopes = np.linspace(lo[1] - margin[1], hi[1] + margin[1], 65)
    a, b = np.meshgrid(intercepts, slopes)
    surface_scores = design[:, 1, None] * b.ravel() + a.ravel()
    surface = (np.logaddexp(0, surface_scores) - outcome[:, None] * surface_scores).mean(axis=0)
    return dict(feature=feature, outcome=outcome, path=path, losses=losses,
                gradients=gradients, learning_rate=learning_rate, x_mean=x_mean,
                x_scale=x_scale, a=a, b=b, surface=surface.reshape(a.shape),
                xlabel=xlabel, outcome_label=outcome_label, curve_title=curve_title)


def logistic_gradient_figure(trace, step=0):
    """Draw one recorded parameter state and its next gradient update."""
    if int(step) != step or not 0 <= step < len(trace['path']):
        raise ValueError('step must identify a recorded parameter state.')
    step = int(step)
    path = trace['path']
    current = path[step]
    has_next = step + 1 < len(path)
    fig = plt.figure(figsize=(14, 4.3), dpi=110)
    curve_ax = fig.add_axes([0.055, 0.27, 0.31, 0.58])
    contour_ax = fig.add_axes([0.435, 0.19, 0.215, 0.66])
    color_ax = fig.add_axes([0.666, 0.19, 0.012, 0.66])
    numbers = fig.add_axes([0.755, 0.04, 0.24, 0.92])
    numbers.set_axis_off()

    grid = np.linspace(trace['feature'].min(), trace['feature'].max(), 300)
    scaled_grid = (grid - trace['x_mean']) / trace['x_scale']
    probability = lambda theta: np.exp(-np.logaddexp(0, -(theta[0] + theta[1] * scaled_grid)))
    curve_ax.scatter(trace['feature'], trace['outcome'], s=21, color=BLUE,
                     alpha=0.2, edgecolors='none', label=trace['outcome_label'])
    curve_ax.plot(grid, probability(current), color=BLUE, lw=1.3, label='Current probability')
    if has_next:
        curve_ax.plot(grid, probability(path[step + 1]), color=RED, lw=1.1,
                      ls='--', label='After next update')
    margin = (grid.max() - grid.min()) * 0.03
    curve_ax.set(xlim=(grid.min()-margin, grid.max()+margin), ylim=(-0.08, 1.08), yticks=[0, 0.5, 1],
                 xlabel=trace['xlabel'], ylabel='Outcome / probability')
    curve_ax.set_title(trace['curve_title'], fontsize=12)
    curve_ax.legend(loc='upper left', bbox_to_anchor=(0, -0.25), fontsize=7.5,
                    ncol=2, borderaxespad=0, framealpha=0.9, columnspacing=0.8)

    # Include more contours near the minimum so the final updates stay visible.
    minimum, maximum = trace['surface'].min(), trace['surface'].max()
    levels = minimum + (maximum - minimum) * np.linspace(0, 1, 14)**2
    contour = contour_ax.contour(trace['a'], trace['b'], trace['surface'],
                                 levels=levels, cmap='viridis', linewidths=0.8)
    bar = fig.colorbar(contour, cax=color_ax)
    bar.set_label('Mean cross-entropy loss', fontsize=9)
    bar.ax.tick_params(labelsize=8)
    contour_ax.plot(path[:step + 1, 0], path[:step + 1, 1], color=RED, lw=1.15)
    contour_ax.scatter(*path[0], marker='s', s=28, color=GRAY, label='Start', zorder=4)
    contour_ax.scatter(*current, marker='*', s=100, color=GOLD, edgecolor='black',
                       linewidth=0.6, label='Current', zorder=6)
    if has_next:
        contour_ax.annotate('', xy=path[step + 1], xytext=current,
                            arrowprops=dict(arrowstyle='->', color=RED, lw=1.3), zorder=7)
    contour_ax.set(xlim=(trace['a'].min(), trace['a'].max()),
                   ylim=(trace['b'].min(), trace['b'].max()),
                   xlabel=r'Intercept $\theta_0$', ylabel=r'Slope $\theta_1$')
    contour_ax.set_title('Follow the gradient updates', fontsize=14)
    contour_ax.legend(loc='upper right', fontsize=8.5)

    gradient = trace['gradients'][step]
    rate = trace['learning_rate']
    def shown(value, places=4):
        rounded = round(float(value), places)
        return f'{0.0 if rounded == 0 else rounded:.{places}f}'
    heading = f'Update {step} → {step + 1}' if has_next else f'After {step} updates'
    numbers.text(0, 0.94, heading, fontsize=14, fontweight='bold')
    numbers.text(0, 0.81, r'$g = X^T(p-y)/n$', fontsize=13)
    numbers.text(0, 0.70, f'g₀ = {shown(gradient[0], 5)}    g₁ = {shown(gradient[1], 5)}', fontsize=10)
    if has_next:
        for index, vertical in [(0, 0.56), (1, 0.34)]:
            numbers.text(0, vertical,
                         rf'$\theta_{{{index},\mathrm{{next}}}} = {shown(current[index])} - {rate:g}({shown(gradient[index])})$',
                         fontsize=10.5)
            numbers.text(0, vertical - 0.09, f'                  = {shown(path[step + 1, index])}', fontsize=11)
        numbers.text(0, 0.13, 'Mean cross-entropy loss', fontsize=10)
        numbers.text(0, 0.04, f"{trace['losses'][step]:.6f} → {trace['losses'][step + 1]:.6f}", fontsize=12)
    else:
        numbers.text(0, 0.54, f'θ₀ = {shown(current[0])}\n\nθ₁ = {shown(current[1])}', fontsize=12)
        numbers.text(0, 0.23, f"Mean loss = {trace['losses'][step]:.6f}", fontsize=11)
        numbers.text(0, 0.08, 'All recorded updates are shown.', fontsize=10)
    return fig


def logistic_gradient_widget(feature, outcome, theta_history, learning_rate, x_mean, x_scale,
                             *, xlabel='Feature value', outcome_label='Recorded outcome (0 or 1)',
                             curve_title='Current probability curve'):
    """Animate the notebook's recorded updates with responsive saved output."""
    import base64
    from io import BytesIO
    import ipywidgets as widgets
    from IPython.display import display

    trace = _logistic_animation_data(feature, outcome, theta_history, learning_rate, x_mean, x_scale,
                                     xlabel=xlabel, outcome_label=outcome_label, curve_title=curve_title)
    last = len(trace['path']) - 1
    # Show each early update; skip increasingly quiet intervals later in playback.
    frames = np.unique(np.r_[np.arange(min(50, last) + 1),
                            np.arange(55, min(150, last) + 1, 5),
                            np.arange(200, last + 1, 50), last]).astype(int)
    play = widgets.Play(value=0, min=0, max=len(frames) - 1, interval=350,
                        repeat=False, description='Play')
    slider = widgets.IntSlider(value=0, min=0, max=last, description='Updates:',
                               continuous_update=False, layout=widgets.Layout(flex='1 1 300px', min_width='0'))
    reset = widgets.Button(description='Reset', layout=widgets.Layout(width='65px'))
    final = widgets.Button(description='Final fit', layout=widgets.Layout(width='80px'))
    picture = widgets.Image(format='png', width='100%', layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', height='auto',
        flex='0 1 auto', object_fit='contain', display='block'))
    syncing = False

    def render():
        figure = logistic_gradient_figure(trace, slider.value)
        buffer = BytesIO()
        figure.savefig(buffer, format='png', bbox_inches=None)
        picture.value = buffer.getvalue()
        plt.close(figure)

    def move_slider(change):
        nonlocal syncing
        if syncing:
            return
        syncing = True
        try:
            slider.value = int(frames[change['new']])
        finally:
            syncing = False

    def update_frame(change):
        nonlocal syncing
        if not syncing:
            syncing = True
            try:
                play.value = int(np.searchsorted(frames, change['new'], side='right') - 1)
            finally:
                syncing = False
        render()

    def jump(value):
        play.playing = False
        slider.value = value

    play.observe(move_slider, names='value')
    slider.observe(update_frame, names='value')
    reset.on_click(lambda button: jump(0))
    final.on_click(lambda button: jump(last))
    render()
    controls = widgets.HBox([play, slider, reset, final], layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', flex_flow='row wrap', align_items='center'))
    widget = widgets.VBox([controls, picture], layout=widgets.Layout(
        width='100%', max_width='100%', min_width='0', height='auto', overflow='hidden'))
    bundle = widget._repr_mimebundle_()
    bundle['image/png'] = base64.b64encode(picture.value).decode('ascii')
    display(bundle, raw=True)
    return widget
