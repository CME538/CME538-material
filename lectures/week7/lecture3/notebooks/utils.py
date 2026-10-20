
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix

def split_plot(splits,n,title):
    fig,ax=plt.subplots(figsize=(10,4.5))
    validation_rows=np.concatenate([val for train,val in splits])
    complete_rotation=len(validation_rows)==n and len(np.unique(validation_rows))==n
    positions=np.arange(n)
    if complete_rotation: positions[validation_rows]=np.arange(n)
    for fold,(train,val) in enumerate(splits,1):
        ax.scatter(positions[train],np.full(len(train),fold),s=36,marker='s',color=BLUE,label='Training' if fold==1 else None)
        ax.scatter(positions[val],np.full(len(val),fold),s=36,marker='s',color=GOLD,label='Validation' if fold==1 else None)
    ax.set_yticks(range(1,len(splits)+1));ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2)
    xlabel='Rows grouped by validation fold (display order only)' if complete_rotation else 'Observation order'
    finish(ax,title,xlabel,'Fold')

def cv_scores(scores):
    fig,ax=plt.subplots()
    ax.plot(np.arange(1,len(scores)+1),scores,'o-',markersize=8)
    ax.axhline(np.mean(scores),ls='--',color=GRAY,label=f'Mean = {np.mean(scores):.3f}')
    ax.set_xticks(range(1,len(scores)+1));ax.set_ylim(0,1);ax.legend()
    finish(ax,'Each fold gives a separate validation score','Validation fold','Accuracy')

def spatial_folds(x,y,groups):
    fig,ax=plt.subplots(figsize=(8,5))
    for group in np.unique(groups):
        m=groups==group;ax.scatter(x[m],y[m],s=30,alpha=.55,label=f'Region {group+1}')
    ax.legend();finish(ax,'Keep a whole region for validation','East–west coordinate (km)','North–south coordinate (km)')

def confusion(y, pred, title):
    counts = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    display = ConfusionMatrixDisplay(
        counts, display_labels=['Other result', '20+ point win']
    )
    display.plot(
        ax=ax, cmap='Blues', colorbar=False, values_format='d',
        im_kw={'vmin': 0, 'vmax': max(1, counts.max())},
    )
    colorbar = fig.colorbar(display.im_, ax=ax, pad=0.04, fraction=0.05)
    colorbar.set_label('Number of observations')
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(title=title, xlabel='Predicted class', ylabel='Actual class')
    ax.grid(False)
    fig.tight_layout()
    plt.show()

def prevalence_counts(counts):
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(counts.Population,counts['True alerts'],color=GREEN,label='True alerts')
    ax.bar(counts.Population,counts['False alerts'],bottom=counts['True alerts'],color=RED,label='False alerts')
    for i,row in counts.iterrows():
        ax.text(i,row['True alerts']+row['False alerts']+6,f'Precision {row.Precision:.1%}',ha='center',fontsize=11)
    ax.set_ylim(0,(counts['True alerts']+counts['False alerts']).max()*1.2);ax.legend()
    finish(ax,'The same error rates produce different alert lists','Population prevalence','Expected alerts per 1,000 observations')

def ranked_queue(queue):
    fig,ax=plt.subplots()
    ax.plot(queue['Rows reviewed'],queue.Precision,'o-',label='Precision among reviewed rows')
    ax.plot(queue['Rows reviewed'],queue.Recall,'s-',label='Recall of positive rows')
    ax.set_ylim(0,1.05);ax.legend()
    finish(ax,'Review capacity changes which mistakes matter','Highest-scoring rows reviewed','Fraction')
