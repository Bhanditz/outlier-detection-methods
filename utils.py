import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

def print_confusion_matrix(true, pred):
    cm = confusion_matrix(true, pred)
    
    df = pd.DataFrame(cm, columns=['Inlier (True)', 'Outlier (True)'], 
                      index=['Inlier (Pred)', 'Outlier (Pred)'])
    
    print(df)

def plot_dbscan_results(dbscan_obj, df):
    labels = dbscan_obj.labels_
    core_sample_idx = dbscan_obj.core_sample_indices_
    components = dbscan_obj.components_
    
    # plot prices, colored by label
    fig, (top, bot) = plt.subplots(2, 1, figsize=(15,14))

    # plot styles
    _unique_labels, _label_counts = np.unique(labels, return_counts=True)
    _pal = sns.husl_palette(len(_unique_labels), h=.5)
    # colormap, with outliers set to grey
    _cmap = dict(zip(_unique_labels, _pal))
    _cmap[-1] = 'grey'

    # x values are weeks
    x = list(range(df.shape[1]))
    
    # top: all prices
    # labels, regions
    for lab, region in zip(labels, df.index):
        price = df.loc[region].values

        _alpha = 0.5
        
        if lab == -1:
            # plot dotted lines for outliers on top and bottom
            top.plot(x, price, color=_cmap[lab], ls=':', alpha=0.3, zorder=1)
            bot.plot(x, price, color=_cmap[lab], ls=':', alpha=0.3, zorder=1)
        else:
            # plot prices on top
            top.plot(x, price, color=_cmap[lab], ls='-', alpha=_alpha, zorder=2)
        
    # bottom: summary plot
    # indices of core samples
    for idx in core_sample_idx:
        # core component from data
        # don't use DBSCAN.components_, b/c this doesn't use raw data?...
        comp = df.iloc[idx]
        
        # get the label for the core component for coloring purposes
        lab = labels[idx]
        _clr = _cmap[lab]
        
        # plot core component on bottom
        bot.plot(x, comp, color=_clr)
        
    # bottom: ranges for components
    # unique labels
    for lab in _unique_labels:
        # get data that was matched with the label
        df_label_mask = labels == lab
        df_subset = df.loc[df_label_mask]
        
        # get min/max values
        # Note: this is a vector
        _ymin = df_subset.min()
        _ymax = df_subset.max()
        
        # color of label
        _clr = _cmap[lab]
        
        # plot fill-between of min/max on bottom
        # don't plot outliers
        if lab != -1:
            bot.fill_between(x, _ymin, _ymax, color=_clr, alpha=0.3)
        
    top.set_title('DBSCAN Results for Avocado Prices')
    bot.set_title('DBSCAN Components for Avocado Prices')
    bot.set_xlabel('Week'); 
    top.set_ylabel('Average Price');
    bot.set_ylabel('Average Price');