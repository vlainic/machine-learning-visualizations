import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10), plt_title=None):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
      ### my addition:
      plt_title: Plot title, can be: None, dict or string. If None, accuracy_score is given as title.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    
    ### added plot title:
    if plt_title is None: ### Automatic title is accuracy_score
        title_str = "Acc={:.2f}%".format(accuracy_score(y_true, y_pred)*100)
    elif type(plt_title) is dict: ### but you can make your own dictionary of metrics names and values, e.g. {'acc': 0.8, 'f1':0.75, etc.}
        title_str =  ""
        for kk in plt_title.keys():
            title_str = title_str + kk + "={:.2f}%, ".format(plt_title[kk]*100)
        title_str = title_str[:-2]
    elif type(plt_title) is str: ### or you can make your own string
        title_str = plt_title
    else: ### Error otherwise
        print("Title format suported are string and dict!")
        assert False
    plt.title(title_str)
    
    sns.set(font_scale=1.0) ### adaptive fontsize
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="YlGnBu") ### options: YlGnBu, jet, summer
    #plt.savefig(filename)
    plt.show()
