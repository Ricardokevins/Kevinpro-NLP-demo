#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2021/12/16 23:14:24
@Author      :Kevinpro
@version      :1.0
'''
# follow https://zhuanlan.zhihu.com/p/55815047

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

# a = torch.randn(4, 2)
# b = a.softmax(dim=1)
# c = a.softmax(dim=0).transpose(0, 1)
# d = b.matmul(c)
# d = d.numpy()

# variables = ['A', 'B', 'C', 'X']
# labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3']


def visualize(src_matrix, X_, Y_, save_path=None):   
    """
    @description  : Visualize Matrix with matplotlib
    ---------
    @param  :
        src_matrix: Matrix to Visualize
        X_: labelX
        Y_: labelY
        save_path: default None
    -------
    @Returns  : 
    -------
    """
    df = pd.DataFrame(src_matrix, columns=X_, index=Y_)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)
    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.set_xticklabels([''] + list(df.columns))
    ax.set_yticklabels([''] + list(df.index))

    if save_path != None:
        plt.savefig(save_path)
    plt.show()

    #ax.text(-1.88, 0.13, 'Ahmed',fontsize=20)
    #ax.text(4.85, -0.87, 'me',  fontsize=20,rotation=90,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

