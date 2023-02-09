import matplotlib.pyplot as plt
from cycler import cycler
import os
from matplotlib.pyplot import MultipleLocator
import numpy as np

def draw_day_action_num(fileName,fileName_1,model_Name, hasUserNum = False):
    #pred
    #y_pred [user_num,future_day,a_feat_size]
    y_pred = np.load(fileName)
    if (hasUserNum):
        # action_y :[future_day,a_feat_size]
        action_y=y_pred.sum(axis=0)
        user_num,future_day,a_feat_size = y_pred.shape
    else:
        future_day, a_feat_size = y_pred.shape
        action_y=y_pred
        user_num = 1
        if (model_Name == 'KDD'):
            action_y *= 2

    feature_idx_list = [i for i in range(a_feat_size)]
    if (a_feat_size > 10):
        random.shuffle(feature_idx_list)
        feature_idx_list = feature_idx_list[:7]
        feature_idx_list = [6, 23, 40, 42, 97, 101, 131]
        a_feat_size = len(feature_idx_list)
    print(feature_idx_list)
        
    # [15, 23, 40, 97, 101, 131]
    x=np.arange(1,future_day+1,1)
    rcColoar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    default_cycler = (cycler(color=rcColoar[:a_feat_size]) )
    plt.rc('axes', prop_cycle=default_cycler)
    label = 'feature_' 

    plt.figure(figsize=(30, 10), dpi=160)
    # plt.figure(figsize=(20, 10), dpi=160)
    plt.axes([0.16, 0.16, 0.75, 0.75])
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    for i in feature_idx_list:
        cur_action = action_y[:,i]
        cur_action_std = cur_action/user_num
        plt.plot(x, cur_action_std, marker='^', linewidth=2, markersize = 10, linestyle = 'solid', label=str(i))

    action_y_pre = action_y
    #true
    y_true = np.load(fileName_1)
    action_y = y_true
    if (hasUserNum):
        user_num, _, _ = y_true.shape
        action_y = action_y.sum(axis=0)
    else:
        user_num = 1

    for i in feature_idx_list:
        cur_action = action_y[:, i]
        cur_action_std = cur_action / user_num
        plt.plot(x, cur_action_std, marker='o', linewidth=2, markersize = 10, linestyle = 'dotted')
    
    # plt.title("Daily activity statistics of different characteristics")
    # plt.legend(loc='best')
    plt.legend(loc='upper right')
    plt.xticks(fontsize= 12)
    plt.yticks(fontsize= 12)
    plt.xlabel(xlabel='Days', fontsize= 25)
    plt.ylabel(ylabel='Active probability of behavior', fontsize= 25)
    plt.savefig('./Figure/'+model_Name+'.png')
    plt.show()

if __name__ == '__main__':
    draw_day_action_num('./Log/7_23/7_23_1011_False_Kwai_MSE_FLTADP_0.001_100_1e-05_pred_1.npy',
    './Log/7_23/7_23_1011_False_Kwai_MSE_FLTADP_0.001_100_1e-05__1.npy'
                        ,'Kwai', True)
    draw_day_action_num('./Log/7_23/7_23_1011_False_KDD_MSE_FLTADP_0.001_100_1e-05_pred_1.npy',
    './Log/7_23/7_23_1011_False_KDD_MSE_FLTADP_0.001_100_1e-05__1.npy'
                        ,'KDD', False)
    # draw_day_action_num('./Log/7_33_1011_False_Baidu_MSE_FLTADP_0.001_100_1e-05_pred_1.npy',
    # './Log/7_33_1011_False_Baidu_MSE_FLTADP_0.001_100_1e-05__1.npy'
    #                     ,'Baidu')