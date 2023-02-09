import os
import numpy as np

def statistical_results(path,head,filename,modelname):
    average_best_mae = 0
    average_rmse = 0
    average_df = 0
    file_names = []
    for i in range(1,6):
        cur_filename = head+str(i)+filename
        file_names.append(cur_filename)
    mae_list = []
    rmse_list = []
    df_list = []
    for file_name in file_names:
        with open(path + file_name, 'r') as f:
            temp_result = f.readlines()[-1].split()
            mae_list.append(float(temp_result[3]))
            rmse_list.append(float(temp_result[5]))
            df_list.append(float(temp_result[7]))
            average_best_mae += float(temp_result[3])
            average_rmse += float(temp_result[5])
            average_df += float(temp_result[7])

    mae_list =np.array(mae_list)
    rmse_list =np.array(rmse_list)
    file_numbers = len(file_names)
    average_best_mae=average_best_mae / file_numbers
    average_rmse = average_rmse / file_numbers
    average_df = average_df / file_numbers
    abs_mae = np.std(mae_list)
    abs_rmse = np.std(rmse_list)

    printStr=modelname +' & ' +'%.4f' % average_best_mae +' $\pm$ '+'%.4f' %abs_mae+\
             ' & '+'%.4f' % average_rmse +' $\pm$ '+'%.4f' %abs_rmse+ '\\\\'
    return printStr

if __name__ == '__main__':
    modelname = ['LR', 'MLP','CFIN','CLSA','DPCNN','LSCNN','RNN','FLTADP']
    modelname = ['LR', 'MLP','CFIN','CLSA','DPCNN','LSCNN','RNN','FLTADP']
    wd = ['1e-05','0.0001','1e-05','1e-05','0.001','1e-05','0.001','1e-05']
    wd = ['1e-05','0.0001','1e-05','1e-05','0.001','1e-05','0.001','1e-05']
    pd_fd = '7_23'
    filePath = './Log/' + pd_fd + '/'
    res = ''
    j=0
    for i in modelname:
        head = pd_fd + '_101'
        fileName = '_False_Kwai_MSE_'+i+'_0.001_100_'+wd[j]+'_log.txt'
        res +=statistical_results(path=filePath,head=head,filename=fileName,modelname=modelname[j])
        res+='\n'
        j=j+1
    print(res)