import os

def statistical_results(path,head,filename,modelname):
    average_best_mae = 0
    average_rmse = 0
    average_df = 0
    file_names = []
    for i in range(1,6):
        cur_filename = head+str(i)+filename
        file_names.append(cur_filename)

    for file_name in file_names:
        with open(path + file_name, 'r') as f:
            temp_result = f.readlines()[-1].split()
            average_best_mae += float(temp_result[3])
            average_rmse += float(temp_result[5])
            average_df += float(temp_result[7])

    file_numbers = len(file_names)
    average_best_mae=average_best_mae / file_numbers
    average_rmse = average_rmse / file_numbers
    average_df = average_df / file_numbers
    abs_mae = 0
    abs_rmse = 0
    abs_df = 0
    for file_name in file_names:
        with open(path + file_name, 'r') as f:
            temp_result = f.readlines()[-1].split()
            abs_mae = max(abs_mae,abs(average_best_mae-float(temp_result[3])))
            abs_rmse = max(abs_rmse,abs(average_rmse-float(temp_result[5])))
            abs_df = max(abs_df,abs(average_df-float(temp_result[7])))

    printStr=modelname +' & ' +'%.4f' % average_best_mae +' $\pm$ '+'%.4f' %abs_mae+\
             ' & '+'%.4f' % average_rmse +' $\pm$ '+'%.4f' %abs_rmse+ \
             ' & '+'%.4f' % average_df +' $\pm$ '+'%.4f' %abs_df+ '\\\\'
    return printStr

if __name__ == '__main__':
    modelname = ['LR','MLP','CFIN','CLSA','DPCNN','LSCNN','RNN','FLTADP']
    wd = ['1e-05','0.0001','1e-05','1e-05','0.001','1e-05','0.001','1e-05']
    filePath = './Log/'
    res = ''
    j=0
    for i in modelname:
        head = '7_23_101'
        fileName = '_False_KDD_MSE_'+i+'_0.001_100_'+wd[j]+'_log.txt'
        res +=statistical_results(path=filePath,head=head,filename=fileName,modelname=modelname[j])
        res+='\n'
        j=j+1
    print(res)