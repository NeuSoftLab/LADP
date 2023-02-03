from DataLoader.LoadData import getDataLoader
import torch
from Model.CFIN import CFIN
from Model.MLP import MLP
from Model.RNN import RNN
from Model.CLSA import CLSA
from Model.DPCNN import DPCNN
from Model.LSCNN import LSCNN
#from Model.MLADP import MLADP
from Model.LR import LR
import argparse
from run import run
import json
from data_pre_process.KDD_pre_process import create_file_by_data as kdd_create_file
from data_pre_process.kwai_pre_process import create_file_by_data as kwai_create_file
from data_pre_process.KDD_pre_time import create_time_file as create_kdd_time_file
from data_pre_process.kwai_pre_time import create_time_file as create_kwai_time_file
model_dict = {
    "CFIN": CFIN,
    "CLSA": CLSA,
    "DPCNN": DPCNN,
    "MLP": MLP,
    "RNN": RNN,
    "LSCNN": LSCNN,
    "LR": LR,
    #"MLADP":MLADP
}

def load_model_param(config_file):
    f = open(config_file, "r")
    model_param = json.load(f)
    return model_param

# test connect
def main():
    # Namespace of Hyper-parameter
    parser = argparse.ArgumentParser()
    # training process
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size e.g. 32 64')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate e.g. 0.001 0.01')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=100, help='max_iter e.g. 100 200 ...')
    # dataset
    parser.add_argument('--DataSet', type=str, default='KDD')
    parser.add_argument('--day', type=int, default=3)
    parser.add_argument('--future_day', type=int, default=23)
    parser.add_argument('--seq_length', type=int, default=7)
    parser.add_argument('--is_pre_process', type=bool, default=False)
    parser.add_argument('--miniData', type=bool, default=False)
    parser.add_argument('--data_dilution_ratio', type=float, default=1.0)
    # loss
    parser.add_argument('--LossFun', type=str, default='MSE')
    # gpu
    parser.add_argument('--cuda', type=int, default=0)
    # Model
    parser.add_argument('--model_name', type=str, default='CFIN')
    # bce_weight
    parser.add_argument('--bce_weight', type=float, default=1)
    # temporary hyperparameters
    parser.add_argument('--multi_task_enable', type=int, default=1)
    parser.add_argument('--imbalance_stratage_enable', type=int, default=0)
    parser.add_argument('--fine_grained', type=int, default=1)
    params = parser.parse_args()

    # pre_process
    if params.is_pre_process:
        if params.DataSet == 'KDD':
            kdd_create_file(params.day, params.future_day, params.data_dilution_ratio)
            create_kdd_time_file(params.day,params.future_day)
        elif params.DataSet == 'Kwai':
            kwai_create_file(params.day, params.future_day, params.data_dilution_ratio)
            create_kwai_time_file(params.day,params.future_day)
    # file open
    #log
    fileName="./Log/" +str(params.day)+'_'+str(params.future_day)+'_'+ str(params.multi_task_enable)+ str(params.imbalance_stratage_enable)\
             +str(params.fine_grained)+str(params.seed) + "_" + str(params.miniData)\
             + "_" + params.DataSet + "_" + params.LossFun + "_" + str(params.model_name) + "_" + str(
        params.learning_rate) + "_" + str(params.max_iter) + "_" + str(params.weight_decay) + "_"
    f = open(fileName + "log.txt", "w")
    # GPU settings
    device = torch.device("cuda:" + str(params.cuda) if torch.cuda.is_available() else "cpu")
    # The hyper-parameter of model
    param = vars(params)
    param['device'] = device
    # Obain Dataset
    """
    train_set{
        ui: [user_num, user_feature]
        uv: [user_num, user_feature]
        ai: [user_num, day, action_feature]
        av: [user_num, day, action_feature]
        time : [user_num,(past_day+future_day),4(past_day+future_day,4)]
        y:  [user_num,truth(Proportion of active days),total_activity_day,day1_1...day1_m(Number of active behaviors on the day),```,dayN_m]
    }
    """

    if params.DataSet == 'Kwai':
        # day_numpy [future_day]
        # len(day_numpy) train_enrollment_id_num
        train_set, valid_set, test_set, day_numpy, param = getDataLoader(params.batch_size, param,data_name='Kwai',data_path='./Kwai')
    elif params.DataSet == 'KDD':
        train_set, valid_set, test_set, day_numpy, param = getDataLoader(params.batch_size, param,data_name='KDD',data_path='./KDD')
    elif params.DataSet == 'Baidu':
        train_set, valid_set, test_set, day_numpy, param = getDataLoader(params.batch_size, param,data_name='Baidu',data_path='./Baidu')
    param['day_numpy_train'] = day_numpy

    # Create Model
    model_name = params.model_name
    model_param = load_model_param("./config/" + model_name + ".json")
    model_param.update(param)
    print(model_param)
    model = model_dict[model_name](model_param)
    model.to(device)
    #print(model_param)
    f.write(str(model_param) + "\n")

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # best result
    best_valid_auc = 0
    best_valid_rmse = 1e9
    best_valid_df = 1e9
    best_auc = 0
    best_valid_MAE = 0
    best_epoch = 0
    rmse = 0
    df = 0
    best_MAE = 0

    # run model
    for i in range(params.max_iter):
        model.train()
        run(i, train_set, model, optimizer, device, model_name, run_type="train", lossFun=params.LossFun, write_file=f,model_param=model_param,fileName=fileName)
        model.eval()
        valid_auc, valid_rmse, valid_df,valid_MAE = run(-1, valid_set, model, optimizer, device, model_name, run_type="valid",
                                              write_file=f,model_param=model_param,fileName=fileName)
        if (params.LossFun == 'BCE' and valid_auc > best_valid_auc) or (
                params.LossFun == 'MSE' and valid_rmse < best_valid_rmse):
        #if valid_df < best_valid_df:
            model.eval()
            best_valid_auc, best_valid_MAE,best_valid_rmse, best_valid_df, best_epoch = valid_auc, valid_MAE,valid_rmse, valid_df, i
            test_auc, test_rmse, test_df,test_MAE = run(-1, test_set, model, optimizer, device, model_name, run_type="test",
                                               write_file=f,model_param=model_param,fileName=fileName)
            best_auc, rmse, df,best_MAE = test_auc, test_rmse, test_df,test_MAE

    print_str = 'best_epoch:  %.4f\nbest_valid_auc %.4f best_valid_MAE %.4f best_valid_rmse %.4f best_valid_df %.4f \nbest_auc %.4f best_MAE %.4f rmse %.4f df %.4f' % (
        best_epoch, best_valid_auc, best_valid_MAE,best_valid_rmse, best_valid_df, best_auc, best_MAE, rmse, df)
    print(print_str)
    f.write(print_str + "\n")
    f.close()
if __name__ == '__main__':
    main()