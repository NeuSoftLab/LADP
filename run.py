import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import mean_squared_error

esp = 1e-5
#torch.set_printoptions(threshold=np.inf)

# Result evaluation
def calEvalResult(loss_value, y_preds, y_trues, test_type, write_file=None,isMask=False):
    y_preds_bool = np.copy(y_preds)
    y_preds_bool[y_preds >= 0.5] = 1.0
    y_preds_bool[y_preds < 0.5] = 0.0
    if isMask:
        y_trues=np.append(y_trues,0)
        y_preds=np.append(y_preds,0)
        y_trues = np.append(y_trues, 1)
        y_preds = np.append(y_preds, 1)
    eps = 1e-5
    y_trues_bool = np.copy(y_trues)
    y_trues_bool[y_trues >= eps] = 1.0
    y_trues_bool[y_trues < eps] = 0.0

    # rmse
    error = y_trues - y_preds
    rmse = (error ** 2).mean() ** 0.5
    # MAE
    MAE = np.absolute(error).mean()
    # df
    df = abs(y_trues.mean() - y_preds.mean()) / y_trues.mean()
    print_str = '%20s loss %3.6f  MAE  %.4f rmse  %.4f  df(ActivateDay.Avg) %.4f' % (
        test_type, loss_value, MAE,rmse, df)

    if not isMask:
        print(print_str)
        if (write_file != None):
            write_file.write(print_str + "\n")
    return rmse, df ,MAE

def calEvalResult_FLTADP(loss_value, predict_results, test_type, write_file=None,isMask=False):
    # y_preds_1, y_trues_1 whether active every day (category) : [data_num, label_num]
    # y_preds_2, y_trues_2 active days percentage (category) : [data_num,] \in [0,1]
    y_preds_1, y_trues_1, y_preds_2, y_trues_2 = predict_results 

    # rmse
    error = (y_trues_2 - y_preds_2)
    rmse = (error ** 2).mean() ** 0.5
    # MAE
    MAE = np.absolute(error).mean()
    # df: average number of active days in the future
    df = abs(y_preds_2.mean() - y_trues_2.mean()) / y_trues_2.mean()
    print_str = '%20s loss %6.6f MAE  %.4f  rmse  %.4f  df(ActivateDay.Avg) %.4f' % (
        test_type, loss_value, MAE, rmse, df)
    if not isMask:
        print(print_str)
        if (write_file != None):
            write_file.write(print_str + "\n")
    return rmse, df, MAE


def run(epoch, datas, model, optimizer, device, model_name="None", run_type="train", lossFun='BCE', write_file=None,model_param=None,fileName=None):
    y_trues = np.array([])
    y_preds = np.array([])
    y_trues_2 = np.array([])
    y_preds_2 = np.array([])
    y_1_inputs = np.array([])
    #mask
    y_trues_filtered = []
    y_preds_filtered = []
    y_trues_2_filtered = []
    y_preds_2_filtered = []

    training_features = None
    all_loss = 0

    if (run_type == "train"):
        model.train()
    else:
        model.eval()
    data_id = 0
    datas_user_num = 0
    for idx, data in enumerate(datas):
        datas_user_num += len(data[0])
        ui, uv, ai, av, y, time = data
        ui = ui.to(device)
        uv = uv.to(device)
        ai = ai.to(device)
        av = av.to(device)
        y = y.to(device)
        if (run_type == "train"):
            av_uv = torch.cat((av.reshape(-1, av.shape[1] * av.shape[2]), uv.reshape(-1, uv.shape[1])), dim=1)
            # training_feature [data_num, day * a_field_dim + u_field_dim]
            if (training_features == None):
                training_features = av_uv
            else:
                training_features = torch.cat((training_features, av_uv), dim=0)
            optimizer.zero_grad()
        if (model_name != "FLTADP" and model_name!="MLADP"):
            # y:(Proportion of active days):batch_size * 1
            y = y[:, 0].reshape(-1, 1)
            loss, y_pred ,filtered_y,filtered_pred_y= model.forward(ui, uv, ai, av, y, lossFun)
            if len(y_trues_2_filtered) == 0:
                for i in range(len(filtered_y)):
                        y_trues_2_filtered.append(filtered_y[i].to("cpu").detach().numpy())
                        y_preds_2_filtered.append(filtered_pred_y[i].to("cpu").detach().numpy())
            else:
                for i in range(len(filtered_y)):
                    y_trues_2_filtered[i] = np.concatenate((y_trues_2_filtered[i],filtered_y[i].to("cpu").detach().numpy()),axis=0)
                    y_preds_2_filtered[i] = np.concatenate((y_preds_2_filtered[i],filtered_pred_y[i].to("cpu").detach().numpy()),axis=0)

            # y_trues is the list of proportion of active days
            y_trues = np.concatenate((y_trues, y.detach().cpu().numpy().reshape(-1)), axis=0)
            # y_pred is the activate or not in future days
            y_preds = np.concatenate((y_preds, y_pred.reshape(-1).detach().cpu().numpy()), axis=0)
        else:
            y_1 = y[:, 2:].detach().to(device)
            day = model_param['day']
            future_day = model_param['future_day']
            batch_size = model_param['batch_size']
            y_1 = y_1.reshape(batch_size, day + future_day, -1)
            y_1 = y_1[:,day:,:]
            y_1_input = y_1.clone()
            one = torch.ones_like(y_1_input)
            zero = torch.zeros_like(y_1_input)
            y_1_input = torch.where(y_1_input == 0, zero, one)
            y_1 = y_1.sum(dim=2)
            one = torch.ones_like(y_1)
            zero = torch.zeros_like(y_1)
            y_1 = torch.where(y_1==0,zero,one)
            y_2 = y[:, 1].detach().long().to(device)
            y_2_input = F.one_hot(y_2, num_classes=model.future_day + 1).float()
            if model_param['multi_task_enable']:
                y_2 = y_2 / (model.future_day + 1)
            else:
                y_2 = y[:, 0]
            time = time.to(device)
            loss, y_pred_1, y_pred_2,filtered_y_1,filtered_y_2,filtered_pred1,filtered_pred2 = model.forward(ui, uv, ai, av, y_1_input, y_2_input, epoch,time)
            if y_trues.shape[0] < 2:
                y_1_inputs = np.sum(y_1_input.detach().cpu().numpy(), axis=0)
                y_trues = y_1.detach().cpu().numpy()
                y_preds = np.sum(y_pred_1.detach().cpu().numpy(), axis=0)
                y_trues_2 = y_2.detach().cpu().numpy()
                y_preds_2 = y_pred_2.detach().cpu().numpy()
                #mask data
                for i in range(len(filtered_y_1)):
                    y_trues_2_filtered.append(filtered_y_2[i].to("cpu").detach().numpy())
                    y_preds_2_filtered.append(filtered_pred2[i].to("cpu").detach().numpy())
            else:
                y_1_inputs = y_1_inputs + np.sum(y_1_input.detach().cpu().numpy(),axis=0)
                y_trues = np.concatenate((y_trues, y_1.detach().cpu().numpy()), axis=0)
                y_preds = y_preds + np.sum(y_pred_1.detach().cpu().numpy(), axis=0)
                y_trues_2 = np.concatenate((y_trues_2, y_2.detach().cpu().numpy()), axis=0)
                y_preds_2 = np.concatenate((y_preds_2, y_pred_2.detach().cpu().numpy()), axis=0)

                # mask data
                for i in range(len(filtered_y_1)):
                    y_trues_2_filtered[i]=np.concatenate((y_trues_2_filtered[i], filtered_y_2[i].to("cpu").detach().numpy()), axis=0)
                    y_preds_2_filtered[i]=np.concatenate((y_preds_2_filtered[i], filtered_pred2[i].to("cpu").detach().numpy()), axis=0)

        if (run_type == "train"):
            loss.backward()
            optimizer.step()
        all_loss += loss.item() / y.shape[0]

    if (epoch != -1):
        run_type = "train: epoch " + str(epoch)

    if (model_name != "FLTADP" and model_name!="MLADP"):
        return calEvalResult(all_loss, y_preds, y_trues, run_type, write_file)
    else:
        # y_preds_1, y_trues_1 whether active every day (category) : [data_num, label_num]
        # y_preds_2, y_trues_2 active days percentage (category) : [data_num, label_num]
        predict_results = y_preds, y_trues, y_preds_2, y_trues_2

        if run_type=='test':
            fileName_1=fileName+'pred_1.npy'
            y_preds = y_preds / datas_user_num
            np.save(fileName_1,y_preds)
            fileName_2 = fileName + '_1.npy'
            y_1_inputs = y_1_inputs / datas_user_num
            np.save(fileName_2, y_1_inputs)

        return calEvalResult_FLTADP(all_loss, predict_results, run_type, write_file)