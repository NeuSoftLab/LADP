from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator
# behavior characteristics
act_feat = []
#User image
user_image = []
#id name
id_name=[]
label_feat_head = ['truth','total_activity_day']
label_feat = []
feature_num = 0
def create_tag_name(past_day=7,future_day=23,data_name='KDD'):
    global feature_num
    if data_name=='Baidu':
        id_name.append('cid')
        feature_num = 154
        for index in range(1,142):
            act_feat.append(str(index)+'_act_count_feature')
        for index in range(1,14):
            act_feat.append(str(index)+'_act_dur_time_feature')
        for index in range(1, 22):
            user_image.append(str(index) + '_user_feature')
    elif data_name=='Kwai':
        id_name.append('user_id')
        feature_num = 6
        for index in range(0,6):
            act_feat.append(str(index)+'#num')
        user_image.append('register_type')
        user_image.append('device_type')
    else:
        feature_num = 7
        id_name.append('enrollment_id')
        for index in range(0,7):
            act_feat.append(str(index) + '#num')
        user_image.append('user_feature_1')
        user_image.append('user_feature_2')
    sd = 'day'
    for i in range( 1, past_day + future_day + 1):
        curday = sd + str(i)
        for j in range(1, feature_num + 1):
            curday_num = curday + '_' + str(j)
            label_feat.append(curday_num)
class DataSet(Dataset):
    def __init__(self, ui, uv, ai, av, y, time):
        super(DataSet, self).__init__()
        self.ui = ui
        self.uv = uv
        self.ai = ai
        self.av = av
        self.y = y
        self.time = time
        self.len = ai.shape[0]
    def __getitem__(self, item):
        return self.ui[item], self.uv[item], self.ai[item], self.av[item], self.y[item], self.time[item]

    def __len__(self):
        return self.len

def dataparse(df_u):
    all_data = df_u
    # feature dimension
    feat_dim = 0
    feat_dict = dict()
    for f in user_image:
        cat_val = all_data[f].unique()
        # Packed into a dictionary, such as {0:0, 1:1, 2:2}, feat_dam is cumulative
        feat_dict[f] = dict(zip(cat_val, range(feat_dim, len(cat_val) + feat_dim)))
        feat_dim += len(cat_val)

    data_indice = all_data.copy()
    data_value = all_data.copy()
    for f in all_data.columns:
        if f in user_image:
            data_indice[f] = data_indice[f].map(feat_dict[f])
            data_value[f] = 1.
        else:
            data_indice.drop(f, axis=1, inplace=True)
            data_value.drop(f, axis=1, inplace=True)
    # feature dimension   data index   data value
    # print(data_indice.shape)
    return feat_dim, data_indice, data_value

def load_data(past_day, future_day, data_name,data_type,data_dilution_ratio):
    if (data_type == "mini"):
        data_type = "_" + data_type
    # Load feature enroll_id as primary key
    df_u = pd.DataFrame()
    df_a = pd.DataFrame()
    df_id = pd.DataFrame()
    # feature_path
    if data_name=='KDD':
        feature_path = './data/KDD/feature/'
    elif data_name=='Kwai':
        feature_path = './data/KwaiData/feature/'
    else:
        feature_path = './data/BaiduData'+data_type+'/feature/'
    # Get day+future_ ID of the user who has been active for day
    for i in range(1,past_day+1):
        #print(i, '-', past_day+future_day+1)
        file_name = "day_" + str(i) + "_activity_feature"
        df = pd.read_csv(feature_path + file_name + '.csv')
        df_cur_id = df[id_name]
        df_id = pd.concat([df_id, df_cur_id])
        # Delete duplicate id
        df_id.drop_duplicates(subset=id_name, inplace=True)
    #Dilute
    r,c=df_id.shape
    r = int(r*data_dilution_ratio)
    df_id = df_id.iloc[:r]

    for i in range(1, past_day + 1):
        #print(i, '-', past_day + 1)
        file_name = "day_" + str(i) + "_activity_feature"
        df = pd.read_csv(feature_path + file_name + '.csv')
        df_cur_u = df[id_name+ user_image]
        df_cur_a = df[id_name + act_feat]
        # Inactive users fill 0
        df_cur_a=pd.merge(df_id, df_cur_a, on=id_name, how='left')
        df_cur_a=df_cur_a.fillna(0)
        #Splice user portraits
        df_u = pd.concat([df_u, df_cur_u])
        # Delete duplicate users
        df_u.drop_duplicates(subset=id_name, inplace=True)
        df_a = pd.concat([df_a, df_cur_a])
    # User portrait of future days
    for i in range(past_day+1,past_day+future_day+1):
        file_name = "day_" + str(i) + "_activity_feature"
        df = pd.read_csv(feature_path + file_name + '.csv')
        df_cur_u = df[id_name + user_image]
        # Splice user portraits
        df_u = pd.concat([df_u, df_cur_u])
        df_u.drop_duplicates(subset=id_name, inplace=True)
    # read label
    #label_path
    if data_name=='KDD':
        label_path = './data/KDD/info/'
    elif data_name=='Kwai':
        label_path = './data/KwaiData/info/'
    else:
        label_path = './data/BaiduData'+data_type+'/info/'
    label = pd.read_csv(label_path+'user_info.csv')
    label = label[id_name+ label_feat]
    label['total_activity_day'] = 0
    i = 1
    #print('ff:',feature_num,future_day)
    while i < feature_num*future_day:
        label['temp_1'] = label.iloc[:, i:i + feature_num].sum(axis=1)
        label = label.assign(temp=label.temp_1.ge(1)*1)
        label['total_activity_day'] += label['temp']
        i = i+feature_num
    label['truth'] = label['total_activity_day'] / future_day
    # emove the previous day+future_ Inactive users in day
    label = pd.merge(df_id,label,on=id_name, how='left')
    # sort
    df_a.sort_values(id_name, inplace=True)
    df_u.sort_values(id_name, inplace=True)
    label.sort_values(id_name, inplace=True)
    # read time
    df_time = pd.read_csv(label_path+'user_time_info.csv')
    df_time = pd.merge(df_id,df_time,on=id_name, how='left')
    df_time.sort_values(id_name, inplace=True)
    del df_time[id_name[0]]
    df_time_split = pd.DataFrame()
    for i in range(1, past_day + future_day + 1):
        df_time_split['year' + str(i)] = pd.to_datetime(df_time["day" + str(i)]).dt.year
        df_time_split['month' + str(i)] = pd.to_datetime(df_time["day" + str(i)]).dt.month
        df_time_split['day' + str(i)] = pd.to_datetime(df_time["day" + str(i)]).dt.day
        df_time_split['week' + str(i)] = df_time["week" + str(i)]
    return df_u, df_a, label, df_time_split

def getDataLoader(batch_size=32, params={}, data_name = 'KDD',data_path='./KDD',data_type = ""):
    past_day = params['day']
    future_day = params['future_day']
    data_dilution_ratio = params['data_dilution_ratio']
    l = ['ui_train','uv_train','ai_train','av_train','y_train','time_train',
         'ui_valid','uv_valid','ai_valid','av_valid','y_valid','time_valid',
         'ui_test','uv_test','ai_test','av_test','y_test','time_test',
         'day_numpy','params']
    save_path_pre = './data/' + data_name + '_' + data_type + '/model_input/'
    save_path_last = str(past_day) + '_' + str(future_day) + '_' + str(params['seed'])+'_'+str(data_dilution_ratio)
    if os.path.exists(save_path_pre + l[0] + save_path_last + '.pt'):
        ui_train = torch.load(save_path_pre + l[0] + save_path_last + '.pt')
        uv_train = torch.load(save_path_pre + l[1] + save_path_last + '.pt')
        ai_train = torch.load(save_path_pre + l[2] + save_path_last + '.pt')
        av_train = torch.load(save_path_pre + l[3] + save_path_last + '.pt')
        y_train = torch.load(save_path_pre + l[4] + save_path_last  + '.pt')
        time_train = torch.load(save_path_pre + l[5] + save_path_last  + '.pt')

        ui_valid = torch.load(save_path_pre + l[6] + save_path_last + '.pt')
        uv_valid = torch.load(save_path_pre + l[7] + save_path_last + '.pt')
        ai_valid = torch.load(save_path_pre + l[8] + save_path_last + '.pt')
        av_valid = torch.load(save_path_pre + l[9] + save_path_last + '.pt')
        y_valid = torch.load(save_path_pre + l[10] + save_path_last + '.pt')
        time_valid = torch.load(save_path_pre + l[11] + save_path_last + '.pt')

        ui_test = torch.load(save_path_pre + l[12] + save_path_last +'.pt')
        uv_test = torch.load(save_path_pre + l[13] + save_path_last +'.pt')
        ai_test = torch.load(save_path_pre + l[14] + save_path_last +'.pt')
        av_test = torch.load(save_path_pre + l[15] + save_path_last +'.pt')
        y_test = torch.load(save_path_pre + l[16] + save_path_last +'.pt')
        time_test = torch.load(save_path_pre + l[17] + save_path_last +'.pt')
        day_numpy = np.load(save_path_pre + l[18] + save_path_last +'.npy')
        params_load = np.load(save_path_pre + l[19] + save_path_last +'.npy', allow_pickle=True).item()
        #折线图
        y = np.load(save_path_pre + 'y' + save_path_last +'.npy')
        av = np.load(save_path_pre + 'av' + save_path_last +'.npy')
        draw_day_action_num(y, past_day,future_day,params_load["a_feat_size"],data_name)

        params_load.update(params)
        params = params_load
    else:
        create_tag_name(past_day,future_day,data_name)
        # load data
        print("Begin Load Df_u, Df_a, label, time_split!")
        df_u, df_a, label,time_split = load_data(past_day, future_day,data_name, data_type,data_dilution_ratio)
        #print(df_u.shape, df_a.shape, label.shape, time_split.shape)
        print("End Load Df_u, Df_a, label, time_split!")

        # Feature dimension   data index   data value(student)
        u_feat_dim, u_data_indice, u_data_value = dataparse(df_u)

        # Turn u_data_indice and u_data_value into array
        ui, uv = np.asarray(u_data_indice.loc[df_u.index], dtype=int), np.asarray(
            u_data_value.loc[df_u.index], dtype=np.float32)
        params["u_feat_size"] = u_feat_dim
        params["u_field_size"] = len(ui[0])
        # action
        av = np.asarray(df_a[act_feat], dtype=np.float32)
        #print("av", av.shape)
        ai = np.asarray([range(len(act_feat)) for x in range(len(df_a))], dtype=int)
        params["a_feat_size"] = len(av[0])
        params["a_field_size"] = len(ai[0])
        # user_id,day,action_type_num
        av = av.reshape((-1, params['day'], len(act_feat)))
        ai = ai.reshape((-1, params['day'], len(act_feat)))
        params['input_size'] = len(act_feat)
        # time
        # time_npy:[user_num,(past_day+future_day)*4]
        time_npy = np.asarray(time_split, dtype=np.float32)
        # time_npy:[user_num,past_day+future_day,4]
        time_npy = time_npy.reshape((-1, past_day + future_day, 4))
        y = np.asarray(label[label_feat_head+label_feat], dtype=np.float32)
        np.save(save_path_pre + 'av' + save_path_last + '.npy', av)
        np.save(save_path_pre + 'y' + save_path_last + '.npy', y)
        draw_day_action_num(y, past_day,future_day, params["a_feat_size"],data_name)

        # Turn all data into tensor
        # train
        ui = torch.tensor(ui)
        uv = torch.tensor(uv)
        time = torch.tensor(time_npy)
        y = torch.tensor(y)
        # Divide the training set into the validation set
        data_num = len(y)
        indices = np.arange(data_num)
        np.random.seed(params['seed'])
        np.random.shuffle(indices)
        split_1 = int(0.6 * data_num)
        split_2 = int(0.8 * data_num)
        ui_train, ui_valid, ui_test = ui[indices[:split_1]], ui[indices[split_1:split_2]], ui[indices[split_2:]]
        uv_train, uv_valid, uv_test = uv[indices[:split_1]], uv[indices[split_1:split_2]], uv[indices[split_2:]]
        ai_train, ai_valid, ai_test = ai[indices[:split_1]], ai[indices[split_1:split_2]], ai[indices[split_2:]]
        av_train, av_valid, av_test = av[indices[:split_1]], av[indices[split_1:split_2]], av[indices[split_2:]]
        time_train, time_valid, time_test = time[indices[:split_1]], time[indices[split_1:split_2]], \
                                            time[indices[split_2:]]
        y_train, y_valid, y_test = y[indices[:split_1]], y[indices[split_1:split_2]], y[indices[split_2:]]
        label = label.iloc[indices[:split_1]]
        day_numpy = user_activate_day_count(params, label)

        torch.save(ui_train, save_path_pre + l[0] + save_path_last + '.pt')
        torch.save(uv_train, save_path_pre + l[1] + save_path_last + '.pt')
        torch.save(ai_train, save_path_pre + l[2] + save_path_last + '.pt')
        torch.save(av_train, save_path_pre + l[3] + save_path_last + '.pt')
        torch.save(y_train, save_path_pre + l[4] + save_path_last + '.pt')
        torch.save(time_train, save_path_pre + l[5] + save_path_last+ '.pt')

        torch.save(ui_valid, save_path_pre + l[6] + save_path_last + '.pt')
        torch.save(uv_valid, save_path_pre + l[7] + save_path_last+ '.pt')
        torch.save(ai_valid, save_path_pre + l[8] + save_path_last + '.pt')
        torch.save(av_valid, save_path_pre + l[9] + save_path_last+ '.pt')
        torch.save(y_valid, save_path_pre + l[10] + save_path_last+ '.pt')
        torch.save(time_valid, save_path_pre + l[11] + save_path_last + '.pt')

        torch.save(ui_test, save_path_pre + l[12] + save_path_last+ '.pt')
        torch.save(uv_test, save_path_pre + l[13] + save_path_last + '.pt')
        torch.save(ai_test, save_path_pre + l[14] + save_path_last + '.pt')
        torch.save(av_test, save_path_pre + l[15] + save_path_last + '.pt')
        torch.save(y_test, save_path_pre + l[16] + save_path_last + '.pt')
        torch.save(time_test, save_path_pre + l[17] + save_path_last+ '.pt')

        np.save(save_path_pre + l[18] + save_path_last + '.npy', day_numpy)
        np.save(save_path_pre + l[19] + save_path_last+ '.npy', params)

    # ui_train: [user_num , user_image_type]
    # ai_train_: [user_num , past_day , action_type]
    # time_train: [user_num ,past_day + future_day]
    # y:[user_num,truth + total_activity_day + day1_1...dayN_feature_num  ]
    train_dataset = DataSet(ui_train, uv_train, ai_train, av_train, y_train, time_train)
    valid_dataset = DataSet(ui_valid, uv_valid, ai_valid, av_valid, y_valid,time_valid)
    test_dataset = DataSet(ui_test, uv_test, ai_test, av_test, y_test,time_test)
    # packaged dataset
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_set = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)
    print(day_numpy)
    print(params)
    return train_set, valid_set, test_set, day_numpy, params

def user_activate_day_count(param, label):
    day_list = []
    for i in range(0,param['future_day']+1):
        cur_day_count = (label['total_activity_day']==i).sum()
        day_list.append(cur_day_count)
    print(day_list)
    day_numpy = np.array(day_list)
    return day_numpy

def draw_day_action_num(y,day,future_day,a_feat_size,data_name):
    #action_y : future_day+day * a_feat_size
    action_y = y[:, 2:]
    action_y[action_y != 0] = 1
    print(a_feat_size)
    action_y = action_y.reshape((-1, day+future_day, a_feat_size))
    user_num,_,_=action_y.shape
    action_y =action_y.sum(axis=0)
    x=np.arange(1,day+future_day+1,1)
    colors_type = ['red', 'green', 'blue','yellow','gold', 'cyan', 'm']
    colors = []
    i = 0
    for  j in range(0,a_feat_size):
        colors.append(colors_type[i])
        i += 1
        if i>=len(colors_type):
            i = 0
    label = 'feature_'
    plt.figure(figsize=(10, 8), dpi=150)
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    for i in range(0,a_feat_size):
        cur_action = action_y[:,i]
        cur_action_std= cur_action/user_num # 概率
        plt.plot(x, cur_action_std, marker='o',c=colors[i], label=label+str(i))
    plt.title("All Daily activity statistics of different characteristics")
    plt.legend(loc='best')
    plt.savefig( data_name + '.jpg')
    plt.show()
if __name__ == '__main__':
    params_ = {'day': 7, 'future_day': 23,'seed':1}
    getDataLoader(batch_size=32,params=params_,data_type='mini')