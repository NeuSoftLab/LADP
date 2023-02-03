import pandas as pd
from sklearn import preprocessing


# Filter the users who are 'day' before the number of registration days and save it as kwai_user_info.csv;
# Obtain all activity information of users who registered in the previous 'day' and save it as all_activity_log.csv
def extraction_data(activity_log_file_path, register_file_path, file_path, day):
    activity_log = pd.read_csv(activity_log_file_path, sep='\t',
                               names=['user_id', 'act_day', 'page', 'video_id', 'author_id', 'act_type'])
    register = pd.read_csv(register_file_path, sep='\t',
                           names=['user_id', 'register_day', 'register_type', 'device_type'])

    register = register[register['register_day'] <= day]
    # user_id  register_type  device_type
    del register['register_day']
    register.to_csv(file_path + 'info/kwai_user_info.csv', index=False)
    print('Kwai user info is created successfully')

    del activity_log['page']
    del activity_log['video_id']
    del activity_log['author_id']

    # user_id
    reg_user_id = register.copy()
    del reg_user_id['register_type']
    del reg_user_id['device_type']

    # user_id  act_day  act_type  register_type  device_type
    activity_log = pd.merge(activity_log, register, on='user_id')
    activity_log.to_csv(file_path + 'log/all_activity_log.csv', index=False)
    print('All activity data is created successfully')


def preprocess_data(all_activity_log_file_path, kwai_user_info_file_path, file_path, day, future_day):
    all_activity_log = pd.read_csv(all_activity_log_file_path)
    register = pd.read_csv(kwai_user_info_file_path)

    reg_user_id = register.copy()
    del reg_user_id['register_type']
    del reg_user_id['device_type']

    # Count and save user activities from 1 to 'day', if there is no activity record, it will be regarded as 0
    action_type_dict = [0, 1, 2, 3, 4, 5]
    for i in range(1, day + 1):
        file = all_activity_log.copy()
        file = file[file['act_day'] == i]
        file.rename(columns={'act_day': 'day_id'}, inplace=True)
        all_num = file.groupby('user_id').agg({'act_type': 'count'})
        all_num.rename(columns={'act_type': 'all#num'}, inplace=True)
        for a in action_type_dict:
            action_ = (file['act_type'] == a)
            file[str(a) + '#num'] = action_
            action_num = file.groupby('user_id').sum(numeric_only=True)[[str(a) + '#num']]
            all_num = pd.merge(all_num, action_num, left_index=True, right_index=True)
        del file
        all_num = pd.merge(all_num, register, how='right', on='user_id')
        # Fill NAN
        all_num = all_num.fillna(0)
        all_num.insert(loc=1, column='day_id', value=i)
        all_num.to_csv(file_path + 'log/day_' + str(i) + '_activity_log.csv', index=False)
        print('Day ' + str(i) + ' activity data is created successfully')
        del all_num

    # Add truth information, 1 means active, 0 means inactive
    future_day_activity_log = all_activity_log.copy()
    future_day_activity_log = future_day_activity_log[(day + 1) <= future_day_activity_log['act_day']]
    future_day_activity_log = future_day_activity_log[(day + 1 + future_day) >= future_day_activity_log['act_day']]
    all_truth = future_day_activity_log.groupby('user_id').count()[['act_type']]
    all_truth[all_truth.act_type > 0] = 1
    all_truth.columns = ['truth']
    user_truth = pd.merge(reg_user_id, all_truth, how='left', on='user_id')
    user_truth['truth'] = user_truth['truth'].fillna(0)

    # user_id  register_type  device_type  truth
    register = pd.merge(register, user_truth, how='inner', on='user_id')
    print('Kwai user info add truth over')

    # Add whether the 'future_day' day is active or not
    # user_id  register_type  device_type  truth  first_day  second_day ... future_day
    for i in range(day + 1, day + future_day + 1):
        file = all_activity_log.copy()
        file = file[file['act_day'] == i]
        whether_activity = file.groupby('user_id').count()[['act_type']]
        del file
        whether_activity[whether_activity.act_type > 0] = 1
        whether_activity.columns = ['day' + str(i - day)]
        user_activity = pd.merge(reg_user_id, whether_activity, how='left', on='user_id')
        del whether_activity
        user_activity['day' + str(i - day)] = user_activity['day' + str(i - day)].fillna(0)
        register = pd.merge(register, user_activity, how='inner', on='user_id')
        del user_activity
    print('Kwai user info add ' + str(day + 1) + ' to ' + str(day + 1 + future_day + 1) + ' activity over')

    # Add 21~30 total active days information
    # user_id  register_type  device_type  truth  first_day  second_day ... future_day  total_activity_num
    register['total_activity_day'] = 0
    for i in range(day + 1, day + future_day + 1):
        register['total_activity_day'] += register['day' + str(i - day)]
    print('Kwai user info add total activity day over')

    # truth = total_activity_day / future_day
    register['truth'] = register['total_activity_day'] / future_day
    register.to_csv(file_path + 'info/kwai_user_info.csv', index=False)


# Standardization
def standardscaler(file_path, days):
    scaler = preprocessing.StandardScaler()
    action_feats = ['all#num', '0#num', '1#num', '2#num', '3#num', '4#num', '5#num']
    for i in range(1, days + 1):
        path = file_path + 'log/day_' + str(i) + '_activity_log.csv'
        df = pd.read_csv(path, engine='python')
        newX = scaler.fit_transform(df[action_feats])
        for j, n_f in enumerate(action_feats):
            df[n_f] = newX[:, j]
        df.to_csv(file_path + 'feature/day_' + str(i) + '_activity_feature.csv', index=False)
        print('Day ' + str(i) + ' activity feature is created successfully')


def create_file_by_data(day, future_day, dilution_ratio=1.0):
    # Source files path
    activity_log_file_path = './data/KwaiData_Log/user_activity_log.txt'
    register_file_path = './data/KwaiData_Log/user_register_log.txt'

    # Activity logs and user info file path
    all_activity_log_file_path = './data/KwaiData/log/all_activity_log.csv'
    kwai_user_info_file_path = './data/KwaiData/info/kwai_user_info.csv'

    # Kwai data file path
    file_path = './data/KwaiData/'

    extraction_data(activity_log_file_path, register_file_path, file_path, day)
    preprocess_data(all_activity_log_file_path, kwai_user_info_file_path, file_path, day, future_day)
    standardscaler(file_path, day)


if __name__ == '__main__':
    # test
    create_file_by_data(7, 23)