import numpy as np
import pandas as pd
import datetime

def preprocess(file_path, day, future_day):
    df = pd.read_csv(file_path)
    length = len(df)
    for i in range(day , day + future_day):
        df['day' + str(i + 1 - day)] = ''
        df['week' + str(i + 1 - day)] = ''
        for j in range(length):
            date = datetime.datetime.strptime(df['from'][j],'%Y-%m-%d').date()  + datetime.timedelta(days = i)
            df['day' + str(i + 1-day)][j] = date
            df['week' + str(i + 1-day)][j] = date.isoweekday()
    del df['from']
    del df['to']
    df.to_csv('./data/KDD/preprocess.csv', index = False)

def data_preprocess(path_name, file_path, type):
    data = pd.read_csv(path_name)
    df = pd.read_csv(file_path)
    new_data = pd.merge(data, df, on=['course_id'], how='left')
    del new_data['username']
    del new_data['course_id']
    path_name = './data/KDD/' + type + '_time.csv'
    new_data.to_csv(path_name, index =False)


def create_time_file(day,future_day):
    file_path = './data/KDD_Log/date.csv'
    train_path = './data/KDD_Log/train/enrollment_train.csv'
    test_path = './data/KDD_Log/test/enrollment_test.csv'
    process_path = './data/KDD/preprocess.csv'
    preprocess(file_path, 0,day + future_day)
    data_preprocess(train_path, process_path, 'train')
    data_preprocess(test_path, process_path, 'test')


