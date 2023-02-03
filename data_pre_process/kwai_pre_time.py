import numpy as np
import pandas as pd
import datetime

def preprocess(file_path, day, future_day):
    df = pd.read_csv(file_path)
    data = df[['user_id']]
    data['day1'] = datetime.datetime.strptime('2022-05-02','%Y-%m-%d').date()
    data['week1'] = datetime.datetime.strptime('2022-05-02','%Y-%m-%d').date().isoweekday()
    for i in range(day + 1 , day + future_day):
        data['day' + str(i + 1 - day)] = ''
        data['week' + str(i + 1 - day)] = ''
        date = data.iloc[0][1]  + datetime.timedelta(days = i)
        data['day' + str(i + 1-day)] = date
        data['week' + str(i + 1-day)] = date.isoweekday()
    data.to_csv('./data/KwaiData/kwai_time.csv', index = False)

def create_time_file(day,future_day):
    file_path = './data/KwaiData/info/kwai_user_info.csv'
    preprocess(file_path, 0, day+future_day)