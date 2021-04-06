import csv
import pandas as pd
import pymongo as py
import numpy as np


class DataProcess:
    def __init__(self):
        self.train_csv_path = './data/train_csv.csv'
        # self.train_csv_label_path = './data/train_csv_label.csv'
        self.test_csv_path = './data/test_csv.csv'
        # self.test_csv_label_path = './data/test_csv_label.csv'

    def check_test_train_feature(self):
        with open(self.train_csv_path, 'r') as f:
            lines = csv.reader(f)
            lines = [i for i in lines]
        titles1 = lines.pop(0)

        with open(self.test_csv_path, 'r') as f:
            lines = csv.reader(f)
            lines = [i for i in lines]
        titles2 = lines.pop(0)

        for i in range(len(titles1)):
            if titles1[i] == titles2[i]:
                continue
            raise Exception('not same')
        print('len feature', len(titles1))
        print('all feature is same')


    def normal(self):
        train_data = pd.read_csv(self.train_csv_path)
        test_data = pd.read_csv(self.test_csv_path)
        # 去掉有nan 和 inf 的行
        train_data = train_data.dropna(axis=0, how='any')
        test_data = test_data.dropna(axis=0, how='any')
        # 替换
        # train_data = train_data.fillna(0)
        # test_data = test_data.fillna(0)
        # 将空值改为0
        for i in ['Flow Byts/s', 'Flow Pkts/s']:
            train_column = train_data[i]
            test_column = test_data[i]
            train_column[train_column==' '] = 0
            test_column[test_column==' '] = 0
        # 用numpy将nan改为0，inf改为有限数字
        # train_data = np.nan_to_num(train_data)
        # test_data = np.nan_to_num(test_data)
        with open(self.test_csv_path, 'r') as f:
            lines = csv.reader(f)
            lines = [i for i in lines]
        titles = lines.pop(0)
        titles.pop(0)
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        for i in titles:
            train_data[i] = self.normalist(train_data, i)
            test_data[i] = self.normalist(test_data, i)
        train_data.to_csv('./data/train_normal_data.csv')
        test_data.to_csv('./data/test_normal_data.csv')

    def normalist(self, data, column_name):
        df_column = data[column_name]
        min = df_column.min()
        max = df_column.max()
        df_column = (df_column - min) / (max - min)
        return df_column

    def run(self):
        self.check_test_train_feature()
        self.normal()


if __name__ == '__main__':
    DataProcess().run()

