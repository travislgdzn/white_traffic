import pandas as pd
#import pymongo as py
import numpy as np
import csv
import time
import os

# # data1 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv')
# # print(data1)
# # print('-----------------------------------------------------------------------')
# # data2 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv')
# # print(data2)
# # print('-----------------------------------------------------------------------')
# # data3 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv')
# # print(data3)
# # print('-----------------------------------------------------------------------')
# # data4 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv')
# # print(data4)
# # print('-----------------------------------------------------------------------')
# # data5 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv')
# # print(data5)
# # print('-----------------------------------------------------------------------')
# # data6 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv')
# # print(data6)
# # print('-----------------------------------------------------------------------')
# # data7 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv')
# # print(data7)
# # print('-----------------------------------------------------------------------')
# # data8 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv')
# # print(data8)
# # print('-----------------------------------------------------------------------')
# # data9 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv')
# # print(data9)
# # print('-----------------------------------------------------------------------')
# # data10 = pd.read_csv('D:\学习资料\论文\小论文\intrusion detection\data\IDS2018\Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv')
# # print(data10)
# # print('-----------------------------------------------------------------------')
#
# ################################################################################################################################################
#
# # with open('.\data\IDS2018\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv', 'r') as f1:
# #     lines = csv.reader(f1)
# #     lines = [i for i in lines]
# # titles = lines.pop(0)
# # pf1 = pd.DataFrame(lines[:1000], columns=list(titles))
# #
# # with open('.\data\IDS2018\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f2:
# #     lines = csv.reader(f2)
# #     lines = [i for i in lines]
# # lines.pop(0)
# # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# #
# # with open('.\data\IDS2018\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f2:
# #     lines = csv.reader(f2)
# #     lines = [i for i in lines]
# # lines.pop(0)
# # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# #
# # with open('.\data\IDS2018\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv', 'r') as f4:
# #     lines = csv.reader(f4)
# #     lines = [i for i in lines]
# # titles = lines.pop(0)
# # print(titles)
# # # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# # #
# # with open('.\data\IDS2018\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f5:
# #     lines = csv.reader(f5)
# #     lines = [i for i in lines]
# # titles = lines.pop(0)
# # print(titles)
# # # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# #
# # with open('.\data\IDS2018\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f6:
# #     lines = csv.reader(f6)
# #     lines = [i for i in lines]
# # titles = lines.pop(0)
# # print(titles)
# # # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# # #
# # with open('.\data\IDS2018\Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f7:
# #     lines = csv.reader(f7)
# #     lines = [i for i in lines]
# #     titles =titles.pop(0)
# # print(titles)
# # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# #
# # with open('.\data\IDS2018\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f2:
# #     lines = csv.reader(f2)
# #     lines = [i for i in lines]
# # lines.pop(0)
# # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# #
# # with open('.\data\IDS2018\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f2:
# #     lines = csv.reader(f2)
# #     lines = [i for i in lines]
# # lines.pop(0)
# # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# #
# # with open('.\data\IDS2018\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'r') as f2:
# #     lines = csv.reader(f2)
# #     lines = [i for i in lines]
# # lines.pop(0)
# # pf2 = pd.DataFrame(lines[:1000], columns = list(titles))
# #
# # res = pf2.append(pf1, ignore_index = True)
# # # print(res)
# # benign = res.loc[res['Label'] != 'Benign']
# # # print(benign)
#
# # benign.to_csv('.\data\IDS2018\Benign.csv')
#
# #####################################################################################################################################
#
#
# # Benign = pd.read_csv('./data/IDS2018/Benign.csv', low_memory=False)
# # Attack = pd.read_csv('./data/IDS2018/Attack.csv', low_memory=False)
# #
# # r_benign = np.random.randint(0, 6000000, 2000)
# # r_attack = np.random.randint(0, 2000000, 1000)
# #
# # new_benign = Benign.iloc[r_benign, :]
# # new_attack = Attack.iloc[r_attack, :]
# #
# # res = pd.concat([new_benign, new_attack], axis=0, ignore_index=True)
# # data = res.sample(frac=1).reset_index(drop=True)
# #
# # dtime = pd.to_datetime(data['Timestamp'])
# # v = (dtime.values - np.datetime64('2015-01-01T08:00:00Z')) / np.timedelta64(1, 'ms')
# # data['Timestamp'] = v
# # # data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms', origin=pd.Timestamp('2015-01-01 08:00:00'))
# # # data.pop('Unnamed: 0')
# #
# # train_csv = data.iloc[1000:].reset_index(drop=True)
# # trian_csv_label = train_csv.pop('Label')
# #
# # test_csv = data.iloc[:1000].reset_index(drop=True)
# # test_csv_label = test_csv.pop('Label')
# #
# # # data.to_csv('./data/IDS2018/data.csv')
# # # train_csv.to_csv('./data/train_csv.csv')
# # # trian_csv_label.to_csv('./data/train_csv_label.csv')
# # test_csv.to_csv('./data/IDS2018/test_csv.csv')
# # test_csv_label.to_csv('./data/test_csv_label.csv')
#
# #################################################################################################################################
#
# # test = pd.read_csv('./test_data/test_csv.csv')
# # # train =pd.read_csv('./data/train_csv.csv')
# #
# # # test = test.fillna(value=0.0, axis=0)
# # # for i in range(len(test)):
# # # test = test.dropna(axis=0, how='any')
# # #
# # #
# # # with open('./data/test_csv.csv', 'r') as f:
# # #     lines = csv.reader(f)
# # #     lines = [i for i in lines]
# # # titles = lines.pop(0)
# # # titles.pop(0)
# # #
# # #
# # # # for i in range(len(test)):
# # # #     lines[i].pop(0)
# # # # lines = np.nan_to_num(lines)
# # # # test_data = np.array(lines, dtype='float')
# # #
# # # # datas = []
# # # # for i in titles:
# # # #     datas.extend([test.loc[:, i]])
# # # #
# # # # datas = np.nan_to_num(datas)
# # # # datas = np.array(datas, dtype='float')
# # # #
# # # # max_list = np.max(datas, axis=1)
# # # # min_list = np.min(datas, axis=1)
# # # # max_min_list = max_list - min_list
# # # # for i in range(len(max_list)):
# # # #     print(max_list[i],i)
# # # # for i in range(len(min_list)):
# # # #     print(min_list[i], i)
# # # # for i in range(len(max_min_list)):
# # # #     print(max_min_list[i],i)
# # # # print(len(max_list))
# # # # print(len(test))
# # #
# # # # normal_data = []
# # # # for row in test.iterrows():
# # # #     for i in range(len(max_min_list)):
# # # #         test.iloc[row, i] = (test.iloc[row, i ] - min_list)/max_min_list
# # # #     normal_data.append(test.loc[row])
# # # # print(normal_data)
# # #
# # # # normal_data = []
# # # # for data in test_data:
# # # #     for i in range(len(max_min_list)):
# # # #         data[i] = (data[i] - min_list[i])/max_min_list[i]
# # # #     normal_data.append(data)
# # # # print(normal_data)
# # # def normal(data, column_name):
# # #     df_column = data[column_name]
# # #     min = df_column.min()
# # #     max = df_column.max()
# # #     max_min = max - min
# # #     df_column = (df_column - min)/max_min
# # #     return df_column
# # #
# # # for i in titles:
# # #     column = normal(test, i)
# # #     test[i] = column
# # # test = test.reset_index(drop=True)
# # # # print(test)
# # # print(titles)
# # test.pop('Unnamed: 0')
# # # test.to_csv('./test_data/d.csv',index=None)
# # d = pd.read_csv('./test_data/d.csv')
# # print(d)
#
# # os.mkdir('./history_more')
# p = pd.pandas.read_csv('/home/user/ZN/intrusion detection/IDS2018/Attack.csv')
# print(p)
# label_list = list(set(tuple(p['Label'])))
# print(label_list)
# label_map = {value: str(key) for key, value in enumerate(label_list)}
# # print(label_map)
# p['Label'] = p['Label'].map(label_map)
# print(p['Label'])
# samples = np.array(list(range(0, 10)))
# print(samples)

train_data = pd.read_csv('/home/user/ZN/intrusion detection/data/train_csv.csv')
test_data = pd.read_csv('/home/user/ZN/intrusion detection/data/test_csv.csv')
Benign = pd.read_csv('/home/user/ZN/intrusion detection/IDS2018/Benign.csv')
Attack = pd.read_csv('/home/user/ZN/intrusion detection/IDS2018/Attack.csv')

# train_label = train_data['Label']
# test_label = test_data['Label']
# train_count = train_label.value_counts()
# test_count = test_label.value_counts()
# print('train_count:\n', train_count)
# print(train_label.shape[0])
# print('------------------------------------')
# print('train_count:\n', test_count)
# print(test_label.shape[0])

# Benign_label = Benign['Label']
# Attack_label = Attack['Label']
# Benign_count = Benign_label.value_counts()
# Attack_count = Attack_label.value_counts()
# print('Benign:\n', Benign_count)
# print(Benign.shape[0])
# print('---------------------------------')
# print('Attack:\n', Attack_count)
# print(Attack.shape[0])
# predict =model.predict(x_test)
# fp ,tp ,th = metric.roc_curve(x_test, predict)
# plt.plot(fp, tp ,'b')

import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))