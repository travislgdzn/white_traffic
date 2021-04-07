import csv
import numpy as np
import pandas as pd


class CsvProcess(object):
    def __init__(self):
        self.csv_path1 = '/home/user/ZN/intrusion detection/IDS2018/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path2 = '/home/user/ZN/intrusion detection/IDS2018/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path3 = '/home/user/ZN/intrusion detection/IDS2018/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path4 = '/home/user/ZN/intrusion detection/IDS2018/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path5 = '/home/user/ZN/intrusion detection/IDS2018/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path6 = '/home/user/ZN/intrusion detection/IDS2018/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path7 = '/home/user/ZN/intrusion detection/IDS2018/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path8 = '/home/user/ZN/intrusion detection/IDS2018/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path9 = '/home/user/ZN/intrusion detection/IDS2018/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv'
        self.csv_path10 = '/home/user/ZN/intrusion detection/IDS2018/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv'


    def creat_data_csv(self):

        with open(self.csv_path1, 'r') as f1:
            lines = csv.reader(f1)
            lines = [i for i in lines]
        titles = lines.pop(0)
        pf1 = pd.DataFrame(lines, columns=list(titles))

        with open(self.csv_path2, 'r') as f2:
            lines = csv.reader(f2)
            lines = [i for i in lines]
        lines.pop(0)
        pf2 = pd.DataFrame(lines, columns=list(titles))

        with open(self.csv_path3, 'r') as f3:
            lines = csv.reader(f3)
            lines = [i for i in lines]
        lines.pop(0)
        pf3 = pd.DataFrame(lines, columns=list(titles))

        # with open(self.csv_path4, 'r') as f4:
        #     lines = csv.reader(f4)
        #     lines = [i for i in lines]
        # lines.pop(0)
        # pf4 = pd.DataFrame(lines, columns=list(titles))

        with open(self.csv_path5, 'r') as f5:
            lines = csv.reader(f5)
            lines = [i for i in lines]
        lines.pop(0)
        pf5 = pd.DataFrame(lines, columns=list(titles))

        with open(self.csv_path6, 'r') as f6:
            lines = csv.reader(f6)
            lines = [i for i in lines]
        lines.pop(0)
        pf6 = pd.DataFrame(lines, columns=list(titles))

        with open(self.csv_path7, 'r') as f7:
            lines = csv.reader(f7)
            lines = [i for i in lines]
        lines.pop(0)
        pf7 = pd.DataFrame(lines, columns=list(titles))

        with open(self.csv_path8, 'r') as f8:
            lines = csv.reader(f8)
            lines = [i for i in lines]
        lines.pop(0)
        pf8 = pd.DataFrame(lines, columns=list(titles))

        with open(self.csv_path9, 'r') as f9:
            lines = csv.reader(f9)
            lines = [i for i in lines]
        lines.pop(0)
        pf9 = pd.DataFrame(lines, columns=list(titles))

        with open(self.csv_path10, 'r') as f10:
            lines = csv.reader(f10)
            lines = [i for i in lines]
        lines.pop(0)
        pf10 = pd.DataFrame(lines, columns=list(titles))

        res = pf1.append([pf2, pf3, pf5, pf6, pf7, pf8, pf9, pf10], ignore_index=True)

        Benign = res.loc[res['Label'] == 'Benign']
        Attack = res.loc[res['Label'] != 'Benign']

        Benign.to_csv('/home/user/ZN/intrusion detection/IDS2018/Benign.csv', index=None)
        Attack.to_csv('/home/user/ZN/intrusion detection/IDS2018/Attack.csv', index=None)

    def creat_train_test(self):
        Benign = pd.read_csv('/home/user/ZN/intrusion detection/IDS2018/Benign.csv', low_memory=False)
        Attack = pd.read_csv('/home/user/ZN/intrusion detection/IDS2018/Attack.csv', low_memory=False)
        # 在Benign 和 Attack 中随机抽取行np.random.randint(start, end, number)
        r_benign = np.random.randint(0, 6000000, 200000)
        r_attack = np.random.randint(0, 2000000, 50000)
        new_benign = Benign.iloc[r_benign, :]
        new_attack = Attack.iloc[r_attack, :]
        # 合并&打乱
        res = pd.concat([new_benign, new_attack], axis=0, ignore_index=True)
        data = res.sample(frac=0.8).reset_index(drop=True)
        # # 将['Timestamp']列的时间改成时间戳
        # dtime = pd.to_datetime(data['Timestamp'])
        # v = (dtime.values - np.datetime64('1970-01-01T08:00:00Z')) / np.timedelta64(1, 'ms')
        # data['Timestamp'] = v
        # # 时间戳转化成时间
        data.pop('Timestamp')
        # data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms', origin=pd.Timestamp('2015-01-01 08:00:00'))
        train_csv = data.iloc[100000:]
        # trian_csv_label = train_csv.loc('Label')
        test_csv = data.iloc[:100000]
        # test_csv_label = test_csv.loc('Label')
        # data.to_csv('/home/user/ZN/intrusion detection/data/data.csv')
        train_csv.to_csv('/home/user/ZN/intrusion detection/data/train_csv.csv', index=None)
        # trian_csv_label.to_csv('./data/train_csv_label.csv')
        test_csv.to_csv('/home/user/ZN/intrusion detection/data/test_csv.csv', index=None)
        # test_csv_label.to_csv('./data/test_csv_label.csv')

    def run(self):
        # self.creat_data_csv()
        self.creat_train_test()

        print('----------------------------------------')



if __name__ == '__main__':
    CsvProcess().run()

    print('--------------------------------------------')
