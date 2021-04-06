import pandas
import numpy as np
import csv
from models_build import ModelBuild
import json
import time
import os
import pandas as pd
from sklearn.preprocessing import Normalizer

from tensorflow.keras.utils import to_categorical


class DataSet(object):
    """
    该类是数据集类，包含２个属性，
    datas：该属性是数据集的特征列的数据信息，shape 是(*, 1, 196)
    labels：该属性包含的是对应数据的label 标签, shape 是(*, 2)
        ２种属性都使用list 占位
    """
    datas: list
    labels: list


class DataRecord:
    """
    该类包含的信息是
    model: 模型的model 信息，该模型是.h5 类型数据
           是tensorflow用于保存模型的特定格式
           该类型的数据保存了模型的包括模型结构，规模大小,
           模型各个层的权重，参数信息
    history: 保存了模型训练过程中的损失和精度所有值，
             该信息需要保存下来为后期的绘制模型训练图提供数据
    summary: 模型的结构信息
    epochs: 模型的迭代次数信息
    """
    model: str
    history: str
    summary: str
    epochs: int


class TrainModels(object):
    def __init__(self, model, name):
        self.train_path = '/home/user/ZN/intrusion detection/data/train_csv.csv'
        self.test_path = '/home/user/ZN/intrusion detection/data/test_csv.csv'
        self.epochs = 100
        self.choose_model = model
        self.model_choice = name
        self.save_dir = f'/home/user/ZN/intrusion detection/history_more'

    def data_pre_process(self):
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)

        train_label = train_data['Label']
        label_list = list(set(tuple(train_label)))
        label_map = {value: str(key) for key, value in enumerate(label_list)}
        for key, value in label_map.items():
            if key == 'Benign':
                label_map[key] = 0
            else:
                label_map[key] = 1
        train_data['Label'] = train_data['Label'].map(label_map)
        test_data['Label'] = test_data['Label'].map(label_map)

        train_data = train_data.dropna(axis=0, how='any')
        test_data = test_data.dropna(axis=0, how='any')
        x_train_inf = np.isinf(train_data)
        train_data[x_train_inf] = 0
        x_test_inf = np.isinf(test_data)
        test_data[x_test_inf] = 0
        train_data = np.array(train_data, dtype='float32')
        test_data = np.array(test_data, dtype='float32')

        x_train = train_data[:, 0:77]
        x_test = test_data[:, 0:77]
        y_train = train_data[:, 78]
        y_test = test_data[:, 78]

        x_train = np.array(x_train, dtype='float32')
        x_test = np.array(x_test, dtype='float32')
        y_train = np.array(y_train, dtype='float32')
        y_test = np.array(y_test, dtype='float32')
        scaler = Normalizer().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        return x_train, x_test, y_train, y_test

    def normalist(self, data, column_name):
        df_column = data[column_name]
        min = df_column.min()
        max = df_column.max()
        df_column = (df_column - min) / (max - min)
        return df_column

    def data_reshape(self, name, x_train, x_test):
        if name == 'gru':
            x_train_re = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test_re = np.reshape(x_test, (x_test.shape[0], 1, x_train.shape[1]))
        elif name == 'lstm':
            x_train_re = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test_re = np.reshape(x_test, (x_test.shape[0], 1, x_train.shape[1]))
        elif name == 'cnn':
            x_train_re = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test_re = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        else:
            x_train_re = x_train
            x_test_re = x_test
        return x_train_re, x_test_re

    def model_fit(self):
        model = self.choose_model
        x_train, x_test, y_train, y_test = self.data_pre_process()
        x_train, x_test = self.data_reshape(self.model_choice, x_train, x_test)
        print(x_train.shape, y_train.shape)
        # count = 0
        # count1 = 0
        # for i in range(len(y_train)):
        #     if y_train[i] == 0:
        #         count += 1
        #     else:
        #         count1 += 1
        # print(count, count1)

        # val_x_train = x_train[:50000]
        # par_x_train = x_train[50000:]
        # val_y_train = y_train[:50000]
        # par_x_train = y_train[50000:]

        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=512, validation_split=0.2)
        data_record = DataRecord()
        data_record.model = model
        data_record.summary = model.to_yaml()
        data_record.history = history
        data_record.epochs = self.epochs
        self.result_save(data_record)

    def result_save(self, data_record):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        path = os.path.join(self.save_dir, f'{self.model_choice}')
        if not os.path.exists(path):
            os.mkdir(path)
        data_record.model.save(os.path.join(path, 'model.h5'))
        with open(os.path.join(path, 'history.json'), 'w') as f:
            datas = {}
            for key, value in data_record.history.history.items():
                datas[key] = [float(i) for i in value]
            f.write(json.dumps(datas))
        with open(os.path.join(path, 'summary_epochs.test'), 'w') as f:
            f.write(data_record.summary)
            f.write('\n\n')
            f.write('epochs:')
            f.write(str(data_record.epochs))

    def run(self):
        start_time = time.strftime('%Y-%m-%d: %H-%M-%S', time.localtime())
        start = time.time()
        self.model_fit()
        end = time.time()
        end_time = time.strftime('%Y-%m-%d: %H-%M-%S', time.localtime())
        print('start time ---------', start_time)
        print('end time -----------', end_time)
        print(f'cost time ---------:{end - start}')


if __name__ == '__main__':
    models_build = ModelBuild()
    TrainModels(models_build.model_lstm, 'LSTM').run()
    # TrainModels(models_build.model_gru, 'GRU').run()
    # TrainModels(models_build.model_dnn, 'DNN').run()
    # TrainModels(models_build.model_cnn, 'CNN').run()
