# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# coding=utf-8
import pymongo
import os
import traceback
import tensorflow as tf
import time
import json
import numpy as np
from models_build import ModelsBuild

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
        """
        该函数配置TrainModels　类的基本信息，
        包括数据库的表信息，模型的选择
        """
        client = pymongo.MongoClient()
        db = client['UNSW_show_1']
        self.train_table = db['train_normal']
        self.test_table = db['test_normal']
        self.epochs = 5
        self.choose_model = model
        self.model_choce = name
        self.save_dir = f'./history_more_{self.epochs}'

    def load_split_data(self, table, size=5120):
        datas = []
        labels = []
        for hit in pymongo.cursor.Cursor(table, batch_size=1000)[:size]:
            hit.pop('_id')
            hit.pop('id')
            hit.pop('attack_cat')
            labels.append(hit.pop('label'))
            # labels.append(hit.pop('label'))
            data = [v for k, v in hit.items()]
            data = np.array(data, dtype='float32')
            datas.append(data)
            if len(data) != 196:
                raise Exception('data shape error')
        datas = np.array(datas, dtype='float32')
        labels = np.array(labels)
        datas = np.asarray(datas).astype('float32')
        labels = np.asarray(labels).astype('float32')
        data_set = DataSet()
        # print(datas.shape)
        if self.model_choce != 'simple':
            datas = np.reshape(datas, (datas.shape[0], 1, datas.shape[1]))
        print('shape of datas', datas.shape)
        print('shape of labels ', labels.shape)
        data_set.datas = datas
        data_set.labels = labels
        return data_set

    def model_fit(self):
        model = self.choose_model
        train_ds = self.load_split_data(self.train_table, 1000000)
        train_datas = train_ds.datas
        train_labels = train_ds.labels

        history = model.fit(train_datas, train_labels,
                            epochs=self.epochs, batch_size=128,
                            validation_split=0.5)
        data_record = DataRecord()
        data_record.model = model
        data_record.summary = model.to_yaml()
        data_record.history = history
        data_record.epochs = self.epochs
        # data_record.times = i
        self.result_save(data_record)

    def result_save(self, data_record):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        # name = time.strftime('%mm%dd_%Hh%Mm%Ss', time.localtime(time.time()))
        path = os.path.join(self.save_dir, f'{self.model_choce}')
        if not os.path.exists(path):
            os.mkdir(path)
        data_record.model.save(os.path.join(path, 'model.h5'))
        with open(os.path.join(path, 'history.json'), 'w') as f:
            # print(data_record.history.history)
            datas = {}
            for key, value in data_record.history.history.items():
                datas[key] = [float(i) for i in value]
            f.write(json.dumps(datas))
        with open(os.path.join(path, 'summary_epochs.txt'), 'w') as f:
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
        print(f'cost time ---------:{end-start}')


if __name__ == '__main__':
    models_build = ModelsBuild()
    TrainModels(models_build.model_gru, 'gru').run()  # 12s per epoch
    TrainModels(models_build.model_cnn_gru, 'cnn_gru').run()    # 7.5s per epoch
    TrainModels(models_build.model_lstm, 'lstm').run()  # 13s per epoch
    TrainModels(models_build.model_cnn_lstm, 'cnn_lstm').run()  # 8s per epoch
    TrainModels(models_build.model_simple, 'simple').run()  # 3s per epoch
