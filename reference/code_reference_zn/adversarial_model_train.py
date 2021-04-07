import numpy as np
import pandas as pd
import os
import json
import time
from tensorflow.keras import models
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
from models_build import ModelBuild

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.data_generators import KerasDataGenerator
from art.defences.trainer import AdversarialTrainer

tf.compat.v1.disable_eager_execution()

class DataSet(object):

    datas: list
    labels: list

class DataRecord:
    model: str
    history: str
    summary: str
    epochs: int

class TrainAdvModel(object):
    def __init__(self, model, name):
        self.train_path = '/home/user/ZN/intrusion detection/data/train_csv.csv'
        self.test_path = '/home/user/ZN/intrusion detection/data/test_csv.csv'
        self.epochs = 100
        self.choose_model = model
        self.model_choice = name
        self.save_dir = f'/home/user/ZN/intrusion detection/history_more'
        self.save_adv_dir = '/home/user/ZN/intrusion detection/history_adv_more'
        self.save_pre_adv_dir = '/home/user/ZN/intrusion detection/history_pre_adv_more'

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

    def pre_adv_model_fit(self):
        model = models.load_model(os.path.join(self.save_dir, f'{self.model_choice}/model.h5'))
        x_train, x_test, y_train, y_test = self.data_pre_process()
        classifier = KerasClassifier(clip_values=(np.min(x_train), np.max(x_train)), model=model)
        fgsm = FastGradientMethod(classifier, eps=0.01, minimal=True, eps_step=0.01, num_random_init=35,
                                  targeted=False, batch_size=32)
        x_train, x_test = self.data_reshape(self.model_choice, x_train, x_test)
        x_adv_train = fgsm.generate(x=x_train)
        # x_adv_test = fgsm.generate(x=x_test)
        # adv_trainer = AdversarialTrainer(classifier, attacks=fgsm, ratio=1.0)
        # # samples = np.array(list(range(0, y_train.shape[0])))
        # # y_train = np.column_stack((samples, y_train))
        # y_train = np.reshape(y_train, (y_train.shape[0],))
        # print(y_train.shape)
        # adv_trainer.fit(x_adv_train, y_train, batch_size=128, nb_epochs=10)
        history = model.fit(x_adv_train, y_train, epochs=self.epochs, batch_size=32, validation_split=0.2)
        data_record = DataRecord()
        data_record.model = model
        data_record.summary = model.to_yaml()
        data_record.history = history
        data_record.epochs = self.epochs
        self.result_save(data_record, self.save_pre_adv_dir)

    def adv_model_fit(self):
        model = self.choose_model
        x_train, x_test, y_train, y_test = self.data_pre_process()
        classifier = KerasClassifier(clip_values=(np.min(x_train), np.max(x_train)), model=model)
        fgsm = FastGradientMethod(classifier, eps=0.01, minimal=True, eps_step=0.01, num_random_init=35,
                                  targeted=False, batch_size=128)
        x_train, x_test = self.data_reshape(self.model_choice, x_train, x_test)
        x_adv_train = fgsm.generate(x=x_train)
        history = model.fit(x_adv_train, y_train, epochs=self.epochs, batch_size=32, validation_split=0.2)
        data_record = DataRecord()
        data_record.model = model
        data_record.summary = model.to_yaml()
        data_record.history = history
        data_record.epochs = self.epochs
        self.result_save(data_record, self.save_adv_dir)

    def result_save(self, data_record, path):
        path_dir = path
        if not os.path.exists(path_dir):
            os.mkdir(path_dir)
        path = os.path.join(path_dir, f'{self.model_choice}')
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

    def printMetrics(self, true, pred):
        print('Accuracy:', accuracy_score(true, pred))
        print('Precision:', precision_score(true, pred, average='weighted'))
        print('Precision:', precision_score(true, pred, average='weighted'))
        print('Recall:', recall_score(true, pred, average='weighted'))
        print('F1-score:', f1_score(true, pred, average='weighted'))

    def run(self):
        start_time = time.strftime('%Y-%m-%d: %H-%M-%S', time.localtime())
        start = time.time()
        self.pre_adv_model_fit()
        self.adv_model_fit()
        end = time.time()
        end_time = time.strftime('%Y-%m-%d: %H-%M-%S', time.localtime())
        print('start time ---------', start_time)
        print('end time -----------', end_time)
        print(f'cost time ---------:{end - start}')


if __name__ == '__main__':
    models_build = ModelBuild()
    # TrainAdvModel(models_build.model_lstm, 'lstm').run()
    # TrainAdvModel(models_build.model_gru, 'gru').run()
    # TrainAdvModel(models_build.model_cnn, 'cnn').run()
    TrainAdvModel(models_build.model_dnn, 'dnn').run()


