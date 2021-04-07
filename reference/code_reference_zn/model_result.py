import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import confusion_matrix
from model_train import TrainModels
from sklearn.preprocessing import label_binarize

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

tf.compat.v1.disable_eager_execution()


class DataSet(object):
    datas: list
    labels: list

class ShowResults(object):
    def __init__(self, dir_name):
        self.epochs = 100
        dir = f'/home/user/ZN/intrusion detection/{dir_name}'
        self.save_dir = os.path.join(dir)
        self.train_path = '/home/user/ZN/intrusion detection/data/train_csv.csv'
        self.test_path = '/home/user/ZN/intrusion detection/data/test_csv.csv'
        self.save_adv_dir = '/home/user/ZN/intrusion detection/history_adv_more'
        self.save_pre_adv_dir = '/home/user/ZN/intrusion detection/history_pre_adv_more'

    def data_pre_process(self, name):
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)
        test_label = test_data['Label']
        label_list = list(set(tuple(test_label)))
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
        train_data = np.array(train_data, dtype='float32')
        test_data = np.array(test_data, dtype='float32')
        x_train_inf = np.isinf(train_data)
        train_data[x_train_inf] = 0
        x_test_inf = np.isinf(test_data)
        test_data[x_test_inf] = 0

        x_train = train_data[:, 0:77]
        x_test = test_data[:, 0:77]
        y_train = train_data[:, 78]
        y_test = test_data[:, 78]
        scaler = Normalizer().fit(x_test)
        x_test = scaler.transform(x_test)
        # y_test = to_categorical(y_test)
        if name == 'LSTM':
            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
        elif name == 'GRU':
            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
        elif name == 'CNN':
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_train, x_test, y_train, y_test


    def generate_adv_samples(self, model_path, name):
        name = name
        model = models.load_model(model_path)
        x_train, x_test, y_train, y_test = self.data_pre_process(name)
        classifier = KerasClassifier(clip_values=(np.min(x_train), np.max(x_train)), model=model)
        fgsm = FastGradientMethod(classifier, eps=0.01, minimal=True, eps_step=0.01, num_random_init=35,
                                  targeted=False, batch_size=128)
        x_adv_test = fgsm.generate(x=x_test)
        return x_adv_test


    def get_files_path(self, flag=''):
        path = []
        for _dir in os.listdir(self.save_dir):
            _dir_path = os.path.join(self.save_dir, _dir)
            for file in os.listdir(_dir_path):
                if file.endswith(flag):
                    path.append(os.path.join(self.save_dir, _dir, file))
        return path

    def show_history(self, acc=False):
        for path in self.get_files_path('json'):
            self.history_info(path, acc=acc)

    def history_info(self, path, skip=2, acc=False):
        if acc == True:
            train_flag = 'binary_accuracy'
            val_flag = 'val_binary_accuracy'
            plt.axis([0, self.epochs, 0.7, 1])
        else:
            train_flag = 'loss'
            val_flag = 'val_loss'
            plt.axis([0, self.epochs, 0, 0.5])
        with open(path) as f:
            data = json.loads(f.read())
        train_acc = data[train_flag]
        val_acc = data[val_flag]
        epochs = range(1, len(train_acc) + 1)
        plt.plot(epochs[::skip], train_acc[::skip], 'b', label='train_history')
        plt.plot(epochs[::skip], val_acc[::skip], 'r:', label='val_history')
        # plt.axis([1, 100, 0.7, 1])
        plt.xlabel('epochs')
        plt.ylabel(train_flag)
        plt.grid()
        plt.title(path.split('/')[-2])

        plt.legend()
        plt.show()

    def one_hot(self, index, num=2):
        data = np.zeros(2)
        data[index] = 1
        return data

    def roc(self, model_path, name):
        model = models.load_model(model_path)
        name = name
        print(name)
        x_train, x_test, y_train, y_test = self.data_pre_process(name)
        x_adv_test = self.generate_adv_samples(model_path, name)
        predicts = model.predict(x_test, verbose=1)
        predicts_adv = model.predict(x_adv_test, verbose=1)
        predicts_class = model.predict_classes(x_test, verbose=1)
        predicts_adv_class = model.predict_classes(x_adv_test, verbose=1)
        # for i in range(len(predicts)):
        #     max_value = max(predicts[i])
        #     for j in range(len(predicts[i])):
        #         if max_value == predicts[i][j]:
        #             predicts[i][j] = 1
        #         else:
        #             predicts[i][j] = 0
        # print(x_test.shape, y_test.shape, predicts.shape)
        predicts = np.reshape(predicts, (y_test.shape[0],))
        predicts_adv = np.reshape(predicts_adv, (y_test.shape[0],))
        self.printMetrics(y_test, predicts_class)
        print('------------------------------------')
        self.printMetrics(y_test, predicts_adv_class)
        # print(predicts)
        # auc_results = metrics.roc_auc_score(y_true=y_test, y_score=predicts)
        # print(f'model{model_path}------------')
        # print(f'auc score {auc_results}')

        fpr, tpr, threshold = metrics.roc_curve(y_test, predicts)
        fpr_adv, tpr_adv, threshold_adv = metrics.roc_curve(y_test, predicts_adv)
        print('fpr:', fpr)
        print('fpr_adv', fpr_adv)
        print('threshold', threshold)
        print('threshold_adv', threshold_adv)
        return fpr, tpr, fpr_adv, tpr_adv

    def printMetrics(self, true, pred):
        # print(true[0], pred[0])
        # 分类正确率 true:真实标签 pred:训练标签  normalize: True:分类正确的百分比； False:分类正确的样本数
        # kappa = cohen_kappa_score(true, pred)
        # ham_distance = hamming_loss(true, pred)
        # print('ham_distanc:', ham_distance)
        # print('kappa:', kappa)
        print('Accuracy:', accuracy_score(true, pred))
        # 计算精度precision = TP/(TP+FP) TP:真正例 FP：假正例 average:评价值的平均值的计算方式。可以接收[None, 'binary'(default),'micro','macro','samples','weighted']
        # 'weighted': 相当于类间带权重。各类别的P x 该类别的样本数量（实际值非预测值）/除以样本总数
        print('Precision:', precision_score(true, pred, average='weighted'))
        # 召回率：提取出的正确信息的条数/样本中的信息条数。就是所有准备的条目有多少被检索出来了。
        print('Recall:', recall_score(true, pred, average='weighted'))
        # f1_socre = 2 x (precision x recall)/(precision + recall)
        print('F1-score:', f1_score(true, pred, average='weighted'))
        # 混淆矩阵：
        # 1>列代表预测的类别；行代表实际的类别
        # 2>对角线上的值表示预测正确的数量/比例；非对角线元素是预测错误的部分
        # 混淆矩阵对角线上的值越高越好。在各分类数据的数量不平衡的情况下， 混淆矩阵可以直观的显示分类模型对应各个类别的准确率
        # print('Confusion Matrix:')
        # print(confusion_matrix(true, pred))

    # def metric_infos(self, model_path):
    #     model = models.load_model(model_path)
    #     x_test, y_test = self.data_pre_process()
    #     predicts = model.predict(x_test)
    #     predicts = np.reshape(predicts, (99773,))
    #     tp = 0
    #     tn = 0
    #     fp = 0
    #     fn = 0
    #     threshold = 0.5
    #     for i in range(len(predicts)):
    #         if predicts[i] >= threshold:
    #             if y_test[i] == 0.:
    #                 fp += 1
    #             else:
    #                 tp += 1
    #         else:
    #             if y_test[i] == 0.:
    #                 tn += 1
    #             else:
    #                 fn += 1
    #     acc = (tp + tn) / (tp + tn + fp + fn)
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #     f1_measure = 2 * tp / (2 * tp + fp + fn)
    #     print(model_path)
    #     print('acc          :', acc)
    #     print('precision    :', precision)
    #     print('recall       :', recall)
    #     print('f1_measure   :', f1_measure)
    #     print()

    def show_models_metrics(self):
        types = ['y', 'b', 'g', 'r', 'm', 'black', 'sandybrown', 'deepskyblue', 'purple']
        plt.plot([0, 1], [0, 1], 'k--')
        for model_path in self.get_files_path('h5'):
            start = time.time()
            label = model_path.split('/')[-2]
            x_labels, y_labels, x_labels_adv, y_labels_adv = self.roc(f'{self.save_dir}/{label}/model.h5', label)
            x_adv_labels, y_adv_labels, x_adv_labels_adv, y_adv_labels_adv = self.roc(f'{self.save_adv_dir}/{label}/model.h5', label)
            x_pre_labels, y_pre_labels, x_pre_labels_adv, y_pre_labels_adv = self.roc(f'{self.save_pre_adv_dir}/{label}/model.h5', label)
            # self.metric_infos(model_path)
            # plt.plot(x_labels, y_labels, types.pop(), label='normal_sample')
            plt.plot(x_labels_adv, y_labels_adv, types.pop(), label='nromal_model')
            plt.plot(x_adv_labels_adv, y_adv_labels_adv, types.pop(), label='adv_model')
            plt.plot(x_pre_labels_adv, y_pre_labels_adv, types.pop(), label='pre_adv_mdoel')
            plt.title(model_path.split('/')[-2])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            end = time.time()
            print(f'cost time: {end-start}')
            print('--------------------------------------')
            plt.legend()
            plt.show()

    def run(self):
        self.show_models_metrics()
        # self.show_history()
        # self.show_history(acc=True)
        # self.show_metrics()

if __name__ == '__main__':
    ShowResults('history_more').run()

