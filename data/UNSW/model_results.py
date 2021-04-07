# from UNSW import __init__
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
import copy
import pymongo
import numpy as np
from sklearn import metrics
import time


class DataSet(object):
    datas: list
    labels: list


class ShowResults(object):
    def __init__(self):
        self.save_dir = os.path.join('./history_more_5')
        client = pymongo.MongoClient()
        db = client['UNSW_show_1']
        self.train_table = db['train_normal']
        self.test_table = db['test_normal']
        self.epochs = 5

    def get_files_path(self, flag=''):
        paths = []
        for _dir in os.listdir(self.save_dir):
            _dir_path = os.path.join(self.save_dir, _dir)
            for file in os.listdir(_dir_path):
                if file.endswith(flag):
                    paths.append(os.path.join(self.save_dir, _dir, file))
        return paths

    def show_history(self, acc=False):
        for path in self.get_files_path('json'):
            self.history_info(path, acc=acc)

    def history_info(self, path, skip=2, acc=False):
        if acc:
            train_flag = 'binary_accuracy'
            val_flag = 'val_binary_accuracy'
            plt.axis([0, self.epochs, 0.5, 1])
        else:
            train_flag = 'loss'
            val_flag = 'val_loss'
            plt.axis([0, self.epochs, 0, 0.5])
        with open(path) as f:
            data = json.loads(f.read())
        train_acc = data[train_flag]
        val_acc = data[val_flag]
        epochs = range(1, len(train_acc) + 1)
        plt.plot(epochs[::skip], train_acc[::skip], 'bo', label='train_history')
        plt.plot(epochs[::skip], val_acc[::skip], 'ro', label='val_history')
        plt.xlabel('epochs')
        plt.ylabel(train_flag)
        plt.grid()
        plt.title(path.split('/')[2])

        plt.legend()
        plt.show()

    def load_split_data(self, table, size=5120, model_path=None):
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
        if 'simple' not in model_path:
            datas = np.reshape(datas, (datas.shape[0], 1, datas.shape[1]))
        print('shape of datas', datas.shape)
        print('shape of labels ', labels.shape)
        data_set.datas = datas
        data_set.labels = labels
        return data_set

    def roc(self, model_path):
        model = tf.keras.models.load_model(model_path)
        data_set = self.load_split_data(self.test_table, 100000, model_path)
        labels = data_set.labels
        predicts = model.predict(data_set.datas)

        predicts = np.reshape(predicts, (82332,))
        # # f1_results = metrics.f1_score(labels, predicts)
        # # metrics
        # print(labels)
        # results = []
        # for i
        # acc_results = metrics.accuracy_score(labels, predicts)
        # pre_results = metrics.precision_score(labels, predicts, average='micro')
        # # recall_results = metrics.recall_score(labels, predicts)
        # #
        # # print(f'f1-measure  score{f1_results}')
        # # print(f'accuracy score {acc_results}')
        # print(f'precision score　{pre_results}')
        # # print(f'recall score {recall_results}')
        auc_results = metrics.roc_auc_score(y_true=labels, y_score=predicts)
        print(f'model {model_path}----')
        print(f'auc score {auc_results}')
        # tp = 0
        # tn = 0
        # fp = 0
        # fn = 0
        # y_labels = []
        # x_labels = []
        # split_predicts = copy.deepcopy(predicts)
        # split_predicts.sort()
        # for threshold in split_predicts[::10]:
        #     for i in range(len(predicts)):
        #         if predicts[i] >= threshold:
        #             if labels[i] == 0.:
        #                 fp += 1
        #             else:
        #                 tp += 1
        #         else:
        #             if labels[i] == 0.:
        #                 tn += 1
        #             else:
        #                 fn += 1
        #     if tp + fn == 0:
        #         tpr = 0
        #     else:
        #         tpr = tp / (tp + fn)
        #     if fp + tn == 0:
        #         fpr = 0
        #     else:
        #         fpr = fp / (fp + tn)
        #     # print(tpr, '----', fpr)
        #     y_labels.append(tpr)
        #     x_labels.append(fpr)
        fpr, tpr, threshold = metrics.roc_curve(labels, predicts)
        x_labels = fpr
        y_labels = tpr
        return x_labels, y_labels

    def metric_infos(self, model_path):
        model = tf.keras.models.load_model(model_path)
        data_set = self.load_split_data(self.test_table, 100000, model_path)
        labels = data_set.labels
        predicts = model.predict(data_set.datas)
        predicts = np.reshape(predicts, (82332,))
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        threshold = 0.5
        for i in range(len(predicts)):
            if predicts[i] >= threshold:
                if labels[i] == 0.:
                    fp += 1
                else:
                    tp += 1
            else:
                if labels[i] == 0.:
                    tn += 1
                else:
                    fn += 1
        acc = (tp + tn) / (tp + tn + fp + fn)
        # dr = tp / (tp + fn)
        # far = fp / (tn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_measure = 2 * tp / (2 * tp + fp + fn)
        print(model_path)
        print('acc          :', acc)
        # print('dr           :', dr)
        # print('far          :', far)
        print('precision    :', precision)
        print('recall       :', recall)
        print('f1_measure   :', f1_measure)
        print()

    def show_models_metrics(self):
        types = ['y', 'b', 'g', 'r', 'm']
        plt.plot([0, 1], [0, 1], 'k--')
        for model_path in self.get_files_path('h5'):
            start = time.time()
            x_labels, y_labels = self.roc(model_path)
            self.metric_infos(model_path)
            plt.plot(x_labels, y_labels, types.pop(), label=model_path.split('/')[-1])
            end = time.time()
            print(f'cost time: {end-start}')
            print('---------------------------------')

        plt.legend()
        plt.show()
    #
    # def show_metrics(self):
    #     for model_path in self.get_files_path('h5'):
    #         start = time.time()
    #         self.metric_infos(model_path)
    #         end = time.time()
    #         print(f'cost time: {end - start}')

    def run(self):
        self.show_models_metrics()
        # self.show_history()
        # self.show_history(acc=True)
        # self.show_metrics()


if __name__ == '__main__':
    ShowResults().run()


