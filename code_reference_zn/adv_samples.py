import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf

from tensorflow.keras import models

from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score


import art
from art.classifiers import KerasClassifier
from art.attacks.evasion import FastGradientMethod, SaliencyMapMethod, BasicIterativeMethod
from art.defences.preprocessor import SpatialSmoothing

tf.compat.v1.disable_eager_execution()


class AdvModel(object):

    def __init__(self, model_name, attack, defences=False):
        self.defence = defences
        self.attack = attack
        self.model_name = model_name
        self.epochs = 100
        self.save_dir = '/home/user/ZN/intrusion detection/history_more'
        self.model_path = f'/home/user/ZN/intrusion detection/history_more/{self.model_name}/model.h5'
        self.test_data = pd.read_csv('/home/user/ZN/intrusion detection/data/test_csv.csv')
        self.train_data = pd.read_csv('/home/user/ZN/intrusion detection/data/train_csv.csv')

    def data_pre_process(self):
        test_data = self.test_data
        train_data = self.train_data
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
        # print(train_data['Label'])

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
        if self.model_name == 'gru':
            x_train_re = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test_re = np.reshape(x_test, (x_test.shape[0], 1, x_train.shape[1]))
        elif self.model_name == 'lstm':
            x_train_re = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test_re = np.reshape(x_test, (x_test.shape[0], 1, x_train.shape[1]))
        elif self.model_name == 'cnn':
            x_train_re = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test_re = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        else:
            x_train_re = x_train
            x_test_re = x_test
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        return x_train_re, x_test_re, y_train, y_test

    def get_files_path(self, flag=''):
        path = []
        for _dir in os.listdir(self.save_dir):
            _dir_path = os.path.join(self.save_dir, _dir)
            for file in os.listdir(_dir_path):
                if file.endswith(flag):
                    path.append(os.path.join(self.save_dir, _dir, file))
        return path

    def generate_adv_samples(self):
        model = models.load_model(self.model_path)
        x_train, x_test, y_train, y_test = self.data_pre_process()
        classifier = KerasClassifier(clip_values=(np.min(x_train), np.max(x_train)), model=model)
        if self.attack == 'fgsm':
            fgsm = FastGradientMethod(classifier, eps=0.01, minimal=True, eps_step=0.01, num_random_init=35,
                                      targeted=False, batch_size=128)
            x_adv_test = fgsm.generate(x=x_test)
            if self.defence == True:
                return x_test, x_adv_test, y_test
            else:
                return x_test, x_adv_test, y_test

        # elif self.attack == 'jsma':
        #     jsma = SaliencyMapMethod(classifier, batch_size=32)
        #     print(x_test.shape, y_test.shape)
        #     x_adv_test_re = jsma.generate(x=x_test, y=None)
        #     print(x_test.shape, x_adv_test_re.shape, y_test.shape)
        #     return x_test, x_adv_test_re, y_test

    def predict(self):
        model = models.load_model(self.model_path)
        x_test, x_adv_test, y_test = self.generate_adv_samples()
        predicts_normal = model.predict_classes(x_test, verbose=1)
        predicts_adv = model.predict_classes(x_adv_test, verbose=1)
        predicts_normal = np.reshape(predicts_normal, (y_test.shape[0],))
        predicts_adv = np.reshape(predicts_adv, (y_test.shape[0],))
        return predicts_normal, predicts_adv, y_test

    def printMetrics(self, true, pred):
        print('Accuracy:', accuracy_score(true, pred))
        print('Precision:', precision_score(true, pred, average='weighted'))
        print('Recall:', recall_score(true, pred, average='weighted'))
        print('F1-score:', f1_score(true, pred, average='weighted'))

    def adversarialFeatures(self):
        feats = dict()
        total = 0
        actual, adversarial, y_test = self.generate_adv_samples()
        data = self.test_data
        orig_attack = actual - adversarial
        for i in range(0, orig_attack.shape[0]):
            ind = np.where(orig_attack[i, :] != 0)[0]
            total += len(ind)
            for j in ind:
                if j in feats:
                    feats[j] += 1
                else:
                    feats[j] = 1
        print('Number of unique features changed:', len(feats.keys()))
        print('Number of average features changed per datapoint', total / len(orig_attack))
        top_10 = sorted(feats, key=feats.get, reverse=True)[:10]
        top_20 = sorted(feats, key=feats.get, reverse=True)[:20]
        print('Top ten feature：', data.columns[top_10])
        top_10_val = [100 * feats[k] / y_test.shape[0] for k in top_10]
        top_20_val = [100 * feats[k] / y_test.shape[0] for k in top_20]
        plt.bar(np.arange(20), top_20_val, align='center')
        # 用data.columns[top_20]替换轴的下标， rotataion: 替换的下标的旋转的角度
        plt.xticks(np.arange(20), data.columns[top_20], rotation='vertical')
        plt.title('Feature participation in adversarial examples')
        plt.ylabel('Precentage(%)')
        plt.xlabel('Feature')

    def show_models_metrics(self):
        types = ['r', 'b']
        predict_normal, predict_adv, y_test = self.predict()
        plt.plot([0, 1], [0, 1], 'k--')
        for i in [predict_normal, predict_adv]:
            start = time.time()
            x_labels, y_labels, threshold = metrics.roc_curve(y_test, i)
            plt.plot(x_labels, y_labels, types.pop(), label=self.model_name)
            self.printMetrics(y_test, i)
            # self.adversarialFeatures()
            end = time.time()
            print(f'cost time: {end - start}')
            print('--------------------------------------')
        plt.legend()
        plt.show()

    def run(self):
        self.show_models_metrics()


if __name__ == '__main__':
    AdvModel('lstm', 'fgsm', defences=True).run()
    # AdvModel('gru', 'fgsm').run()
    # AdvModel('cnn', 'fgsm').run()
    # AdvModel('dnn', 'fgsm').run()
