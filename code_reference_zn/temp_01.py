import numpy as np
import pandas as pd
import csv
import art

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Convolution1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.svm import SVC

from art.attacks.evasion import FastGradientMethod, SaliencyMapMethod, BasicIterativeMethod
from art.classifiers import KerasClassifier



Benign = pd.read_csv('./data/IDS2018/Benign.csv', low_memory=False)
Attack = pd.read_csv('./data/IDS2018/Attack.csv', low_memory=False)
# 在Benign 和 Attack 中随机抽取行np.random.randint(start, end, number)
r_benign = np.random.randint(0, 6000000, 3000)
r_attack = np.random.randint(0, 2000000, 1000)
new_benign = Benign.iloc[r_benign, :]
new_attack = Attack.iloc[r_attack, :]
# 合并&打乱
res = pd.concat([new_benign, new_attack], axis=0, ignore_index=True)
data = res.sample(frac=1).reset_index(drop=True)
# 将['Timestamp']列的时间改成时间戳
dtime = pd.to_datetime(data['Timestamp'])
v = (dtime.values - np.datetime64('1970-01-01T08:00:00Z')) / np.timedelta64(1, 'ms')
data['Timestamp'] = v
# 时间戳转化成时间
# data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='ms', origin=pd.Timestamp('2015-01-01 08:00:00'))
data.pop('Unnamed: 0')
train_csv = data.iloc[1000:]
# trian_csv_label = train_csv.pop('Label')
test_csv = data.iloc[:1000]
# test_csv_label = test_csv.pop('Label')
train_csv.to_csv('./test_data/train_csv.csv')
# trian_csv_label.to_csv('./test_data/train_csv_label.csv')
test_csv.to_csv('./test_data/test_csv.csv')
# test_csv_label.to_csv('./test_data/test_csv_label.csv')

# ---------------------------------------------------------------------------------------
with open("./test_data/train_csv.csv", 'r') as f:
    lines = csv.reader(f)
    lines = [i for i in lines]
titles = lines.pop(0)
titles.pop(0)

train_data = pd.read_csv('./test_data/train_csv.csv')
test_data = pd.read_csv('./test_data/test_csv.csv')
x_train = pd.read_csv('./test_data/train_csv.csv')
y_train = x_train.pop('Label')
x_test = pd.read_csv('./test_data/test_csv.csv')
y_test = x_test.pop('Label')
train_data.pop('Unnamed: 0')
test_data.pop('Unnamed: 0')
x_train.pop('Unnamed: 0')
x_test.pop('Unnamed: 0')
label = list(set(tuple(y_train)))
label_map = {str(key): value for key, value in enumerate(label)}
# print(label)
# print(label_map)
for i in range(len(y_train)):
    y_train[i] = [key for key, value in label_map.items() if value == y_train[i]]
    y_train[i] = y_train[i][0]
for i in range(len(y_test)):
    y_test[i] = [key for key, value in label_map.items() if value == y_test[i]]
    y_test[i] = y_test[i][0]
# print(y_train)
# print(y_test)
# print('---------------------')


# 去掉nan
x_train = x_train.dropna(axis=0, how='any')
x_test = x_test.dropna(axis=0, how='any')
# 去掉inf
x_train_inf = np.isinf(x_train)
x_train[x_train_inf] = 0
x_test_inf = np.isinf(x_test)
x_test[x_test_inf] = 0

print(x_train.shape)
print(x_test.shape)
# print(y_train)
print('-------------------------------')

x_train = np.array(x_train, dtype='float64')
x_test = np.array(x_test, dtype='float64')
y_train = np.array(y_train, dtype='float64')
y_test = np.array(y_test, dtype='float64')

# 正则化，对测试集和训练集进行拟合和转换
scaler = Normalizer().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x_train)
print('-------------------------------')
# print(x_test)

# 将Label ont-hot
y_train_oh = to_categorical(y_train)
y_test_oh = to_categorical(y_test)
# print(y_train_oh)
# print(y_test_oh)

# ----------------------------------------------------

x_train_lstm = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test_lstm = np.reshape(x_test, (x_test.shape[0], 1, x_train.shape[1]))
# print(x_train_lstm.shape)

# ----------------------------------------------------

x_train_cnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test_cnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
print(x_train_cnn.shape)

# ----------------------------------------------------

def printMetrics(true, pred):
    # 分类正确率 true:真实标签 pred:训练标签  normalize: True:分类正确的百分比； False:分类正确的样本数
    print('Accuracy:', accuracy_score(true, pred))
    # 计算精度precision = TP/(TP+FP) TP:真正例 FP：假正例 average:评价值的平均值的计算方式。可以接收[None, 'binary'(default),'micro','macro','samples','weighted']
    # 'weighted': 相当于类间带权重。各类别的P x 该类别的样本数量（实际值非预测值）/除以样本总数
    print('Precision:', precision_score(true, pred, average='weighted'))
    # 召回率：提取出的正确信息的条数/样本中的信息条数。就是所有准备的条目有多少被检索出来了。
    print('Recall:', recall_score(true, pred, average='weighted'))
    # f1_socre = 2 x (precision x recall)/(precision + recall)
    print('F1-score:', f1_score(true ,pred, average='weighted'))
    # 混淆矩阵：
    # 1>列代表预测的类别；行代表实际的类别
    # 2>对角线上的值表示预测正确的数量/比例；非对角线元素是预测错误的部分
    # 混淆矩阵对角线上的值越高越好。在各分类数据的数量不平衡的情况下， 混淆矩阵可以直观的显示分类模型对应各个类别的准确率
    print('Confusion Matrix:')
    print(confusion_matrix(true, pred))

def adversarialFeatures(actual, adversarial, data):
    feats = dict()
    total = 0
    orig_attack = actual - adversarial
    for i in range(0, orig_attack.shape[0]):
        ind = np.where(orig_attack[i, :] != 0)[0]
        # 求攻击类别总数
        total += len(ind)
        # 每种攻击类别的数量
        for j in ind:
            if j in feats:
                feats[j] += 1
            else:
                feats[j] = 1
    print('Number of unique features changed:', len(feats.keys()))
    print('Number of average features changed per datapoint', total/len(orig_attack))

    # sorted(iterable, cmp=None, key=None reverse=False)
    # iterable:可迭代对象 cmp:比较参数，参数的值从迭代对象中取出，大于返回1，小于返回—1,等于返回0
    # key:主要用来比较的元素  reverse: True:降序 False:升序（默认）
    top_10 = sorted(feats, key=feats.get, reverse=True)[:10]
    top_20 = sorted(feats, key=feats.get, reverse=True)[:20]

    print('Top ten feature：',data.columns[top_10])
    top_10_val = [100* feats[k] / y_test.shape[0] for k in top_10]
    top_20_val = [100 * feats[k] / y_test.shape[0] for k in top_20]

    # plt.bar(x, height, width=0.8, botton=None, align='center', data=None,**kwargs) 柱形图
    # x(np.arange(20)):20条柱子, height(top_20_val):每条柱的高度, width:宽度，默认0.8, align: 条形的中心位置
    plt.bar(np.arange(20), top_20_val, align='center')
    # 用data.columns[top_20]替换轴的下标， rotataion: 替换的下标的旋转的角度
    plt.xticks(np.arange(20), data.columns[top_20], rotation='vertical')
    plt.title('Feature participation in adversarial examples')
    plt.ylabel('Precentage(%)')
    plt.xlabel('Feature')

# -----------------------------------------------------
# 训练模型

dnnmodel = Sequential()
dnnmodel.add(Dense(1024, input_dim = 79, activation='relu'))
# 1>加快收敛速度 2>选择更小的L2正则约束参数，提高网络泛华能力 3>可以把训练数据彻底打乱
dnnmodel.add(BatchNormalization())
# 正则化，随机失活
dnnmodel.add(Dropout(0.01))
dnnmodel.add(Dense(768, activation='relu'))
dnnmodel.add(BatchNormalization())
dnnmodel.add(Dropout(0.01))
dnnmodel.add(Dense(512,activation='relu'))
dnnmodel.add(BatchNormalization())
dnnmodel.add(Dropout(0.01))
dnnmodel.add(Dense(256,activation='relu'))
dnnmodel.add(BatchNormalization())
dnnmodel.add(Dropout(0.01))
dnnmodel.add(Dense(128,activation='relu'))
dnnmodel.add(BatchNormalization())
dnnmodel.add(Dropout(0.01))
dnnmodel.add(Dense(10, activation='softmax'))
dnnmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
dnnmodel.summary()

cnnmodel = Sequential()
# Convolution1D(nb_filter, filter_length, init = 'glorot_uniform' , activation = None , weights = None , border_mode = 'valid' , subsample_length = 1 ,
# W_regularizer = None , b_regularizer = None , activity_regularizer = None , W_constraint = None, b_constraint = None , bias = True , input_dim = None , input_length = None)
# nb_filter: 卷积核的数量，也是输出的维度  filter_length: 每个过滤器的长度
cnnmodel.add(Convolution1D(64, 3, activation='relu', input_shape=(79,1)))
cnnmodel.add(Convolution1D(64, 3, activation='relu'))
cnnmodel.add(MaxPooling1D(pool_size=2))
cnnmodel.add(Convolution1D(128, 3, activation='relu'))
cnnmodel.add(Convolution1D(128, 3, activation='relu'))
cnnmodel.add(MaxPooling1D(pool_size=2))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(128, activation="relu"))
cnnmodel.add(Dropout(0.5))
cnnmodel.add(Dense(10, activation="softmax"))
# categorical_crossentropy：交叉熵损失函数。多输出的loss的函数。一般最后一层用 softmax 时使用这个损失函数
# adam: 利用梯度的一阶矩和二阶矩估计动态调整每个参数的学习率。 优点：每一个迭代学习率都有一个明确的范围，使得参数变化很稳定
# accuracy：真实标签和模型预测均为标量，如果真实标签序列为[1, 1, 3, 0, 2, 5]，预测序列为[1, 2, 3, 1, 2, 5]，此时可以看到命中了四个，则[accuracy] = 4/6 = 0.6667
cnnmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnnmodel.summary()

lstmmodel = Sequential()
lstmmodel.add(LSTM(64, input_dim=79, return_sequences=True))
lstmmodel.add(Dropout(0.1))
lstmmodel.add(LSTM(64, return_sequences=True))
lstmmodel.add(Dropout(0.1))
lstmmodel.add(LSTM(64, return_sequences=True))
lstmmodel.add(Dropout(0.1))
lstmmodel.add(LSTM(64, return_sequences=False))
lstmmodel.add(Dropout(0.1))
lstmmodel.add(Dense(10, activation='softmax'))
lstmmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstmmodel.summary()
lstmmodel.load_weights()

# ----------------------------------------------------------
# 在训练集上训练后生成的权重值

dnnmodel.load_weights()
cnnmodel.load_weights()
lstmmodel.load_weights()

# -----------------------------------------------------------
# 在测试集上预测

# predict: 返回数值，表示样本属于每一个类别的概率
# predict_classes：返回类别的索引，即该样本所属类别的标签
# verbose: 日志显示。 0：不在标准输出流输出日志信息；1：输出进度条记录；2：为每个epoch输出一行记录
dnnPred = dnnmodel.predict_classes(x_test_cnn, verbose=1)
printMetrics(y_test, dnnPred)

cnnPred = cnnmodel.predict_classes(x_test_cnn, verbose=1)
printMetrics(y_test, cnnPred)

lstmPred = lstmmodel.predict_classes(x_test_cnn, verbose=1)
printMetrics(y_test, lstmPred)

# ----------------------------------------------------------
# 分类器
classifier_dnn = KerasClassifier(clip_values=(np.min(x_train), np.max(x_train)), model=dnnmodel)
classifier_lstm = KerasClassifier(clip_values=(np.min(x_train), np.max(x_train)), model=lstmmodel)
classifier_cnn = KerasClassifier(clip_values=(np.min(x_train), np.max(x_train)), model=cnnmodel)
# ----------------------------------------------------------
# fgsm
fgsm = FastGradientMethod(classifier_cnn, eps=0.01, minimal=True, eps_step=0.01, num_random_init=35, targeted=False, batch_size=32)
x_test_adv_fgsm = fgsm.generate(x=x_test)

# jsma
jsma = SaliencyMapMethod(classifier_lstm, batch_size=32)
x_test_adv_jsma=jsma.generate(x=x_test)

# 显示正常的测试集合被攻击修改后的测试集的区别
adversarialFeatures(x_test, x_test_adv_fgsm,test_data)
adversarialFeatures(x_test, x_test_adv_jsma, test_data)

# 显示dnn网络在对抗攻击下的表现
dnnPredfgsm = dnnmodel.predict_classes(x_test_adv_fgsm, verbose=1)
dnnPredjsma = dnnmodel.predict_classes(x_test_adv_jsma, verbose=1)
printMetrics(y_test, dnnPredfgsm)
printMetrics(y_test, dnnPredjsma)

# 显示cnn网络在对抗攻击下的表现
x_test_adv_cnn_fgms = np.array(x_test_adv_fgsm, (x_test_adv_fgsm[0], x_test_adv_fgsm[1], 1))
x_test_adv_cnn_jsma = np.array(x_test_adv_jsma, (x_test_adv_fgsm[0], x_test_adv_fgsm[1], 1))
cnnPredfgsm = cnnmodel.predict_classes(x_test_adv_cnn_fgms, verbose=1)
cnnPredjsma = cnnmodel.predict_classes(x_test_adv_cnn_jsma, verbose=1)
printMetrics(y_test, cnnPredfgsm)
printMetrics(y_test, cnnPredjsma)

# 显示lstm网络在对抗攻击下的表现
x_test_adv_lstm_fgms = np.array(x_test_adv_fgsm, (x_test_adv_fgsm[0], 1, x_test_adv_fgsm[1]))
x_test_adv_lstm_jsma = np.array(x_test_adv_jsma, (x_test_adv_fgsm[0], 1, x_test_adv_fgsm[1]))
lstmPredfgsm = cnnmodel.predict_classes(x_test_adv_lstm_fgms, verbose=1)
lstmPredjsma = cnnmodel.predict_classes(x_test_adv_lstm_jsma, verbose=1)
printMetrics(y_test, lstmPredfgsm)
printMetrics(y_test, lstmPredjsma)

def advEval(DL):
    print('\nFGSM\n')
    printMetrics(y_test, DL.predict(x_test_adv_fgsm))
    print('\nJSMA\n')
    printMetrics(y_test, DL.predict(x_test_adv_jsma))

# 逻辑回归
LR = LogisticRegression()
LR.fit(x_train, y_train)
LR_Predprob = LR.predict_proba(x_test)
LR_Pred = LR_Predprob.argmax(axis=1)
printMetrics(y_test, LR_Pred)
advEval(LR)

# 连续型朴素贝叶斯
GNB = GaussianNB()
GNB.fit(x_train,y_train)
GNB_Predprob=GNB.predict_proba(x_test)
GNB_Pred=GNB_Predprob.argmax(axis=1)
printMetrics(y_test, GNB_Pred)
# np.savetxt(LOG_PATH+'GaussianNB/predict_proba.txt', GNB_Predprob, fmt='%06f')
# np.savetxt(LOG_PATH+'GaussianNB/confusion_matrix.txt', confusion_matrix(y_test,GNB_Pred), fmt='%01d')
advEval(GNB)

# K近邻算法
KNN = KNeighborsClassifier()
KNN.fit(x_train,y_train)
KNN_Predprob=KNN.predict_proba(x_test)
KNN_Pred=KNN_Predprob.argmax(axis=1)
printMetrics(y_test, KNN_Pred)
# np.savetxt(LOG_PATH+'KNeighborsClassifier/predict_proba.txt', KNN_Predprob, fmt='%06f')
# np.savetxt(LOG_PATH+'KNeighborsClassifier/confusion_matrix.txt', confusion_matrix(y_test,KNN_Pred), fmt='%01d')
advEval(KNN)

#
AB = AdaBoostClassifier(n_estimators=100)
AB.fit(x_train,y_train)
AB_Predprob=AB.predict_proba(x_test)
AB_Pred=AB_Predprob.argmax(axis=1)
printMetrics(y_test, AB_Pred)
advEval(AB)

# 随机森林
RF = RandomForestClassifier(n_estimators=100)
RF.fit(x_train,y_train)
RF_Predprob=RF.predict_proba(x_test)
RF_Pred=RF_Predprob.argmax(axis=1)
printMetrics(y_test, RF_Pred)
advEval(RF)

# 线性支持向量机。 kernel：核函数
LSVM = svm.SVC(kernel='linear',probability=True)
LSVM.fit(x_train,y_train)
LSVM_Predprob=LSVM.predict_proba(x_test)
LSVM_Pred=LSVM_Predprob.argmax(axis=1)
printMetrics(y_test, LSVM_Pred)
advEval(LSVM)

# 支持向量机
RSVM = svm.SVC(kernel='rbf',probability=True)
RSVM.fit(x_train,y_train)
RSVM_Predprob=RSVM.predict_proba(x_test)
RSVM_Pred=RSVM_Predprob.argmax(axis=1)
printMetrics(y_test, RSVM_Pred)
advEval(RSVM)