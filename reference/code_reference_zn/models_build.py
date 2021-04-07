import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Convolution1D, MaxPooling1D, Flatten, LSTM, GRU, Embedding

class ModelBuild:

    @property
    def model_dnn(self):
        dnnmodel = Sequential()
        dnnmodel.add(Dense(1024, input_dim=77, activation='relu'))
        # 1>加快收敛速度 2>选择更小的L2正则约束参数，提高网络泛华能力 3>可以把训练数据彻底打乱
        dnnmodel.add(BatchNormalization())
        # 正则化，随机失活
        dnnmodel.add(Dropout(0.01))
        dnnmodel.add(Dense(768, activation='relu'))
        dnnmodel.add(BatchNormalization())
        dnnmodel.add(Dropout(0.01))
        dnnmodel.add(Dense(512, activation='relu'))
        dnnmodel.add(BatchNormalization())
        dnnmodel.add(Dropout(0.01))
        dnnmodel.add(Dense(256, activation='relu'))
        dnnmodel.add(BatchNormalization())
        dnnmodel.add(Dropout(0.01))
        dnnmodel.add(Dense(128, activation='relu'))
        dnnmodel.add(BatchNormalization())
        dnnmodel.add(Dropout(0.01))
        dnnmodel.add(Dense(1, activation='sigmoid'))
        dnnmodel.compile(loss='binary_crossentropy',
                          optimizer='Adam',
                          metrics=['binary_accuracy'])
        return dnnmodel

    @property
    def model_cnn(self):
        cnnmodel = Sequential()
        # Convolution1D(nb_filter, filter_length, init = 'glorot_uniform' , activation = None , weights = None , border_mode = 'valid' , subsample_length = 1 ,
        # W_regularizer = None , b_regularizer = None , activity_regularizer = None , W_constraint = None, b_constraint = None , bias = True , input_dim = None , input_length = None)
        # nb_filter: 卷积核的数量，也是输出的维度  filter_length: 每个过滤器的长度
        cnnmodel.add(Convolution1D(64, 3, activation='relu', input_shape=(77, 1)))
        cnnmodel.add(Convolution1D(64, 3, activation='relu'))
        cnnmodel.add(MaxPooling1D(pool_size=2))
        cnnmodel.add(Convolution1D(128, 3, activation='relu'))
        cnnmodel.add(Convolution1D(128, 3, activation='relu'))
        cnnmodel.add(MaxPooling1D(pool_size=2))
        cnnmodel.add(Flatten())
        cnnmodel.add(Dense(128, activation="relu"))
        cnnmodel.add(Dropout(0.5))
        cnnmodel.add(Dense(1, activation='sigmoid'))
        # categorical_crossentropy：交叉熵损失函数。多输出的loss的函数。一般最后一层用 softmax 时使用这个损失函数
        # adam: 利用梯度的一阶矩和二阶矩估计动态调整每个参数的学习率。 优点：每一个迭代学习率都有一个明确的范围，使得参数变化很稳定
        # accuracy：真实标签和模型预测均为标量，如果真实标签序列为[1, 1, 3, 0, 2, 5]，预测序列为[1, 2, 3, 1, 2, 5]，此时可以看到命中了四个，则[accuracy] = 4/6 = 0.6667
        cnnmodel.compile(loss='binary_crossentropy',
                          optimizer='Adam',
                          metrics=['binary_accuracy'])
        cnnmodel.summary()
        return cnnmodel

    @property
    def model_lstm(self):
        lstmmodel = Sequential()
        # lstmmodel.add(Embedding(2, 64, input_length=78))
        lstmmodel.add(LSTM(64, input_dim=77, return_sequences=True))
        lstmmodel.add(Dropout(0.5))
        for i in range(1):
            lstmmodel.add(LSTM(64, return_sequences=True))
            lstmmodel.add(Dropout(0.5))
        lstmmodel.add(LSTM(64, return_sequences=False))
        lstmmodel.add(Dropout(0.5))
        lstmmodel.add(Dense(1, activation='sigmoid'))
        lstmmodel.compile(loss='binary_crossentropy',
                          optimizer='Adam',
                          metrics=['binary_accuracy'])
        lstmmodel.summary()
        return lstmmodel

    @property
    def model_gru(self):
        grumodel = Sequential()
        grumodel.add(GRU(64, input_dim=77, return_sequences=True))
        grumodel.add(Dropout(0.1))
        grumodel.add(GRU(64, return_sequences=True))
        grumodel.add(Dropout(0.1))
        grumodel.add(GRU(64, return_sequences=True))
        grumodel.add(Dropout(0.1))
        grumodel.add(GRU(64, return_sequences=False))
        grumodel.add(Dropout(0.1))
        grumodel.add(Dense(1, activation='sigmoid'))
        grumodel.compile(loss='binary_crossentropy',
                          optimizer='Adam',
                          metrics=['binary_accuracy'])
        grumodel.summary()
        return grumodel


if __name__ == '__main__':
    m = ModelBuild()
    print(m.model_cnn)
    print(m.model_dnn)
    print(m.model_gru)
    print(m.model_lstm)
