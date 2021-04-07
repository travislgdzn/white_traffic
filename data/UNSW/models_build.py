import tensorflow as tf


class ModelsBuild:
    @property
    def model_simple(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(196,)))
        model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=[tf.keras.metrics.binary_accuracy])
        model.summary()
        return model

    @property
    def model_lstm(self):
        """
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(128, input_dim=196, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.1))
        for i in range(4):
            model.add(tf.keras.layers.LSTM(128, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.LSTM(128, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=[tf.keras.metrics.binary_accuracy])
        model.summary()

        return model

    @property
    def model_gru(self):
        """
        以下是GRU 模型，
        该模型使用tensorflow 的　keras API 接口信息，
        模型的第一层是ＧＲＵ 层
            设置输入的数据的宽度是196 ,
            该层输出的神经元数量是128
            return_sequences=True
            设置返回全部time step 的 hidden state值
        第二层是　Dropout 层
            该层主要是为了防止模型训练过程中过拟合所设计的
            该层每次随机失活0.1（３０％） 的数据量来保证模型的拟合程度
            该方法是比较简单有效的防止模型过拟合的方式
        接下去每２层都是重复上述两层结构
        模型最后一层　Dense:
            在模型的最后一层是一个全连接层，该层用作是模型的输出层
            输出一个一维数组，长度为２
            该层激活函数选择softmax 激活

        模型编译：
        优化器函数：
            adam 优化算法，该算法是自适应算法，
            自动调节学习速率，可以有效避免模型陷入
            局部最小值，同时也保证了一定的学习速率，
            也不会由于学习速率过大导致模型在最小值附件跳跃
        损失函数:
            选用标签交叉熵损失函数
        评估指标：
            标签精度算法

        model.summary() 打印模型的结构信息，
        包括每层的学习参数

        """
        model = tf.keras.models.Sequential()
        """"""
        model.add(tf.keras.layers.GRU(128, input_dim=196, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.1))
        for i in range(4):
            model.add(tf.keras.layers.GRU(128, return_sequences=True))
            model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.GRU(128, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=[tf.keras.metrics.binary_accuracy])
        model.summary()

        return model

    @property
    def model_cnn_lstm(self):
        """
        cnn_lstm　模型
        """

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=(1, 196)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Conv1D(64, 1, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Conv1D(32, 1, activation='relu'))
        model.add(tf.keras.layers.LSTM(32, input_shape=(None, 32), return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.LSTM(32, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=[tf.keras.metrics.binary_accuracy])

        model.summary()

        return model

    @property
    def model_cnn_gru(self):
        """
        cnn_gru　模型
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(128, 1, activation='relu', input_shape=(1, 196)))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Conv1D(64, 1, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Conv1D(32, 1, activation='relu'))
        model.add(tf.keras.layers.GRU(32, input_shape=(None, 32), return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.GRU(32, return_sequences=False))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=[tf.keras.metrics.binary_accuracy])

        model.summary()

        return model


if __name__ == '__main__':
    m = ModelsBuild()
    print(m.model_simple)
    print(m.model_cnn_lstm)
    print(m.model_cnn_gru)
    print(m.model_gru)
    print(m.model_lstm)
