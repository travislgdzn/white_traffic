import tensorflow as tf
"""
该段程序是配置ＧＰＵ　设置运行程序
在运行过程中允许扩大GPU 的使用容量，允许按需增长,
防止程序运行过程中因为程序的GPU 
显存超过默认的容量导致程序运行失败
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

