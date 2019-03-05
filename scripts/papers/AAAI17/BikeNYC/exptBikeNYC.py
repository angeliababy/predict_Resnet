# -*- coding: utf-8 -*-
""" 
Usage:
    THEANO_FLAGS="device=gpu0" python exptBikeNYC.py
"""
from __future__ import print_function
import os
try:
    import cPickle as pickle    #python 2
except ImportError as e:
    import pickle   #python 3
import numpy as np
import math
import time

import sys
sys.path.append(r'../../../../../DeepST')#将路径目录添加到系统环境变量 path 下

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deepst.models.STResNet import stresnet
from deepst.config import Config
import deepst.metrics as metrics
from deepst.datasets import BikeNYC
np.random.seed(1337)  # for reproducibility
from deepst.preprocessing import MinMaxNormalization

# parameters
# data path, you may set your own data path with the global envirmental
# variable DATAPATH
DATAPATH = Config().DATAPATH
nb_epoch = 500  # 500,number of epoch at training stage
nb_epoch_cont = 100  # 100,number of epoch at training (cont) stage
batch_size = 32  # batch size
T = 24  # number of time intervals in one day

lr = 0.0002  # learning rate
len_closeness = 3  # length of closeness dependent sequence
len_period = 3  # length of peroid dependent sequence
len_trend = 3  # length of trend dependent sequence
nb_residual_unit = 4   # number of residual units

nb_flow = 1  # there are two types of flows: new-flow and end-flow
# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)
print('factor: ', m_factor)
path_result = 'RET'
path_model = 'MODEL'

# size = 10
# mmn._max=36675
# mmn._min=10297
# 数据预处理
def Preprocess():
    # 读取数据文件
    f = open("area.csv", "r")
    # 临时存储某时间的人数
    person_num = []
    # 存储各时间的人数尺寸(n,3,3)
    imgs = []

    i, l = 0, 0
    for line in f:
        l += 1
        if l == 1:
            continue
        i += 1
        line = line.strip().split(',')
        # 将人数转化为小于1的数，后面求实际人数需转化过来
        number = (float(line[2])-0) / (3073-0) * 2 - 1
        person_num.append(number)
        # 每次读16个数据
        if i % (16*8) == 0:
            # 转化成一维数组
            person_num = np.array(person_num)
            # 改变形状，类似图像形式
            person_num = person_num.reshape(16, 8)
            imgs.append(person_num)
            i = 0
            person_num = []

    # 训练数据（输入三种类型的数据，并各自转化为多通道形式）
    train_x1, train_x2, train_x3, train_y = [], [], [], []
    for i in range(484, 1300):
    # 取短期、周期、趋势三组件数据，各不同长度序列
        image1 = [imgs[i - 3], imgs[i - 2], imgs[i - 1]]
        image2 = [imgs[i - 72], imgs[i - 48], imgs[i - 24]]
        image3 = [imgs[i - 484], imgs[i - 336], imgs[i - 168]]
        train_x1.append(image1)
        train_x2.append(image2)
        train_x3.append(image3)
        lab = [imgs[i]]
        train_y.append(lab)  # 最终输出
    train_x = [np.array(train_x1), np.array(train_x2), np.array(train_x3)]
    train_y = np.array(train_y)

    # # 测试数据（输入三种类型的数据，并各自转化为多通道形式）
    # test_x1, test_x2,test_x3, test_y = [], [], [], []
    # for i in range(1200, 1300):
    #     # 取短期、周期、趋势三组件数据
    #     image1 = [imgs[i - 3], imgs[i - 2], imgs[i - 1]]
    #     image2 = [imgs[i - 72], imgs[i - 48], imgs[i - 24]]
    #     image3 = [imgs[i - 484], imgs[i - 336], imgs[i - 168]]
    #     test_x1.append(image1)
    #     test_x2.append(image2)
    #     test_x3.append(image3)
    #     lab = [imgs[i]]
    #     test_y.append(lab)  # 最终输出
    # test_x = [np.array(test_x1), np.array(test_x2), np.array(test_x3)]
    # test_y = np.array(test_y)

    return train_x, train_y


if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)


def build_model(external_dim):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = stresnet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf,
                     external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)
    return model


def main():
    # load data
    print("loading data...")
    # X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
    #     T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
    #     preprocess_name='preprocessing.pkl', meta_data=False)
    X_train, Y_train = Preprocess()

    # print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
    print(np.max(X_train),np.min(X_train),np.max(Y_train),np.min(Y_train))

    external_dim = False
    # print('=' * 10)
    # print("compiling model...")
    # print(
    #     "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")
    model = build_model(external_dim)
    hyperparams_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
        len_closeness, len_period, len_trend, nb_residual_unit, lr)
    fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))

    early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    print("training model...")
    ts = time.time()
    history = model.fit(X_train, Y_train,
                        nb_epoch=nb_epoch,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    model.save_weights(os.path.join(
        'MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
    print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

    print('=' * 10)
    print('evaluating using the model that has the best loss on the valid set')

    model.load_weights(fname_param)
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
                           0] // 48, verbose=0)
    print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
          (score[0], score[1], score[1] * (3073) / 2. * m_factor))

    # score = model.evaluate(
    #     X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    # print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

    # print('=' * 10)
    # print("training model (cont)...")
    # fname_param = os.path.join(
    #     'MODEL', '{}.cont.best.h5'.format(hyperparams_name))
    # model_checkpoint = ModelCheckpoint(
    #     fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
    # history = model.fit(X_train, Y_train, nb_epoch=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
    #                     model_checkpoint], validation_data=(X_test, Y_test))
    # pickle.dump((history.history), open(os.path.join(
    #     path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
    # model.save_weights(os.path.join(
    #     'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
    #
    # print('=' * 10)
    # print('evaluating using the final model')
    # score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[
    #                        0] // 48, verbose=0)
    # print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))
    #
    # score = model.evaluate(
    #     X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    # print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2. * m_factor))

if __name__ == '__main__':
    main()
