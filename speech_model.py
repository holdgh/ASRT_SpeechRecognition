# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016-2099 Ailemon.net
#
# This file is part of ASRT Speech Recognition Tool.
#
# ASRT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# ASRT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASRT.  If not, see <https://www.gnu.org/licenses/>.
# ============================================================================

"""
@author: nl8590687
声学模型基础功能模板定义
"""
import os
import time
import random
import numpy as np

from utils.ops import get_edit_distance, read_wav_data
from utils.config import load_config_file, DEFAULT_CONFIG_FILENAME, load_pinyin_dict
from utils.thread_backup import threadsafe_generator


class ModelSpeech:
    """
    语音模型类

    参数：
        speech_model: 声学模型类型 (BaseModel类) 实例对象
        speech_features: 声学特征类型(SpeechFeatureMeta类)实例对象
    """

    def __init__(self, speech_model, speech_features, max_label_length=64):
        self.data_loader = None
        self.speech_model = speech_model
        self.trained_model, self.base_model = speech_model.get_model()
        self.speech_features = speech_features
        self.max_label_length = max_label_length

    @threadsafe_generator
    def _data_generator(self, batch_size, data_loader):
        """
        数据生成器函数，用于Keras的generator_fit训练
        batch_size: 一次产生的数据量
        """
        # 按照batch_size初始化标签向量，元素值为0.0
        labels = np.zeros((batch_size, 1), dtype=np.float64)
        # 获取数据总量【音频文件的数量】
        data_count = data_loader.get_data_count()
        index = 0
        # 此处循环如何终止呢？注意yield关键字的用法，此函数其实是一个生成器generator【每次通过next方法获取一个结果，下次接着上次终止的地方执行】，而非一个普通的函数。
        while True:
            # 按照batch_size初始化输入数据矩阵，维数(16, 1600, 200, 1)，元素值为0.0
            X = np.zeros((batch_size,) + self.speech_model.input_shape, dtype=np.float64)
            # 按照batch_size初始化输出数据矩阵，维数(16, 64)，元素值为0
            y = np.zeros((batch_size, self.max_label_length), dtype=np.int16)
            # 这两个列表用来收集什么的长度呢？有必要收集吗？
            input_length = []
            label_length = []

            # 读取batch_size【16】个音频文件数据
            for i in range(batch_size):
                # 按下标index获取一条数据：音频信号矩阵【长什么样呢？】、采样率、标签向量【由对应音频文件的标签拼音的索引组成】
                wavdata, sample_rate, data_labels = data_loader.get_data(index)
                # 对音频信号依据采样率进行特征提取【此处涉及特征提取算法】
                data_input = self.speech_features.run(wavdata, sample_rate)
                # 对于特征数据进行重塑
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                # 必须加上模pool_size得到的值，否则会出现inf问题，然后提示No valid path found.
                # 但是直接加又可能会出现sequence_length <= xxx 的问题，因此不能让其超过时间序列长度的最大值，比如200
                # 8
                pool_size = self.speech_model.input_shape[0] // self.speech_model.output_shape[0]
                # 取200与【特征矩阵第一维维数除以8的商与其余数的加和】之间的最小值【没明白作者的注释什么意思？出现inf问题，然后提示No valid path found】
                # 一个正整数不小于【该正整数除以另一个正整数的商与其余数的加和】
                # data_input.shape[0] // pool_size + data_input.shape[0] % pool_size <= data_input.shape[0]
                inlen = min(data_input.shape[0] // pool_size + data_input.shape[0] % pool_size,
                            self.speech_model.output_shape[0])
                # 收集输入特征长度？为什么不直接取data_input.shape[0]呢
                input_length.append(inlen)
                # 收集输入特征
                X[i, 0:len(data_input)] = data_input
                # 收集标签
                y[i, 0:len(data_labels)] = data_labels
                # 收集标签长度
                label_length.append([len(data_labels)])
                # 文件索引，用以循环读取下一个音频数据
                index = (index + 1) % data_count

            label_length = np.matrix(label_length)
            input_length = np.array([input_length]).T
            # yield有什么作用？与返回值或者终止循环有关吗？yield用法请见yield_learn.py
            yield [X, y, input_length, label_length], labels

    def train_model(self, optimizer, data_loader, epochs=1, save_step=1, batch_size=16, last_epoch=0, call_back=None):
        """
        训练模型

        参数：
            optimizer：tensorflow.keras.optimizers 优化器实例对象
            data_loader：数据加载器类型 (SpeechData) 实例对象
            epochs: 迭代轮数
            save_step: 每多少epoch保存一次模型
            batch_size: mini batch大小
            last_epoch: 上一次epoch的编号，可用于断点处继续训练时，epoch编号不冲突
            call_back: keras call back函数
        """
        save_filename = os.path.join('save_models', self.speech_model.get_model_name(),
                                     self.speech_model.get_model_name())

        self.trained_model.compile(loss=self.speech_model.get_loss_function(), optimizer=optimizer)
        print('[ASRT] Compiles Model Successfully.')
        # 对于输入数据进行分批特征提取，注意这里的结果yielddatas是一个迭代器
        yielddatas = self._data_generator(batch_size, data_loader)

        data_count = data_loader.get_data_count()  # 获取数据的数量
        # 计算每一个epoch迭代的训练批次
        num_iterate = data_count // batch_size
        # 设置迭代起始和终止轮数
        iter_start = last_epoch
        iter_end = last_epoch + epochs
        for epoch in range(iter_start, iter_end):  # 迭代轮数
            try:
                epoch += 1
                print('[ASRT Training] train epoch %d/%d .' % (epoch, iter_end))
                # 随机打乱标签编号/文件编号
                data_loader.shuffle()
                # 第一个参数yielddatas是一个迭代器，这是tensorflow.python.keras.models.Model对象fit_generator的参数要求
                self.trained_model.fit_generator(yielddatas, num_iterate, callbacks=call_back)
            except StopIteration:
                print('[error] generator error. please check data format.')
                break
            # 每迭代save_step次，保存一次模型数据
            if epoch % save_step == 0:
                # 创建保存模型的目录结构：save_models/"model_name"/"model_name"_epoch_"epoch"
                if not os.path.exists('save_models'):  # 判断保存模型的目录是否存在
                    os.makedirs('save_models')  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
                if not os.path.exists(os.path.join('save_models', self.speech_model.get_model_name())):  # 判断保存模型的目录是否存在
                    os.makedirs(
                        os.path.join('save_models', self.speech_model.get_model_name()))  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉
                # 保存模型文件
                self.save_model(save_filename + '_epoch' + str(epoch))

        print('[ASRT Info] Model training complete. ')

    def load_model(self, filename):
        """
        加载模型参数
        """
        self.speech_model.load_weights(filename)

    def save_model(self, filename):
        """
        保存模型参数
        """
        self.speech_model.save_weights(filename)

    def evaluate_model(self, data_loader, data_count=-1, out_report=False, show_ratio=True, show_per_step=100):
        """
        评估检验模型的识别效果
        """
        # 获取音频文件数量
        data_nums = data_loader.get_data_count()

        if data_count <= 0 or data_count > data_nums:  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = data_nums

        try:
            ran_num = random.randint(0, data_nums - 1)  # 获取[0, 音频文件数量-1]一个随机数，也即随机获取一个索引
            # 初始化总字数
            words_num = 0
            # 初始化识别错误字数
            word_error_num = 0
            # 格式化当前时间，2024年6月6日13:51:36形如20240606_135136
            nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            if out_report:
                txt_obj = open('Test_Report_' + data_loader.dataset_type + '_' + nowtime + '.txt', 'w',
                               encoding='UTF-8')  # 打开文件并读入
                txt_obj.truncate((data_count + 1) * 300)  # 预先分配一定数量的磁盘空间，避免后期在硬盘中文件存储位置频繁移动，以防写入速度越来越慢
                txt_obj.seek(0)  # 从文件首开始

            txt = ''
            i = 0
            while i < data_count:
                wavdata, fs, data_labels = data_loader.get_data((ran_num + i) % data_nums)  # 从随机数开始连续向后取一定数量数据
                # 提取特征
                data_input = self.speech_features.run(wavdata, fs)
                # 特征重塑
                data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                if data_input.shape[0] > self.speech_model.input_shape[0]:
                    print('*[Error]', 'wave data lenghth of num', (ran_num + i) % data_nums, 'is too long.',
                          'this data\'s length is', data_input.shape[0],
                          'expect <=', self.speech_model.input_shape[0],
                          '\n A Exception raise when test Speech Model.')
                    i += 1
                    continue
                # 数据格式出错处理 结束
                # 预测数据，得到识别语音的拼音索引集合
                pre = self.predict(data_input)

                words_n = data_labels.shape[0]  # 获取每个句子的字数
                words_num += words_n  # 把句子的总字数加上
                # 编辑距离就是与误差相关的量【data_labels为当前音频文件的标签【拼音索引向量】】
                edit_distance = get_edit_distance(data_labels, pre)  # 获取编辑距离
                if edit_distance <= words_n:  # 当编辑距离小于等于句子字数时
                    word_error_num += edit_distance  # 使用编辑距离作为错误字数
                else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_error_num += words_n  # 就直接加句子本来的总字数就好了
                # 每隔show_per_step次打印测试结果
                if i % show_per_step == 0 and show_ratio:
                    print('[ASRT Info] Testing: ', i, '/', data_count)
                # 将标签拼音索引和预测结果打印到txt文件中
                txt = ''
                if out_report:
                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'
                    txt_obj.write(txt)
                # 预测下一个音频文件
                i += 1

            # print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
            # 错误率计算公式：错误字数/总字数的百分数
            print('*[ASRT Test Result] Speech Recognition ' + data_loader.dataset_type + ' set word error ratio: ',
                  word_error_num / words_num * 100, '%')
            # 输出错误率到txt文件
            if out_report:
                txt = '*[ASRT Test Result] Speech Recognition ' + data_loader.dataset_type + ' set word error ratio: ' + str(
                    word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt_obj.truncate()  # 去除文件末尾剩余未使用的空白存储字节
                txt_obj.close()

        except StopIteration:
            print('[ASRT Error] Model testing raise a error. Please check data format.')

    def predict(self, data_input):
        """
        预测结果

        返回语音识别后的forward结果
        """
        return self.speech_model.forward(data_input)

    def recognize_speech(self, wavsignal, fs):
        """
        最终做语音识别用的函数，识别一个wav序列的语音
        """
        # 获取输入特征
        data_input = self.speech_features.run(wavsignal, fs)
        # 将输入特征转化为浮点数类型
        data_input = np.array(data_input, dtype=np.float64)
        # print(data_input,data_input.shape)
        # 将输入特征重塑【行列不变，增加第三维度，值为1】
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        # 根据输入特征预测拼音结果索引集合
        r1 = self.predict(data_input)
        # 获取拼音列表，形如当前目录的_pinyin_list.txt文件
        list_symbol_dic, _ = load_pinyin_dict(load_config_file(DEFAULT_CONFIG_FILENAME)['dict_filename'])
        # 遍历预测拼音结果索引集合，获取最终拼音结果列表
        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])

        return r_str

    def recognize_speech_from_file(self, filename):
        """
        最终做语音识别用的函数，识别指定文件名的语音
        """
        # 读取wav语音文件
        wavsignal, sample_rate, _, _ = read_wav_data(filename)
        # 识别语音
        r = self.recognize_speech(wavsignal, sample_rate)
        return r

    @property
    def model(self):
        """
        返回tf.keras model
        """
        return self.trained_model
