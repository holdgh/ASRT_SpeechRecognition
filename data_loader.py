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
加载语音数据集用的数据加载器的定义
"""
import os
import random
import numpy as np
from utils.config import load_config_file, DEFAULT_CONFIG_FILENAME, load_pinyin_dict
from utils.ops import read_wav_data


class DataLoader:
    """
    数据加载器

    参数：\\
        config: 配置信息字典
        dataset_type: 要加载的数据集类型，包含('train', 'dev', 'test')三种
    """
    def __init__(self, dataset_type:str):
        self.dataset_type = dataset_type

        self.data_list = list()
        self.wav_dict = dict()
        self.label_dict = dict()
        self.pinyin_list = list()
        self.pinyin_dict = dict()
        self._load_data()

    def _load_data(self):
        """
        依据配置文件，获取音频文件路径字典【标签编号/文件编号：音频文件路径】wav_dict、标签字典【标签编号/文件编号：拼音标签内容】label_dict、标签编号/文件编号列表data_list、拼音列表pinyin_list【见_pinyin_list.txt】和拼音索引字典pinyin_dict【见_pinyin_dict.json】
        """
        # 读取json配置文件
        config = load_config_file(DEFAULT_CONFIG_FILENAME)
        # 依据配置文件dict_filename配置项加载拼音与汉字的字典映射【拼音列表【见_pinyin_list.txt】和拼音索引【见_pinyin_dict.json】】
        self.pinyin_list, self.pinyin_dict = load_pinyin_dict(config['dict_filename'])
        # 按照配置文件遍历获取train/dev/test数据集
        for index in range(len(config['dataset'][self.dataset_type])):
            # 获取音频文件路径列表文件
            filename_datalist = config['dataset'][self.dataset_type][index]['data_list']
            # 获取音频文件目录
            filename_datapath = config['dataset'][self.dataset_type][index]['data_path']
            with open(filename_datalist, 'r', encoding='utf-8') as file_pointer:
                # 按行读取为列表
                lines = file_pointer.read().split('\n')
                for line in lines:
                    # 过滤空行
                    if len(line) == 0:
                        continue
                    # 对于非空行，按照空格获取当前行的内容列表
                    tokens = line.split(' ')
                    # data_list是干啥用的？用于音频文件匹配标签使用的，相当于标签编号【也可以理解为文件编号】。
                    # 收集标签编号【也可以理解为文件编号】
                    self.data_list.append(tokens[0])
                    # 每一行的最后一个元素为音频文件相对路径，需拼接上音频文件目录，方可作为最终的音频文件路径
                    # 以标签编号【也可以理解为文件编号】为key，对应的音频文件路径为value，收集起来
                    self.wav_dict[tokens[0]] = os.path.join(filename_datapath, tokens[1])
            # 获取标签文件路径
            filename_labellist = config['dataset'][self.dataset_type][index]['label_list']
            with open(filename_labellist, 'r', encoding='utf-8') as file_pointer:
                # 按行读取为列表
                lines = file_pointer.read().split('\n')
                for line in lines:
                    # 过滤空行
                    if len(line) == 0:
                        continue
                    # 对于非空行，按照空格获取当前行的内容列表
                    tokens = line.split(' ')
                    # 以标签编号【也可以理解为文件编号】为key，对应的拼音标签内容为value，收集起来
                    self.label_dict[tokens[0]] = tokens[1:]

    def get_data_count(self) -> int:
        """
        获取数据集总数量
        """
        return len(self.data_list)

    def get_data(self, index: int) -> tuple:
        """
        按下标获取一条数据：音频信号矩阵【长什么样呢？】、采样率、标签向量【由对应音频文件的标签拼音的索引组成】
        """
        # 按下标获取标签编号/文件编号
        mark = self.data_list[index]

        # 依据标签编号/文件编号获取音频文件路径，读取相应路径的音频文件数据，获取音频信号和采样率
        wav_signal, sample_rate, _, _ = read_wav_data(self.wav_dict[mark])
        labels = list()
        # 依据标签编号/文件编号获取相应音频文件的标签内容【由拼音组成的一句话】
        for item in self.label_dict[mark]:
            if len(item) == 0:
                continue
            # 将拼音对应的索引加入到labels中
            labels.append(self.pinyin_dict[item])
        # 将labels【拼音索引列表】转换为向量
        data_label = np.array(labels)
        return wav_signal, sample_rate, data_label

    def shuffle(self) -> None:
        """
        随机打乱数据
        """
        random.shuffle(self.data_list)
