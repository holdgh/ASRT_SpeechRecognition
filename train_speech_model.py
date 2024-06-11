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
用于训练语音识别系统语音模型的程序
"""

import os
from tensorflow.keras.optimizers import Adam

from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from data_loader import DataLoader
from speech_features import SpecAugment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
# 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
OUTPUT_SIZE = 1428
# 定义声学模型，用于将wav语音信号转换为二维频谱信号？
sm251bn = SpeechModel251BN(
    input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
    output_size=OUTPUT_SIZE
)
# 获取声学特征？
feat = SpecAugment()
# 加载训练数据
# 依据配置文件，获取音频文件路径字典【标签编号/文件编号：音频文件路径】wav_dict、标签字典【标签编号/文件编号：拼音标签内容】label_dict、标签编号/文件编号列表data_list、拼音列表pinyin_list【见_pinyin_list.txt】和拼音索引字典pinyin_dict【见_pinyin_dict.json】
train_data = DataLoader('train')
# 定义优化器
opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=10e-8)
# 依据声学模型和声学特征获取语音模型
ms = ModelSpeech(sm251bn, feat, max_label_length=64)

# ms.load_model('save_models/' + sm251bn.get_model_name() + '.model.h5')
ms.train_model(optimizer=opt, data_loader=train_data,
               epochs=50, save_step=1, batch_size=16, last_epoch=0)
# 保存模型文件：save_models/SpeechModel251bn
ms.save_model('save_models/' + sm251bn.get_model_name())
