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
用于通过ASRT语音识别系统预测一次语音文件的程序
"""

import os

from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    AUDIO_LENGTH = 1600
    AUDIO_FEATURE_LENGTH = 200
    CHANNELS = 1
    # 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
    OUTPUT_SIZE = 1428
    # 定义语音模型
    sm251bn = SpeechModel251BN(
        input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
        output_size=OUTPUT_SIZE
    )
    feat = Spectrogram()
    ms = ModelSpeech(sm251bn, feat, max_label_length=64)
    # 加载训练好的模型
    ms.load_model('D:\project\AI\ASRT_SpeechRecognition\save_models\\' + sm251bn.get_model_name() + '.model.h5')
    # 获取识别结果
    # res = ms.recognize_speech_from_file('filename.wav')
    # ['gei3', 'yi3', 'ting3', 'yao4', 'shi4', 'da4', 'jia1', 'hao3', 'wo3', 'shi4', 'lian2', 'tong2', 'shang4', 'hai3', 'qiang2', 'ye4', 'hu4', 'lian2', 'wang3', 'you3', 'xian4', 'gong1', 'xi1', 'de5', 'yuan2', 'gong1', 'wo3', 'jiu4']
    # 给以挺要誓大家好我是连同上海强业互联网友献工悉的员工我就
    res = ms.recognize_speech_from_file('D:\project\AI\ASRT_SpeechRecognition\data\FormatFactoryPart1.wav')
    # ['shi2', 'xian4', 'shu4', 'ju4', 'jian1', 'ce4', 'shi4', 'ping2', 'jian1', 'kong4', 'he2', 'gao4', 'yin3', 'shu4', 'ju4', 'de5', 'shi1', 'shi2', 'duan3', 'gui1', 'di4', 'san1', 'ge4', 'shi4', 'jiao4', 'er2', 'yu4', 'chan3', 'pin3', 'ran2', 'yao4', 'fu4', 'ze2', 'dui4']
    # 实现数据监测试平间空和告引数据的失时短规第三个视觉而与产品然要负责对
    # res = ms.recognize_speech_from_file('D:\project\AI\ASRT_SpeechRecognition\data\FormatFactoryPart6.wav')
    print('*[提示] 声学模型语音识别结果：\n', res)

    ml = ModelLanguage('model_language')
    ml.load_model()
    str_pinyin = res
    # todo 将拼音转换为文本？同样的读音，对应多个字，如何确定对应哪个字呢？
    res = ml.pinyin_to_text(str_pinyin)
    print('语音识别最终结果：\n', res)
