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
ASRT语音识别的语言模型

基于N-Gram的语言模型
"""

import os

from utils.ops import get_symbol_dict, get_language_model


class ModelLanguage:
    """
    ASRT专用N-Gram语言模型
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.dict_pinyin = dict()
        self.model1 = dict()
        self.model2 = dict()

    def load_model(self):
        """
        加载N-Gram语言模型到内存
        """
        # 拼音字典【拼音【音节加音调】：对应的汉字列表】
        self.dict_pinyin = get_symbol_dict('dict.txt')
        # 字：出现频次
        self.model1 = get_language_model(os.path.join(self.model_path, 'language_model1.txt'))
        # 词【有两个汉字组成，严格来说并非是词，而是两个字紧挨在一起，例如："我是"并非一个词，只是这两个字可以挨在一起】：出现频次
        self.model2 = get_language_model(os.path.join(self.model_path, 'language_model2.txt'))
        model = (self.dict_pinyin, self.model1, self.model2)
        # 由【拼音【音节加音调】：对应的汉字列表】组成的元组
        return model

    def pinyin_to_text(self, list_pinyin: list, beam_size: int = 100) -> str:
        """
        拼音转文本，一次性取得全部结果
        """
        result = list()
        tmp_result_last = list()
        for item_pinyin in list_pinyin:
            tmp_result = self.pinyin_stream_decode(tmp_result_last, item_pinyin,
                                                   beam_size)  #
            # 获取当前拼音与其前置翻译结果的序列【已包含了前期结果，因此后续直接用tmp_result赋值给tmp_result_last】
            if len(tmp_result) == 0 and len(tmp_result_last) > 0:
                result.append(tmp_result_last[0][
                                  0])  # 当前拼音无法翻译为汉字时【无法匹配的原因是：当前拼音匹配不到单字或者当前拼音与前期中结果列表中任一结果的最后一个汉字组合的2-gram
                # 子序列无法匹配语言模型model2】，取中间结果中最大出现概率的前期翻译结果纳入最终结果列表中
                tmp_result = self.pinyin_stream_decode([], item_pinyin, beam_size)  # 当前拼音无法翻译为汉字时，将当前拼音作为首字进行翻译
                if len(tmp_result) > 0:  #
                    # 如果当前拼音作为首字可以翻译成功，则将当前拼音的翻译结果纳入最终结果列表中；如果当前拼音作为首字仍然无法翻译【无法翻译为汉字的原因是：当前拼音匹配不到单字】，则忽略该拼音
                    result.append(tmp_result[0][0])
                tmp_result = []  # 当前拼音无法翻译为汉字时，重置中间结果为空
            tmp_result_last = tmp_result

        if len(tmp_result_last) > 0:  # 如果最后一次中间结果非空，则将中间结果中出现概率最大的汉字序列纳入到最终结果中
            result.append(tmp_result_last[0][0])

        return ''.join(result)

    def pinyin_stream_decode(self, temple_result: list,
                             item_pinyin: str,
                             beam_size: int = 100) -> list:
        """
        拼音流式解码，逐字转换，每次返回中间结果
        """
        # 如果这个拼音不在汉语拼音字典里的话，直接返回空列表，不做decode
        if item_pinyin not in self.dict_pinyin:
            return []

        # 获取拼音下属的字的列表，cur_words包含了该拼音对应的所有的字
        cur_words = self.dict_pinyin[item_pinyin]
        # 第一个字做初始处理
        if len(temple_result) == 0:
            lst_result = list()
            # 计算所有单字频次之和
            total_count = 0.0
            for cur_word in cur_words:
                if cur_word in self.model1:
                    total_count += float(self.model1[cur_word])
            for word in cur_words:
                # 没有出现在单字频次字典中的汉字，丢弃
                if word in self.model1:
                    # 计算单字出现频率
                    word_probability = float(self.model1[word]) / total_count
                    lst_result.append([word, word_probability])
                # 添加该字到可能的句子列表，设置初始概率为1.0
                # lst_result.append([word, 1.0])
            lst_result = sorted(lst_result, key=lambda x: x[1], reverse=True)
            return lst_result

        # 开始处理已经至少有一个字的中间结果情况【前期中间结果非空的情况】
        new_result = list()
        for sequence in temple_result:
            for cur_word in cur_words:
                # 得到2-gram的汉字子序列，sequence[0]为汉字列表，sequence[0][
                # -1]为汉字列表的最后一个汉字【当前拼音的前一个拼音对应的中间结果中的汉字，也即当前汉字的前一个汉字】，sequence[1]为汉字列表对应的出现概率
                tuple2_word = sequence[0][-1] + cur_word
                if tuple2_word not in self.model2:
                    # 如果2-gram子序列不存在，则更换下一个cur_word
                    continue
                # 计算状态转移概率
                prob_origin = sequence[1]  # 原始概率
                count_two_word = float(self.model2[tuple2_word])  # 二字频数
                count_one_word = float(self.model1[tuple2_word[-2]])  # 【2-gram子序列的倒数第二个字，也即当前字的前一个字】单字频数
                cur_probility = prob_origin * count_two_word / count_one_word  #
                # 计算当前2-gram子序列的出现概率=（原始概率【当前字之前的序列出现概率】*二字频数）/单子频数
                new_result.append([sequence[0] + cur_word, cur_probility])
        # 对当前结果序列列表按照概率大小，从大到小排序【todo 考虑new_result有没有可能为空的情况，如果为空，就相当于无法连同前期中间结果翻译当前拼音，但当前拼音是有单字匹配的】
        new_result = sorted(new_result, key=lambda x: x[1], reverse=True)
        if len(new_result) > beam_size:  # 如果当前结果序列的元素个数大于阈值100，则取概率较大的前100个结果
            return new_result[0:beam_size]
        return new_result


if __name__ == '__main__':
    ml = ModelLanguage('model_language')
    ml.load_model()

    _str_pinyin = ['zhe4', 'zhen1', 'shi4', 'ji2', 'hao3', 'de5']
    _RESULT = ml.pinyin_to_text(_str_pinyin)
    print('语音转文字结果:\n', _RESULT)
