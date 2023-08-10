#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import pickle
import string
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords

# 路径设置
current_dir = os.path.abspath(os.getcwd())
dataset = '20ng'
input_path = os.path.abspath(os.path.join(current_dir, '..', 'data_tgcn', dataset, 'build_train', dataset))
output_path = os.path.abspath(os.path.join(current_dir, '..', 'data_tgcn', dataset, 'stanford'))

# 读取病例
yic_content_list = []
with open(input_path + '.clean.txt', 'r', encoding="gbk") as f:
    lines = f.readlines()
    for line in lines:
        yic_content_list.append(line.strip())

stop_words = set(stopwords.words('english'))

# 获取句法依存关系对
rela_pair_count_str = {}
with StanfordCoreNLP('D:\迅雷下载\stanford-corenlp-full-2018-02-27\stanford-corenlp-full-2018-02-27', lang='zh') as nlp:
    for doc_id in range(len(yic_content_list)):
        print(doc_id)
        words = yic_content_list[doc_id]
        window = "\n".join(words)
        window = window.translate(str.maketrans('', '', string.punctuation))
        res = nlp.dependency_parse(window)
        for tuple in res:
            pair = (tuple[0], tuple[1])
            if pair[0] == 'ROOT' or pair[1] == 'ROOT':
                continue
            if pair[0] == pair[1]:
                continue
            if pair[0] in stop_words or pair[1] in stop_words:
                continue
            word_pair_str = str(pair[0]) + ',' + str(pair[1])
            if word_pair_str in rela_pair_count_str:
                rela_pair_count_str[word_pair_str] += 1
            else:
                rela_pair_count_str[word_pair_str] = 1
            # two orders
            word_pair_str = str(pair[0]) + ',' + str(pair[1])
            if word_pair_str in rela_pair_count_str:
                rela_pair_count_str[word_pair_str] += 1
            else:
                rela_pair_count_str[word_pair_str] = 1

#将rela_pair_count_str存成pkl格式
output1 = open(output_path + '/{}_stan.pkl'.format(dataset), 'wb')
pickle.dump(rela_pair_count_str, output1)
output1.close()

