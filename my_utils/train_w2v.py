# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train_w2v
   Description :
   Author :       lpl
   Date：         2019/3/11
-------------------------------------------------
   Change Activity:
                   2019/3/11:
-------------------------------------------------
"""
import codecs
import multiprocessing

__author__ = 'lpl'
from gensim.models import word2vec, Word2Vec
import pandas as pd
import time
# data_100000 = pd.read_csv(r'E:\python2\review_analysis\file\corpus\without_removestop_corpus.csv', header = 0)
# content = data_100000.review_text.values
def train_vord2vec():
    with open(r'E:\python2\review_analysis\file\corpus\without_removestop_corpus.txt','r',encoding='utf-8') as read1:
        lines= read1.readlines()
    print(lines[5])
    print(len(lines))
    start = time.clock()
    sentences = word2vec.Text8Corpus(r'E:\python2\review_analysis\file\corpus\without_removestop_corpus.txt')  # 加载语料
    model=word2vec.Word2Vec(sentences, size=256, window=6, min_count=5,
                         workers=multiprocessing.cpu_count(),iter=10)
    model_name = r'E:\python2\review_analysis\file\model\w2v\without_removestop_model_w2v.pkl'
    model.save(model_name)
    end = time.clock()
    print('Running time: %s Seconds' % (end - start))


def load_model():
    model_name = r'E:\python2\review_analysis\file\model\w2v\without_removestop_model_w2v.pkl'
    model = Word2Vec.load(model_name)
    model.wv.save_word2vec_format(r'E:\python3\review_analysis\file\model\w2v\without_removestop_01.model.txt', r'E:\python3\review_analysis\file\model\w2v\without_removestop_01.vocab.txt',
                                  binary=False)  # 将模型保存成文本，model.wv.save_word2vec_format()来进行模型的保存的话，会生成一个模型文件。里边存放着模型中所有词的词向量。这个文件中有多少行模型中就有多少个词向量。
    y2 = model.wv.similarity(u"love", u"like")
    print(y2)
    print(model['love'])

    for i in model.wv.most_similar(u"love"):
        print (i[0],i[1])

if __name__ == '__main__':
    load_model()
    pass