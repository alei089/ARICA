# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     gen_sentence_vec
   Description :
   Author :       lpl
   Date：         2019/3/23
-------------------------------------------------
   Change Activity:
                   2019/3/23:
-------------------------------------------------
"""
from gensim.models import Word2Vec

__author__ = 'lpl'
import numpy as np

VEC_SIZE = 180
MAXLEN = 56


def add_unexist_word_vec(w2v, vocab, vec_size=VEC_SIZE):
    """
    将词汇表中没有embedding的词初始化()
    :param w2v:经过word2vec训练好的词向量
    :param vocab:总体要embedding的词汇表
    """
    for word in vocab:
        if word not in w2v and vocab[word] >= 1:
            w2v[word] = np.random.uniform(-0.25, 0.25, vec_size)


def pad_sentences(data, maxlen=MAXLEN, values=0., vec_size=VEC_SIZE):
    """padding to max length
    :param data:要扩展的数据集
    :param maxlen:扩展的h长度
    :param values:默认的值
    """
    length = len(data)
    if length < maxlen:
        for i in range(maxlen - length):
            data.append(np.array([values] * vec_size))
    return data


def get_vec_by_sentence_list(word_vecs, sentence_list, maxlen=MAXLEN, values=0., vec_size=VEC_SIZE):
    """
    :param sentence_list:句子列表
    :return:句子对应的矩阵向量表示
    """
    data = []

    for sentence in sentence_list:
        # get a sentence
        sentence_vec = []
        words = sentence.split()
        for word in words:
            sentence_vec.append(word_vecs[word].tolist())

        # padding sentence vector to maxlen(w * h)
        sentence_vec = pad_sentences(sentence_vec, maxlen, values, vec_size)
        # add a sentence vector
        data.append(np.array(sentence_vec))
    return data

def gen_sen_vec(vec_model,sentences):
    '''
    生成句子表示矩阵
    :param file:  迭代输入
    :return:
    '''
    sents_mat=[]
    for sen in sentences:
        sen_vec=np.zeros(VEC_SIZE)
        for word in sen:
            try:
                word_vec=vec_model[word]
            except:
                word_vec=np.zeros((VEC_SIZE))
            sen_vec+=word_vec
        sents_mat.append(sen_vec)

    return sents_mat


if __name__ == '__main__':
    model = Word2Vec.load(r'E:\python3\review_analysis\file\model\w2v\without_removestop_model_w2v.pkl')
    print(np.random.uniform(-0.25, 0.25, 180))
    print(np.zeros((VEC_SIZE)))

    model['dfgfd']
    pass
