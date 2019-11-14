#
# coding=utf-8
# Created by 轩玮 on 2019/8/18.
#
import os

from gensim.models import CoherenceModel, word2vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

from scipy.spatial.distance import cosine
import pickle

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
import warnings

from my_utils.string_util import is_english, cleantxt, wordtokenizer, filter_duplicate_space, remove_duplicate, filter_quota, \
    error_correction

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
1. 由语料生成LDA模型   语料保存为 best_lda.pkl
2. 由LDA生成中心向量
3. 提起句子到中心的方法
"""


class LdaModel:
    """

    """

    def __init__(self):
        os.environ['MALLET_HOME'] = r'E:\python3\review_analysis\file\LDA\mallet-2.0.8'
        self.mallet_path = r'E:\python3\review_analysis\file\LDA\mallet-2.0.8\bin\mallet'  # update this path
        self.best_lda_path = "orandroid.pkl"
        self.model_wv = word2vec.Word2Vec.load(
            r'E:\python3\review_analysis\file\model\w2v\without_removestop_200_w2v_20190505.pkl')  # size维度，min_count最少出现次数

    def get_best_lda_model(self, dictionary, corpus, texts, limit, id2word, start=2, step=1,model_name="best_lda.pkl"):

        """
        Compute c_v coherence for various number of topics
        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        ldamodel
        """
        max_coherence = 0
        for num_topics in range(start, limit, step):
            print(f"topic nums-->{num_topics}")
            model = gensim.models.wrappers.LdaMallet(self.mallet_path, corpus=corpus, num_topics=num_topics,
                                                     id2word=id2word)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            temp_coherence = coherencemodel.get_coherence()
            if temp_coherence > max_coherence:
                print(f"the best topic num---->{num_topics}")
                max_coherence = temp_coherence
                print(max_coherence)
                temp_model = model
        with open(model_name[-12:-4], 'wb') as fw:
            pickle.dump(temp_model, fw)
        # return temp_model

    def dump_best_lda(self,csv_name):
        """
        1. 对文本进行处理
        2. 向量化
        :return:
        """

        def sent_to_words(sentences):
            for sentence in sentences:
                yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

        # Define functions for stopwords, bigrams, trigrams and lemmatization
        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        def make_bigrams(texts):
            return [bigram_mod[doc] for doc in texts]

        def make_trigrams(texts):
            return [trigram_mod[bigram_mod[doc]] for doc in texts]

        def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
            """https://spacy.io/api/annotation"""
            texts_out = []
            for sent in texts:
                doc = nlp(" ".join(sent))
                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            return texts_out

        def clean(review, is_filter_duplicate_space=True, is_remove_duplicate=True,
                  is_to_lower=True, is_check_correct=True):
            """
            只做简单的处理，主要矫正拼写错误，入库后续进一步处理
            2. 判断是否为英文
            3. 去除重复字母
            4. 过滤非英文字符
            5. 转换小写
            6. 单词矫正
            :param review: 评论
            :param is_filter_duplicate_space:
            :param is_remove_duplicate:
            :param is_to_lower:
            :param is_check_correct:
            :return:
            """
            try:
                review = str(review)
                if is_english(review) == None:
                    return None
                if not review:
                    return None
                review = cleantxt(review)  # 过滤非英文字符
                if not review:
                    return None
                review = ' '.join(wordtokenizer(review))
                if is_to_lower:
                    review = review.lower()
                if is_check_correct:
                    review = error_correction(review)
                if is_filter_duplicate_space:
                    review = filter_duplicate_space(review)
                if is_remove_duplicate:
                    review = remove_duplicate(review)
                review = filter_quota(review)

            except:
                print(f"{review}处理异常")
                return None
            return review

        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'android', 'app', 'application'])

        # Convert to list
        data = pd.read_csv(csv_name,encoding='utf-8')

        data = data.review_text.values.tolist()
        data = [clean(review) for review in data]
        data = [review for review in data if review is not None]

        # Remove Emails
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]

        data_words = list(sent_to_words(data))
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)
        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)
        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        # data_lemmatized = data_words_bigrams
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        self.get_best_lda_model(dictionary=id2word, corpus=corpus, texts=data_lemmatized, id2word=id2word, start=2,
                                limit=25,
                                step=1,model_name=csv_name)

    def gen_topic_vec(self, lda_model):
        """

        :param lda_model:  lda模型
        :return: 加权后的中心向量
        """
        size_lda = lda_model.num_topics
        size_word = 200
        the_id = []  # 每个主题的前10个词的ID
        the_vl = []  # 每个主题的前10个词的value
        the_w = []  # 每个主题的前10个词的占权重

        for x in range(size_lda):
            the_id.append([xx[0] for xx in lda_model.show_topic(x, topn=10)])
            the_sum = sum([xx[1] for xx in lda_model.show_topic(x, topn=10)])
            the_w.append([xx[1] / the_sum for xx in lda_model.show_topic(x, topn=10)])
            # print x,"主题",the_sum,the_w

        m = 0
        the_wv = np.zeros([size_lda, size_word])  # 每个主题映射到word2vec,主题数，word2vec
        # 主题下每个词在word2vec下坐标加权求和
        for words in the_id:
            n = 0
            for word in words:
                the_wv[m] += [x_word * the_w[m][n] for x_word in self.model_wv[word]]
                n += 1
            m += 1
        return the_wv

    def gen_sen_vec(vec_model, sentences, size):
        '''
        生成句子表示矩阵
        :param vec_model: 模型
        :param sentences: 句子
        :param size: 向量长度
        :return: 向量列表
        '''
        sents_mat = []
        for sen in sentences:
            sen_vec = np.zeros(size)
            for word in sen.split():
                try:
                    word_vec = vec_model[word]
                except:
                    word_vec = np.zeros((size))
                sen_vec += word_vec
            sents_mat.append(sen_vec / len(sen.split()))

        return np.array(sents_mat)

    def get_topic_id(sent_vec, topic_vecs):
        '''
        返回句子的主题
        :param sent_vec:
        :param topic_vecs:
        :return:
        '''
        min_cosine = 1
        topic_id = -1
        for index, vec in enumerate(topic_vecs):
            temp_vec = cosine(sent_vec, vec)
            if temp_vec < min_cosine:
                min_cosine = temp_vec
                topic_id = index
        return topic_id


# if __name__ == '__main__':
# #
#     pp = LdaModel()

    # print(pp.clean("I like  this is my best application how about you ? bbbbbbbjjjjjjj i'm i''m          hello  >>>>>>........."))
    # print(error_correction(r"""I like  this is my best application how about you ? bbbbbbb"jjj'''jjjj i'm i''m          hello  >>>>>>........."""))
    # print(pp.clean(r"""I like  this is my best application how about you ? bbbbbbb"jjj'''jjjj i'm i''m          hello  >>>>>>........."""))
#     pp.dump_best_lda(csv_name=r"E:\python3\review_analysis\process\csv_test\deezer.android.app_train.csv")  # 生成lda 并保存模型
#     pp.dump_best_lda(csv_name=r"E:\python3\review_analysis\process\csv_test\com.discord_train.csv")  # 生成lda 并保存模型
# #
#     topics_vec = pp.gen_topic_vec(pickle.load(open(pp.best_lda_path), 'rb'))  # 生成主题向量
#
#     vec = pp.gen_sen_vec(pp.model_wv, ['I like eating'], 200)
#     id = pp.get_topic_id(vec[0], topics_vec)
#
#     id = pp.get_topic_id(vec[0], pp.get_topic_vec())
