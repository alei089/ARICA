#
# coding=utf-8
# Created by 轩玮 on 2019/7/15.
#
import pickle
import traceback
from datetime import datetime
import nltk
import pandas as pd
import pymysql
import math
from sklearn.externals import joblib
from bert_serving.client import BertClient
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from pandas._libs import sparse
from process.gen_lda_model import LdaModel
from scipy.spatial.distance import cosine
from scipy import sparse

from sklearn.cluster import DBSCAN
from my_utils.string_util import my_filter, senttokenize, is_english, wordtokenizer, filter_duplicate_space, remove_duplicate, \
    filter_quota, error_correction, cleantxt, RateSentiment, get_peaks_troughs, RateSentiment1
import time
from gensim.models import Word2Vec, CoherenceModel
import numpy as np

'''
程序主流程
1. 读取数据
2. 数据预处理
3. 写入数据库（句子）
4. 对句子意图分类
5. 主题分类
'''

class Preproccess:
    """
    程序前处理
    """

    def __init__(self):
        file = 'fle'
        self.rating_dict = {
            "Rated 1 stars out of five stars": "1",
            "Rated 2 stars out of five stars": "2",
            "Rated 3 stars out of five stars": "3",
            "Rated 4 stars out of five stars": "4",
            "Rated 5 stars out of five stars": "5",
        }
        self.english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '{',
                                '}', '`', '<', ">", '/', "^", "-", "_", "``", "''", "...", "......"]  # 标点符号列表
        self.sr = stopwords.words('english')
        self.sr.extend(["app", "please", "fix", "android", "google", "youtube", "uber", "facebook"])
        self.lemmatizer = WordNetLemmatizer()  # 词还原主干函数

    def clean(self,review, is_filter_duplicate_space=True, is_remove_duplicate=True,
              is_to_lower=False, is_check_correct=True):
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
            # review = cleantxt(review)  # 过滤非英文字符
            if is_filter_duplicate_space:
                review = filter_duplicate_space(review)
            if is_remove_duplicate:
                review = remove_duplicate(review)
            if not review:
                return None
            if is_check_correct:
                review = error_correction(review)
            review = ' '.join(wordtokenizer(review))
            if is_to_lower:
                review = review.lower()
            review = filter_quota(review)

        except:
            print(f"{review}处理异常")
            return None
        return review

    def sent_filter(self,text):
        texts_filtered = [word for word in wordtokenizer(text) if
                          word not in self.english_punctuations]  # 去掉标点符号
        #                 texts_filtered1=[Word(word).spellcheck()[0][0] for word in texts_filtered]#拼写校正
        texts_filtered2 = [word for word in texts_filtered if word not in self.sr]  # 去掉停用词
        texts_filtered3=[word for word in texts_filtered2 if len(word)>1]#去掉单词长度小于1的单词
        # texts_filtered3 = [word for word in texts_filtered2 if word not in sentimentword]  # 去掉情感词
        texts_filtered4 = [self.lemmatizer.lemmatize(word) for word in texts_filtered2]  # 主干还原法
        texts_filtered5 = [word for word in texts_filtered4 if len(word) > 1]  # 去掉单词长度小于1的单词
        refiltered = nltk.pos_tag(texts_filtered5)
        filtered = [w for w, pos in refiltered if
                    pos in ['NN', 'VB', "VBG", "VBD", "VBN", 'JJ', "NNS"]]  # 提取名词、动词、形容词
        if len(filtered) >= 1:  # 提取大于1个词的评论
            return " ".join(filtered)
        return None

    def replace_rating(self, rating_str):
        """
        把评星变成数字（1，2，3，4，5）
        :param rating_str:
        :return:
        """
        try:
            rating = self.rating_dict[rating_str]
        except:
            rating = rating_str
        return rating

    def replace_review_date(self, date_str):
        """
        把日期变成YYYYmmdd格式
        :param date_str:
        :return:
        """
        try:
            date = time.strftime("%Y%m%d", time.strptime(date_str, "%B %d, %Y"))
        except:
            date = date_str
        return date

    def process(self, file_path):
        """
        读取文件内容，并返回df
        :param file_path:
        :return:
        """
        data_frame = pd.read_csv(file_path,
                                 usecols=["app_id", "product_name", "rating_star", "helpful_number", "review_date",
                                          "user_name", "review_text",'avatar_url'])
        data_frame['review_date'] = data_frame['review_date'].apply(lambda x: self.replace_review_date(x))
        data_frame['rating_star'] = data_frame['rating_star'].apply(lambda x: self.replace_rating(x))
        data_frame['filter_review'] = data_frame['review_text'].apply(
            lambda x: self.clean(str(x)))
        data_frame = data_frame[data_frame['filter_review'] != None]
        return data_frame

class Process:
    """
    1. 原始数据入库
    2. 取数据进行处理
    """
    def __init__(self):
        self.database = 'review'
        self.config = {
            'host': '127.0.0.1',
            'port': 3306,
            'user': 'root',
            'database': self.database,
            'passwd': 'root',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }

    def gen_sen_vec(self, vec_model, sentences, size=180):
        '''
        生成句子表示矩阵
        :param file:  迭代输入
        :return:
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
            if sen.split():
                sents_mat.append(sen_vec / len(sen.split()))
            else:
                sents_mat.append(sen_vec)

        return sents_mat

    def df2db(self, df):
        '''
        数据处理入库
        :param path:
        :return:
        '''
        conn = pymysql.connect(**self.config)
        cursor = conn.cursor()
        now_date = time.strftime("%Y%m%d", time.localtime())

        sql = '''insert into source_review  (app_id,avatar_url,helpful_number,product_name,rating_star,review_date,review_text,user_name,filter_review,create_date,update_date) values  (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
        list_reviews = []

        for i in range(len(df)):
            row = df.iloc[i]
            temp_list = [str(i) for i in list(row)]
            temp_list.extend([now_date,now_date])
            list_reviews.append(temp_list)

        try:
            print("插入")
            for review in list_reviews:
                # print(review)
                cursor.execute(sql, review)
            conn.commit()  # 提交到数据库执行，一定要记提交哦

        except Exception as e:
            conn.rollback()  # 发生错误时回滚
            print(e)

        finally:
            # 关闭游标连接
            cursor.close()
            # 关闭数据库连接
            conn.close()

    def sents2db(self, app_id):
        """
        获取句子并入库
        :param app_id:
        :return:
        """
        conn = pymysql.connect(**self.config)
        cursor = conn.cursor()

        try:
            sql = f'SELECT * FROM  source_review where app_id = "{app_id}" and filter_review !="None"'
            print(sql)
            count = cursor.execute(sql)
            print('应用总句子数 total records:', cursor.rowcount)
            # 获取表名信息
            desc = cursor.description
            header = []
            for ss in desc:
                header.append(ss[0])
            # cursor.scroll(10, mode='absolute')
            results = cursor.fetchall()
        except Exception as e:
            traceback.print_exc()
            # 发生错误时会滚
            conn.rollback()
            print(e)
        sql = '''insert into app_sentence  (app_id,review_id,helpful_number,review_date,source_sentence,filter_sentence) values (%s,%s,%s,%s,%s,%s)'''
        try:
            for record in results:
                print(record)
                for sent in senttokenize(record.get('filter_review')):
                    filter_sentence = Preproccess().sent_filter(sent)
                    if filter_sentence != None and filter_sentence != ' ' and filter_sentence != "":
                        sent_record = []
                        sent_record.append(record.get("app_id"))
                        sent_record.append(record.get("review_id"))
                        sent_record.append(record.get("helpful_number"))
                        sent_record.append(record.get("review_date"))
                        sent_record.append(sent)
                        sent_record.append(filter_sentence)
                        cursor.execute(sql, sent_record)
            conn.commit()  # 提交到数据库执行，一定要记提交哦
        except Exception as e:
            conn.rollback()  # 发生错误时回滚
            print(e)
        finally:
            # 关闭游标连接
            cursor.close()
            # 关闭数据库连接
            conn.close()

    def intention_classify(self, list):
        """
        bert 意图分类
        :param sents:
        :return: 意图分类结果&&bert向量
        """
        bc = BertClient(check_length=False)
        aa = bc.encode(list)
        train_x = sparse.csr_matrix(aa)
        # clf = joblib.load(r'E:\python3\review_analysis\LogisticRegression.pkl')
        clf = joblib.load(r'E:\python3\review_analysis\classify\bert_LR_20191111.pkl')
        y_ = clf.predict(train_x)
        return y_,aa

    def topic_classify(self, list):
        """
        主题分类
        :param list: 句子列表
        :return: 句子主题
        """
        # model = Word2Vec.load(r'E:\python3\review_analysis\file\model\w2v\without_removestop_model_w2v.pkl')
        model = Word2Vec.load(r'E:\python3\review_analysis\file\model\w2v\without_removestop_200_w2v_20190505.pkl')
        data = np.array(self.gen_sen_vec(model, list,size = 200))
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

        lda_model = LdaModel()
        topics_vec = lda_model.gen_topic_vec(pickle.load(open(lda_model.best_lda_path, 'rb')))  # 生成主题向量
        print(len(topics_vec))
        print(topics_vec[1])
        print("=====")
        print(data[1])
        list = []
        for sent_vec in data:
            list.append(get_topic_id(sent_vec, topics_vec))
        return list

    def DBSCAN_c(self, lists):
        """
        聚类
        :param lists:  bert vec
        :return: List 句子聚类标签
        """
        # model = Word2Vec.load(r'E:\python3\review_analysis\file\model\w2v\without_removestop_model_w2v.pkl')
        data = sparse.csr_matrix(np.array(lists))

        model_dbscan = DBSCAN(eps=0.1,  # 邻域半径
                              min_samples=2,  # 最小样本点数，MinPts
                              metric='cosine',
                              metric_params=None,
                              algorithm='auto',  # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
                              leaf_size=30,  # balltree,cdtree的参数
                              p=None,  #
                              n_jobs=4)

        model_dbscan.fit(data)
        list_clu = model_dbscan.fit_predict(data)
        return list_clu

    def data2view(self,app_id):

        def get_best_review(sent_vecs):
            '''
            返回最接近的句子
            :param sent_vec: 中心向量
            :param topic_vecs:
            :return:
            '''
            min_cosine = 1
            sent_id = -1
            mean_vec = sent_vecs.sum(axis=0)/len(sent_vecs)

            for index, vec in enumerate(sent_vecs):
                temp_vec = cosine(mean_vec, vec)
                if temp_vec < min_cosine:
                    min_cosine = temp_vec
                    sent_id = index
            return sent_id

        conn = pymysql.connect(**self.config)
        cursor = conn.cursor()
        model = Word2Vec.load(r'E:\python3\review_analysis\file\model\w2v\without_removestop_200_w2v_20190505.pkl')
        try:
            sql = f'insert into review_view(app_id,topic_id,score,sentence_cluster,sentence_classify) select t.app_id,t.topic_id,t.sum_score,t.sentence_cluster,t.sentence_classify from '\
            f'(SELECT sentence_cluster,topic_id,sentence_classify,app_id,sum(sentence_score) as  sum_score FROM  app_sentence where app_id = "{app_id}"  and sentence_cluster!= -1 group by app_id,sentence_cluster,sentence_classify,topic_id) t'
            print(sql)
            count = cursor.execute(sql)
            sql1 = f'insert into review_view(app_id,topic_id,score,sentence_cluster,sentence_classify,review) select t.app_id,t.topic_id,t.sum_score,t.sentence_cluster,t.sentence_classify,t.source_sentence from '\
            f'(SELECT sentence_cluster,topic_id,sentence_classify,app_id,sentence_score as  sum_score,source_sentence FROM  app_sentence where app_id = "{app_id}"  and sentence_cluster=-1) t'
            print(sql1)
            sql2 = f'update review_view set score = 1*score where app_id= "{app_id}" and sentence_cluster=-1 and sentence_classify=4'
            sql3 = f'update review_view set score = 0.8*score where app_id= "{app_id}" and sentence_cluster=-1 and sentence_classify=0'
            sql4 = f'update review_view set score = 0.5*score where app_id= "{app_id}" and sentence_cluster=-1 and sentence_classify=1'
            sql5 = f'update review_view set score = 0.5*score where app_id= "{app_id}" and sentence_cluster=-1 and sentence_classify=2'
            sql6 = f'update review_view set score = 0.1*score where app_id= "{app_id}" and sentence_cluster=-1 and sentence_classify=3'

            cursor.execute(sql1)
            cursor.execute(sql2)
            cursor.execute(sql3)
            cursor.execute(sql4)
            cursor.execute(sql5)
            cursor.execute(sql6)
            # print('total records:', cursor.rowcount)
            conn.commit()
        except Exception as e:
            traceback.print_exc()
            # 发生错误时会滚
            conn.rollback()
            print(e)

        sql = f"""select app_id,topic_id,score,sentence_cluster,sentence_classify,id from review_view where app_id = '{app_id}' and sentence_cluster!=-1"""
        df = pd.read_sql(sql, conn)
        list_text = []
        list_sentence_id = []
        for i in range(len(df)):
            row = df.iloc[i]
            temp = f"""select sentence_id,source_sentence,sentence_classify,filter_sentence from app_sentence where topic_id = {row["topic_id"]} and sentence_cluster = {row["sentence_cluster"]} and sentence_classify = {row["sentence_classify"]} and app_id = '{row["app_id"]}'"""
            best_text = pd.read_sql(temp,conn)
            data = np.array(self.gen_sen_vec(model, best_text["filter_sentence"].tolist(), size=200))
            sent_id = get_best_review(data)
            list_text.append(best_text.iloc[sent_id]["source_sentence"])
            list_sentence_id.append(best_text.iloc[sent_id]["sentence_id"])
            rr = "'"
            rrr = r"\'"

        try:
            for i in range(len(df)):
                row = df.iloc[i]
                if row["sentence_classify"] == 0:
                    weight = 2
                elif row["sentence_classify"] == 1:
                    weight = 1
                    print(row["sentence_classify"])
                elif row["sentence_classify"] == 2:
                    weight = 1
                elif row["sentence_classify"] == 3:
                    weight = 0.1
                elif row["sentence_classify"] == 4:
                    weight = 3
                else:
                    weight = 1
                sql = f"""update review_view set sentence_id = {list_sentence_id[i]},review = '{list_text[i].replace(rr,rrr)}',score = {weight*row["score"]} where id = {row["id"]}"""
                print(sql)
                count = cursor.execute(sql)
            conn.commit()  # 提交到数据库执行，一定要记提交哦
        except Exception as e:
            traceback.print_exc()
            # 发生错误时会滚
            conn.rollback()
            print(e)
        finally:
            # 关闭游标连接
            cursor.close()
            # 关闭数据库连接
            conn.close()

    def update_db(self, app_id):
        conn = pymysql.connect(**self.config)
        cursor = conn.cursor()
        sql = f'SELECT * FROM  app_sentence where app_id = "{app_id}"  and filter_sentence != "None"'
        df = pd.read_sql(sql, conn)
        print(df.head(5))
        source_sentence = df['source_sentence']
        filter_reviews = df['filter_sentence']
        sentiment_scores = df['sentiment_score']
        sentence_id = df['sentence_id'].tolist()
        # print("cluster...")
        # cluster = self.DBSCAN_c(filter_reviews.tolist())
        print("topic_classify...")
        topic_classifies = self.topic_classify(filter_reviews.tolist())
        print("intention_classify...")
        intention_classifies,bert_vecs = self.intention_classify(source_sentence.tolist())
        bert_dict ={}
        for i in range(len(bert_vecs)):
            bert_dict[sentence_id[i]] = bert_vecs[i]
        now = datetime.now().strftime('%Y%m%d')
        min_date = df['review_date'].min()
        print(min_date)
        time_score = [round((float(review_date) - float(min_date) + 1)/(int(now) - int(min_date) +1),8) for review_date in df['review_date']]
        threshold = 3
        length_score = [round(1/(1+math.e**(-(len(s.split())-threshold))),8) for s in filter_reviews]
        print(length_score)
        max_help = df['helpful_number'].max()
        help_score = [round((num+1)/(max_help+1),8) for num in df['helpful_number']]
        print(help_score)
        # sentiment_score = [RateSentiment(sentence) for sentence in source_sentence]
        # sentiment_score = []
        # SentiStrength.jar

        try:
            for i in range(len(df)):
                row = df.iloc[i]
                sentence_score = round(length_score[i] * time_score[i] * help_score[i] * sentiment_scores[i],8)
                # print(sentence_score)
                update_sql = f"""update app_sentence set sentence_classify = '{intention_classifies[i]}' ,length_score = '{length_score[i]}',sentence_score = '{sentence_score}',time_score = '{time_score[i]}',help_score = '{help_score[i]}', topic_id = '{topic_classifies[i]}'   where sentence_id = '{row["sentence_id"]}'"""
                cursor.execute(update_sql)
            conn.commit()  # 提交到数据库执行，一定要记提交哦
        except Exception as e:
            conn.rollback()  # 发生错误时回滚
            print(e)

            # 聚类操作
        try:
            select_sql = f'SELECT  topic_id, sentence_classify FROM  app_sentence where app_id = "{app_id}"  and filter_sentence != "None" group by topic_id, sentence_classify'
            dbscan_df = pd.read_sql(select_sql, conn)
            for i in range(len(dbscan_df)):
                row = dbscan_df.iloc[i]
                select_sql1 = f'SELECT  sentence_id, source_sentence FROM  app_sentence where app_id = "{app_id}"  and filter_sentence != "None" and topic_id = {row["topic_id"]} and sentence_classify={row["sentence_classify"]}'
                topic_classify_df = pd.read_sql(select_sql1, conn)
                cluster_vec = []
                for j in range(len(topic_classify_df)):
                    cluster_vec.append(bert_dict[topic_classify_df.iloc[j]['sentence_id']])
                dbscan_res =  self.DBSCAN_c(cluster_vec)
                for j in range(len(topic_classify_df)):
                    row_cluster = topic_classify_df.iloc[j]
                    update_cluster = f'update app_sentence set sentence_cluster = {dbscan_res[j]} where sentence_id = {row_cluster["sentence_id"]}'
                    cursor.execute(update_cluster)
            conn.commit()  # 提交到数据库执行，一定要记提交哦
        except Exception as e:
            conn.rollback()  # 发生错误时回滚
            print(e)
        finally:
            # 关闭游标连接
            cursor.close()
            # 关闭数据库连接
            conn.close()

    def update_sentiment(self,app_id):
        """
        1. 获取句子评论分
        2. 获取波谷
        3. 波谷部分为1，其他为0，更新数据库
        :param app_id:
        :return:
        """
        conn = pymysql.connect(**self.config)
        cursor = conn.cursor()
        sql = f'select source_sentence, sentence_id  from app_sentence where app_id = \'{app_id}\' order by sentence_id'
        df1 =  pd.read_sql(sql, conn)
        file_name = 'F:/aaa.txt'
        df1['source_sentence'].to_csv(file_name,header=None,index=False,encoding='utf-8')
        file_name = RateSentiment1("F:/aaa.txt")
        df2 = pd.read_csv(file_name, sep='\t',encoding='ansi')
        list1 = df2['Positive'].tolist()
        list2 = df2['Negative'].tolist()
        list3 = []
        for i in range(len(list1)):
            if abs(list1[i])>abs(list2[i]):
                max_score = list1[i]
            else:
                max_score = list2[i]
            list3.append(max_score)
        try:
            for i in range(len(df1)):
                row = df1.iloc[i]
                update_sql = f"update app_sentence set sentiment_score = {list3[i]} where sentence_id = {row['sentence_id']}"
                cursor.execute(update_sql)
            conn.commit()  # 提交到数据库执行，一定要记提交哦
        except Exception as e:
            conn.rollback()  # 发生错误时回滚
            print(e)

        sql = f'select review_date, avg(sentiment_score) as avg_sentiment  from app_sentence where app_id = \'{app_id}\' group by review_date'
        df = pd.read_sql(sql, conn)
        h = df['avg_sentiment']

        peaks, troughs = get_peaks_troughs(df, 5)
        # plt.subplot(1, 2, 1)
        #
        # plt.subplot(1, 2, 2)
        # plt.plot(df['review_date'], h)
        # for x, y in peaks:
        #     plt.text(x, y, y, fontsize=10, verticalalignment="bottom", horizontalalignment="center")
        # for x, y in troughs:
        #     plt.text(x, y, y, fontsize=10, verticalalignment="top", horizontalalignment="center")
        #
        # plt.show()
        try:
            update_sql = f"update app_sentence set sentiment_score = 0.5 where  app_id = '{app_id}'"
            cursor.execute(update_sql)
            conn.commit()
            for review_date, y in troughs:
                dates = [int(review_date)-1,int(review_date),int(review_date)+1]
                update_sql = f"update app_sentence set sentiment_score = 1 where review_date in {tuple(dates)} and app_id = '{app_id}'"
                count1 = cursor.execute(update_sql)
                print(count1)
            conn.commit()  # 提交到数据库执行，一定要记提交哦
        except Exception as e:
            conn.rollback()  # 发生错误时回滚za
            print(e)
        finally:
            # 关闭游标连接
            cursor.close()
            # 关闭数据库连接
            conn.close()

if __name__ == '__main__':
    print("hello")
    # print(pre.sent_filter("what ever u want love it"))
    # print(pre.lemmatizer.lemmatize("ever"))
    pre = Preproccess()
    df = pre.process(r"E:\python3\review_analysis\process\csv_test\com.bestringtonesapps.freeringtonesforandroid_train.csv")# 1. 数据前处理得到df
    print(len(df))
    # df.sort_values("helpful_number",inplace=True,ascending=False)
    df.drop_duplicates(subset=['avatar_url','rating_star','review_text','review_date','user_name','product_name','helpful_number'],inplace=True)
    print(len(df))
    df.drop_duplicates(subset=['avatar_url'],inplace=True)
    print(len(df))
    df.drop_duplicates(subset=['rating_star','review_text','review_date','user_name','product_name','helpful_number'],inplace=True)
    print(len(df))

    # print(df.head())
    mainProcess = Process()
    mainProcess.df2db(df=df)# 2. 将前处理后的数据进行入库
    mainProcess.df2db(df=df)# 2. 将前处理后的数据进行入库
    mainProcess.sents2db(app_id="com.bestringtonesapps.freeringtonesforandroid")# 3. 将评论分为句子入库
    mainProcess.update_sentiment(app_id="com.bestringtonesapps.freeringtonesforandroid") # 4. 获取情感得分
    mainProcess.update_db(app_id="com.bestringtonesapps.freeringtonesforandroid")# 5. 把句子分类，并更新数据库
    mainProcess.data2view(app_id="com.bestringtonesapps.freeringtonesforandroid")# 6. 将整理的数据放入数据库


