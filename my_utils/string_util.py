# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     string_util
   Description :   文本预处理
   Author :       lpl
   Date：         2019/2/18
-------------------------------------------------
   Change Activity:
                   2019/2/18:
-------------------------------------------------
"""

from langid import langid
import re
import string


import six
import nltk
from nltk.tokenize import sent_tokenize

from nltk.corpus import wordnet, stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import subprocess
import shlex
import os

wordMapper = {}
with open(r'E:\python3\review_analysis\file\wordMapper.txt') as f:
    lines = f.readlines()
    for line in lines:
        key, value = line.strip().split(',')
        wordMapper[key] = value


def get_peaks_troughs(h, rangesize):
    """
    获取波峰波谷
    :param h: 传入数据
    :param rangesize: 跨度
    :return: 波峰 波谷
    """
    peaks = list()
    troughs = list()

    S = 0
    for x in range(1, len(h) - 1):
        row  = h.iloc[x]
        next_row = h.iloc[x+1]
        if S == 0:
            if row['avg_sentiment'] > next_row['avg_sentiment']:
                S = 1  ## down
            else:
                S = 2  ## up
        elif S == 1:
            if row['avg_sentiment']  < next_row['avg_sentiment']:
                S = 2
                ## from down to up
                if len(troughs):
                    ## check if need merge
                    (prev_x, prev_trough) = troughs[-1]
                    if int(row['review_date']) - int(prev_x) < rangesize:
                        if prev_trough > row['avg_sentiment']:
                            troughs[-1] = (row['review_date'], row['avg_sentiment'])
                    else:
                        troughs.append((row['review_date'], row['avg_sentiment']))
                else:
                    troughs.append((row['review_date'], row['avg_sentiment']))


        elif S == 2:
            if row['avg_sentiment'] > next_row['avg_sentiment']:
                S = 1
                ## from up to down
                if len(peaks):
                    prev_x, prev_peak = peaks[-1]
                    if int(row['review_date']) - int(prev_x) < rangesize:
                        if prev_peak < row['avg_sentiment']:
                            peaks[-1] = (row['review_date'], row['avg_sentiment'])
                    else:
                        peaks.append((row['review_date'], row['avg_sentiment']))
                else:
                    peaks.append((row['review_date'], row['avg_sentiment']))

    return peaks, troughs

def wordtokenizer(sentence):
    '''
    句子分词
    :param sentence: 输入句子
    :return: list
    '''
    words = word_tokenize(sentence)
    return words


def senttokenize(text):
    '''
    返回句子列表
    :param text: 文本
    :return:
    '''
    return [i.strip() for i in sent_tokenize(text)]


def filter_duplicate_space(text):
    '''
    去除重复出现的字母等
    :param text:
    :return: text
    '''
    return ''.join(
        [x for i, x in enumerate(text) if not (i < len(text) - 1 and not (x.strip()) and not (text[i + 1].strip()))])


def tokenize_text(text):
    '''
    分词 返回单词列表
    :param text:
    :return:
    '''
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    '''
    移除特殊字符
    :param text:
    :return:
    '''
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_duplicate(text):
    '''
    去除重复出现次数多的字母等
    :param text:
    :return: text
    '''
    if six.PY2:
        text = text.decode('utf8')
    l = []

    start = 0
    end = 0
    duplicates = []
    while start < len(text):
        while end < len(text) and text[start] == text[end]:
            duplicates.append(text[start])
            end += 1
        l.append(''.join(duplicates[:5]))
        duplicates = []
        start = end

    text = ''.join(l)

    if six.PY2:
        text = text.encode('utf8')
    return text


def filter_quota(text):
    return text.replace("''", '" ').replace("``", '" ')


def remove_stopwords(txt):
    # stop_list = STOP_LIST
    stop_list = stopwords.words('english')
    stop_list.extend(
        ["app", "please", "fix", "android", "google", "youtube", "as", "uber", "dont", "cousin", "pp", "facebook",
         "fitbit","application"])
    filtered = [w for w in txt if (w not in stop_list)]
    return ' '.join(filtered)


def cleantxt(raw):
    '''
    过滤非英文
    :param raw:
    :return:
    '''
    if raw == None:
        return None
    fil = re.compile(u"""[^0-9a-zA-Z\u4e00-\u9fa5.,()?]+""", re.UNICODE)
    return fil.sub(' ', raw)


def expand_contractions(text, contraction_mapping):
    '''
    缩写还原（暂时未用，用的wordmapper）
    :param text:
    :param contraction_mapping:
    :return: 文本
    '''
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        if expanded_contraction:
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction
        return ''

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def filte_sents(text):
    '''
    过滤不用的句子 邮箱、网址等
    :param sents:
    :return:
    '''
    sents = nltk.sent_tokenize(text)
    filte_sents = []
    url_pattern = r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
    compiled_url_pattern = re.compile(url_pattern)
    email_pattern = r"[\w!#$%&'*+/=?^_`{|}~-]+(?:\.[\w!#$%&'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?"
    id = r'(?<=").*?(?=")'
    compiled_email_pattern = re.compile(email_pattern)
    for i in range(len(sents)):
        if (not compiled_url_pattern.search(sents[i]) and not compiled_email_pattern.search(sents[i])):
            r1 = '[★、*…【】●《》？“”‘’！[\\]^_`{|}~]+'  # 进行自定义过滤字符
            filte_sents.append(re.sub(r1, '', sents[i]).strip())
    if filte_sents:
        return ' '.join(filte_sents)
    else:
        return None


def get_wordnet_pos(treebank_tag):
    '''
    获取词性
    :param treebank_tag:
    :return:
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    '''
    词性还原
    :param sentence:  输入句子
    :return:
    '''
    if sentence:
        res = []
        lemmatizer = WordNetLemmatizer()
        for word, pos in pos_tag(word_tokenize(sentence)):
            wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
            res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

        return ' '.join(res)
    else:
        return sentence


def get_error_correction_dic():
    dict1 = {}
    with open(r'E:\python3\review_analysis\file\wordMapper.txt') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.strip().split(',')
            dict1[key] = value
    return dict1


def is_english(str):
    '''
    返回英语的评论
    :param str:
    :return:
    '''
    if str == None:
        return None
    if langid.classify(str)[0] == 'en':  # 去掉非英文评论
        return str
    else:
        return None


def error_correction(text):
    '''
    用字典纠正错误
    :param map_dict: 字典
    :param text: 评论
    :return:
    '''
    words = text.strip().split()
    sen = [word if word not in wordMapper.keys() else wordMapper[word] for word in words]
    return ' '.join(sen)


def my_filter(text, is_filter_duplicate_space=True, is_remove_duplicate=True, is_filte_sents=True,
              is_remove_stopwords=True,
              is_to_lower=True, is_lemmatize_sentence=True, is_check_correct=False):
    '''
    1. 把评论字符化
    2. 判断是否为英文
    3. 去除重复字母
    4. 过滤非英文字符
    5. 转换小写
    6. 单词矫正
    7. 去除停用词
    8. 词性还原
    :param text: 评论
    :param is_filter_duplicate_space:
    :param is_remove_duplicate:
    :param is_filte_sents:
    :param is_remove_stopwords:
    :param is_to_lower:
    :param is_lemmatize_sentence:
    :param is_check_correct:
    :return:
    '''
    try:
        text = str(text)
        if is_english(text) == None:
            return None
        if is_filte_sents:
            text = filte_sents(text)
        if not text:
            return None
        text = cleantxt(text)  # 过滤非英文字符
        if not text:
            return None
        text = ' '.join(wordtokenizer(text))
        if is_filter_duplicate_space:
            text = filter_duplicate_space(text)
        if is_remove_duplicate:
            text = remove_duplicate(text)
        text = filter_quota(text)
        if is_to_lower:
            text = text.lower()
        if is_check_correct:
            text = error_correction(text)
        if is_remove_stopwords:
            text = wordtokenizer(text)
            text = remove_stopwords(text)
        # text = remove_special_characters(text)
        if is_lemmatize_sentence:
            text = lemmatize_sentence(text)
    except:
        print(f"{text}处理异常")
        return None
    return text



def RateSentiment(sentiString):
    '''
    输出评论的情感得分 如：[3,-1]
    :param sentiString: 评论句子
    :return:
    '''
    SentiStrengthLocation = r"E:\python3\review_analysis\lib\SentiStrength.jar"  # SentiStrength.jar所在的路徑
    SentiStrengthLanguageFolder = r"E:\python3\review_analysis\lib\SentStrength_Data\\"     # SentStrength_Data
    if not os.path.isfile(SentiStrengthLocation):
        print("SentiStrength not found at: ", SentiStrengthLocation)
    if not os.path.isdir(SentiStrengthLanguageFolder):
        print("SentiStrength data folder not found at: ", SentiStrengthLanguageFolder)
        # open a subprocess using shlex to get the command line string into the correct args list format
    shlex.split("java -jar '" + SentiStrengthLocation + "' stdin sentidata '" + SentiStrengthLanguageFolder + "'")
    p = subprocess.Popen(shlex.split(
        "java -jar '" + SentiStrengthLocation + "' stdin sentidata '" + SentiStrengthLanguageFolder  + "'"),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    # communicate via stdin the string to be rated. Note that all spaces are replaced with +
    b = bytes(sentiString.replace(" ", "+"), 'utf-8')  # Can't send string in Python 3, must send bytes
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")  # convert from byte
    stdout_text = stdout_text.rstrip().replace("\t", " ")
    # print("{"+stdout_text.replace("+","")+"}")
    scores = stdout_text.split()[:2]
    if abs(int(scores[1])) >= abs(int(scores[0])):
        return scores[1]
    else:
        return scores[0]






# 分析每一句评论的情感指数
def RateSentiment1(file_name):
    SentiStrengthLocation = "../lib/SentiStrength.jar"  # SentiStrength.jar所在的路徑
    SentiStrengthLanguageFolder = "../lib/SentStrength_Data/"  # SentStrength_Data
    if not os.path.isfile(SentiStrengthLocation):
        print("SentiStrength not found at: ", SentiStrengthLocation)
    if not os.path.isdir(SentiStrengthLanguageFolder):
        print("SentiStrength data folder not found at: ", SentiStrengthLanguageFolder)

        # 使用shlex打开子进程以将命令行字符串转换为正确的args列表格式
    p = subprocess.Popen(f"java -jar ../lib/SentiStrength.jar sentidata ../lib/SentStrength_Data/ input {file_name}",
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,shell=True)
    # 通过stdin通信要进行评级的字符串。请注意，所有空格都替换为+
    stdout_byte, stderr_text = p.communicate()

    print(stdout_byte)
    print(stderr_text)
    return str(stdout_byte).strip(r'\r\n\'').split("in: ")[1]


#
if __name__ == '__main__':

    aa = ['I downloaded this application on my new phone so I can have the same ringtone I did on my old phone .','I saw a snippet of another review that complained that this application only has "  factory-like "  ringtones .','Most other ringtone apps do not have anything I would even consider using .','Could not be bothered to try and sort it out so uninstalled it .','Search function did not work .','The handful of tones I heard were awful .','This application is a complete waste of time just delete it and save your memory space !','Great quality and a wide variety of styles to choose from .','Ringtones are available in "  packs "  that must be downloaded before knowing what is in them .','I can not look up the music I want so if your going to make a application make it good','This is a horrible app , anything more or less popular , or classical , or pretty much anything at all you type in the search just does not appear to be on this app .','It is not a huge selection of tones , but I still found a few I liked .','The free ones only had a few ads to skip over .','I deleted something off my phone that cost 8 dollars .','Boring ringtones , if you are looking for fun pop ringtones then this is not the right application to be looking at','Super easy to use and changes your ringtone as fast as you can push the buttons .','Pop-up ads every 15 seconds .','I would like to use an application a few weeks before rating .','App sucks , searched for a artist and song title and shows Nothing , waste of a app .','It is great with slot of good ringtones']
    for u in aa:
        print(RateSentiment(u))
# #     print("ff")
#     config = {
#         'host': '127.0.0.1',
#         'port': 3306,
#         'user': 'root',
#         'database': 'review',
#         'passwd': 'root',
#         'charset': 'utf8mb4',
#         'cursorclass': pymysql.cursors.DictCursor
#     }
#
#     conn = pymysql.connect(**config)
#     cursor = conn.cursor()
#     sql = f'select review_date, avg(sentiment_score) as avg_sentiment  from app_sentence where app_id = \'com.google.android.apps.youtube.music\' group by review_date'
#     df = pd.read_sql(sql, conn)
#
#
#     h = df['avg_sentiment']
#     # h = [1,2,3,2,3,4,3,34,56,5,4,32,2,3,4,6,7,4,32,34]
#
#     peaks, troughs = get_peaks_troughs(df, 2)
#     print(peaks)
#     print(troughs)
#
#     plt.subplot(1, 2, 1)
#
#     plt.subplot(1, 2, 2)
#     plt.plot(df['review_date'], h)
#     for x, y in peaks:
#         plt.text(x, y, y, fontsize=10, verticalalignment="bottom", horizontalalignment="center")
#     for x, y in troughs:
#         plt.text(x, y, y, fontsize=10, verticalalignment="top", horizontalalignment="center")
#
#     plt.show()
    # print(RateSentiment("Sentiment analysis is the process of assigning a quantitative value of each positive or negative for each review [18]."))
    # file_name = RateSentiment1("F:/aaa.txt")
    # df = pd.read_csv(file_name,sep='\t',encoding='utf-8')
    # print(df.head(5))