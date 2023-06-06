#!/usr/bin/python
# coding:utf-8
from tqdm import trange, tqdm
import answer_span_selection
import utils
import jieba
"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: metric.py
@time: 2019/6/1 15:59
实验3.4 评价方式包含三个指标，主要看字符级别的bleu1值，其他供参考
1. precision，recall，F1值：取所有开发集上的平均
2. EM（exact match）值：精确匹配的准确率
3. 字符级别的bleu1值
"""

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu

def precision_recall_f1(prediction, ground_truth):
    """
    计算预测答案prediction和真实答案ground_truth之间的字符级别的precision，recall，F1值，
    Args:
        prediction: 预测答案（未分词的字符串）
        ground_truth: 真实答案（未分词的字符串）
    Returns:
        floats of (p, r, f1)
    eg:
    >>> prediction = '北京天安门'
    >>> ground_truth = '天安门'
    >>> precision_recall_f1(prediction, ground_truth)
    >>> (0.6, 1.0, 0.7499999999999999)
    """
    #     # 对于中文字符串，需要在每个字之间加空格
    #     prediction = " ".join(prediction)
    #     ground_truth = " ".join(ground_truth)

    #     prediction_tokens = prediction.split()
    #     ground_truth_tokens = ground_truth.split()

    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction)
    r = 1.0 * num_same / len(ground_truth)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


def exact_match(all_prediction, all_ground_truth):
    """
    计算所有预测答案和所有真实答案之间的准确率
    Args:
        all_prediction: 所有预测答案（数组）
        all_ground_truth: 所有真实答案（数组）
    Returns:
        floats of em
    eg:
    >>> all_prediction = ['答案A', '答案B', '答案C']
    >>> all_ground_truth = ['答案A', '答案B', '答案D']
    >>> exact_match(all_prediction, all_ground_truth)
    >>> 0.6666666666666666
    """
    assert len(all_prediction) == len(all_ground_truth)
    right_count = 0
    for pred_answer, true_answer in zip(all_prediction, all_ground_truth):
        if pred_answer == true_answer:
            right_count += 1
    return 1.0 * right_count / len(all_ground_truth)


def bleu1(prediction, ground_truth):
    '''
    计算单个预测答案prediction和单个真实答案ground_truth之间的字符级别的bleu1值,(可能会有warning， 不用管)
    Args:
        prediction: 预测答案（未分词的字符串）
        ground_truth: 真实答案（未分词的字符串）
    Returns:
        floats of bleu1
    eg:
    >>> prediction = '北京天安门'
    >>> ground_truth = '天安门'
    >>> bleu1(prediction, ground_truth)
    >>> 0.6
    '''
    prediction = ' '.join(prediction).split()
    ground_truth = [' '.join(ground_truth).split()]
    bleu1 = sentence_bleu(ground_truth, prediction, weights=(1, 0, 0, 0))
    return bleu1

def eval():
    data=utils.load_json('./data/train.json')
    classdata=utils.load_json_total('./prerpocessed/trainclassresult.json')
    Blue=10
    numcnt=0
    humcnt=0
    loccnt=0
    timecnt=0
    othercnt=0
    Blue_num=0
    Blue_hum=0
    Blue_loc=0
    Blue_time=0
    Blue_objdesunk=0
    for row in tqdm(data):
        qid=row['qid']
        query=row['question']
        answers=row['answer_sentence']
        res_tru=row['answer']
        classtype=classdata[str(qid)]
        query_lst_temp=(' '.join(jieba.cut(query,use_paddle=True))).split(' ')
        query_lst=[]
        for word in query_lst_temp:
            if not word=='':
                query_lst.append(word)
        ans_lsts=[]
        for sent in answers:
            sent_lst=answer_span_selection.tokenize(sent)
            ans_lsts.append(sent_lst)
        res_hat=answer_span_selection.get_ans(query_lst,classtype,ans_lsts)
        blue=bleu1(res_hat,res_tru)
        Blue+=blue
        if classtype=='NUM':
            Blue_num+=blue
            numcnt+=1
        if classtype=='HUM':
            Blue_hum+=blue
            humcnt+=1
        if classtype=='LOC':
            Blue_loc+=blue
            loccnt+=1
        if classtype=='TIME':
            Blue_time += blue
            timecnt += 1
        if classtype=='OBJ' or classtype=='DES' or classtype=='UNKNOW':
            Blue_objdesunk+=blue
            othercnt+=1

    print('评价blue',Blue/len(data))
    print('HUM  blue', Blue_hum / humcnt)
    print('NUM  blue', Blue_num / numcnt)
    print('LOC  blue', Blue_loc / loccnt)
    print('TIME  blue', Blue_time / timecnt)
    print('OTHER  blue', Blue_objdesunk / othercnt)