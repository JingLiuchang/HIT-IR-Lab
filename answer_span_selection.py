from sklearn.feature_extraction.text import TfidfVectorizer
import utils
import pylcs
from tqdm import trange, tqdm
import jieba
from jieba import posseg
from scipy.linalg import norm
import numpy as np
from os.path import exists
from os import system
import json
import re
import metric
import thulac

svmrankresultpath='./prerpocessed/svmresult.json'
seg_passagepath='./prerpocessed/seg_passages.json'
passagedictpath='./prerpocessed/passages_multi_sentences_dict.json'
testdictpath='./prerpocessed/test_dict.json'
spansvmrank='./prerpocessed/spansvmrank.json'

thu1 = thulac.thulac()
def tokenize(sentence):
    sentence = sentence.replace(' ', '').replace('\n', '')
    seglist = jieba.cut(sentence, use_paddle=True)
    seg = ' '.join(seglist)
    seglist = seg.split(' ')
    return seglist

def repoccess():
    file=open('./prerpocessed/spansvmrank.json','w',encoding='utf-8')
    svmrankdatalist=utils.load_json(svmrankresultpath)
    passagesdict=utils.load_json_total(passagedictpath)
    querydict=utils.load_json_total(testdictpath)
    for row in tqdm(svmrankdatalist):
        qid=row['qid']
        pid=row['pid'][0]
        ans_seglist=row['answer_sentence']
        query_seg=row['query']
        row['spanseg_answer']=[]
        row['spanseg_query']=' '.join(jieba.cut(querydict[str(qid)],use_paddle=True))#问题不做去停用词
        for ans_seg in ans_seglist:
            i=0
            for sent in passagesdict[str(pid)]:
                if utils.tokenize_jieba(sent, modify=False)==ans_seg:
                    if sent[len(sent)-1] in [':','：']:
                        sent+=passagesdict[str(pid)][i+1]
                    if tokenize(sent) not in row['spanseg_answer']:
                        row['spanseg_answer'].append(tokenize(sent))
                    break
                i+=1
        '''for sent in passagesdict[str(pid)]:
            if utils.tokenize_jieba(sent, modify=False) in ans_seglist:
                if tokenize(sent) not in row['spanseg_answer']:
                    row['spanseg_answer'].append(tokenize(sent))'''
        if len(row['spanseg_answer'])!=len(ans_seglist):
            print('error',qid)
            break
        result_dp = json.dumps(row, ensure_ascii=False)
        file.write(result_dp + '\n')
    file.close()

#input格式:["1989", "年", "入选", "国家集训队", "教练", "章恒"]
def syntag(segsent_list):
    s = ''.join(segsent_list)
    word_tags = jieba.posseg.cut(s, use_paddle=True)
    words = []
    tags = []
    for item in word_tags:
        words.append(item.word)
        tags.append(item.flag)
    return words,tags
def thusyntag(segsent_list):
    s= ''.join(segsent_list)  # 默认模式
    text = thu1.cut(s, text=True)  # 进行一句话分词
    words = []
    tags = []
    text_lst = text.split(' ')
    for item in text_lst:
        item = item.split('_')
        words.append(item[0])
        tags.append(item[1])
    return words,tags

def _get_hum_ans(query_lst, ans_lst):
    query, ans, res = ''.join(query_lst), ''.join(ans_lst), ''
    #ans_lst,tags=syntag(ans_lst)
    ans_lst, tags = thusyntag(ans_lst)
    for idx, tag in enumerate(tags):
        if (tag == 'np' or tag == 'ni') and ans_lst[idx] not in res:#把答案中全部人名nr和机构名nt加入答案
            res += ans_lst[idx]
    return res
def HUMans(query_lst,ans_lsts):
    for ans_list in ans_lsts:
        res = _get_hum_ans(query_lst, ans_list)
        if res:
            break
    return res

def _get_loc_ans(query_lst, ans_lst):
    query, ans, res = ''.join(query_lst), ''.join(ans_lst), ''
    #ans_lst, tags = syntag(ans_lst)
    ans_lst, tags = thusyntag(ans_lst)
    for idx, tag in enumerate(tags):
        if (tag == 'ns' or tag == 'nl') and ans_lst[idx] not in res:#地名、处所
            res += ans_lst[idx]
    return res

def LOCans(query_lst,ans_lsts):
    for ans_list in ans_lsts:
        res = _get_loc_ans(query_lst, ans_list)
        if res:
            break
    return res

def _get_num_ans(query_lst: list, ans_lst: list):
    query, ans, res_lst = ''.join(query_lst), ''.join(ans_lst), []
    #ans_lst, tags = syntag(ans_lst)
    ans_lst, tags = thusyntag(ans_lst)
    for idx, tag in enumerate(tags):
        if tag == 'm' and idx < len(tags) - 1 and tags[idx + 1] == 'q':#选取数词+量词的形式
            res_lst.append(ans_lst[idx] + ans_lst[idx + 1])
    if res_lst:
        return res_lst[0]
    else:
        return ''

def NUMans(query_lst,ans_lsts):
    query=''.join(query_lst)
    checks=['时候', '年', '天', '周', '日', '多久', '时间','月']
    flag=0
    for check in checks:
        if check in query:
            flag=1
    if not flag:
        for ans_list in ans_lsts:
            res = _get_num_ans(query_lst, ans_list)
            if res:
                break
        return res
    else:
        res=TIMEans(query_lst,ans_lsts)
        return res



def _get_time_ans(query_lst: list, ans_lst: list):  # query_type：参考哈工大问题分类体系.
    query, ans, res_lst = ''.join(query_lst), ''.join(ans_lst), []
    mat = re.findall(r'\d{2,4}[年月日]?[-到至～]\d{2,4}[年月日]?', ans)
    if mat:
        return mat
    mat = re.findall(r'\d{1,4}[年/-]\d{1,2}[月/-]\d{1,2}[日号]?', ans)
    if mat:
        return mat
    mat = re.findall(r'(?:周|星期|礼拜)[1-7一二三四五六日]', ans)
    if mat:
        return mat
    mat = re.findall(r'\d{1,4}[年/-]\d{1,2}月?', ans)
    if mat:
        return mat
    mat = re.findall(r'\d{2,4}年', ans)
    if mat:
        return mat
    mat = re.findall(r'\d{1,2}月', ans)
    if mat:
        return mat
    mat = re.findall(r'\d{1,2}日}', ans)
    if mat:
        return mat
    return ['']

def TIMEans(query_lst,ans_lsts):
    for ans_list in ans_lsts:
        res = _get_time_ans(query_lst, ans_list)
        if res:
            break
    res = res[0]
    return res



def checkmh(ans):
    start=-1
    end=len(ans)
    i=0
    for word in ans:
        if word==':' or word=='：':
            start=i
        if word in ['.','。','!','！','?','？'] and start>0:
            end=i
            break
        i+=1
    if start>0:
        return ans[start + 1:end]
    else:
        return ''

def _get_objdesunk_ans(query_lst: list, ans_lsts: list):
    stop= ['在', '是', '的', '了', '由', '有', '于','叫']
    query= ''.join(query_lst)
    ans=''#答案句合并为一个,未分词
    anslist=[]#答案句，分词，并合并为一个,中间以sep分割
    ans=''.join(ans_lsts[0])
    res=checkmh(ans)
    if res:
        return res
    for ans_lst in ans_lsts:
        anslist+=ans_lst
        anslist.append('sep')

    if '什么' in query_lst:
        qlist_stop=[]
        for word in query_lst:
            if word not in stop:
                qlist_stop.append(word)
        end=qlist_stop.index('什么')
        start=end-2
        if start<0:
            start=0
        XX=qlist_stop[start:end]
        if len(XX)==1:
            XX.insert(0,'EMP')
        if len(XX)==0:
            XX=['EMP','EMP']
        if XX[0] in anslist and XX[1] in anslist and ((anslist.index(XX[1])-anslist.index(XX[0]))==1):
            ansstartindex=anslist.index(XX[1])
            while ansstartindex<len(anslist):
                res+=anslist[ansstartindex]
                ansstartindex+=1
                if ansstartindex>=len(anslist) or anslist[ansstartindex] in ['.','。','!','！','?','？','sep']:
                    break
        else:
            for i in range(len(anslist)):
                candidate = anslist[i:i + 2]
                a = set(XX)
                b = set(candidate)
                common = a & b
                if len(common) > 0:
                    ansstartindex = i+2
                    while ansstartindex<len(anslist):
                        res += anslist[ansstartindex]
                        ansstartindex += 1
                        if ansstartindex>=len(anslist) or anslist[ansstartindex] in ['.', '。', '!', '！', '?', '？', 'sep']:
                            break
                    break
        if res:
            return res

    if '哪' in query:
        qlist_stop = []
        for word in query_lst:
            if word not in stop:
                qlist_stop.append(word)
        end=0
        for i in range(len(qlist_stop)):
            #print(i,qlist_stop[i])
            #print(query_lst,ans_lsts)
            if qlist_stop[i][0]=='哪':
                end=i
                break
        start=end-2
        if start<0:
            start=0
        XX = qlist_stop[start:end]
        YY=qlist_stop[end+1]
        if len(XX)==1:
            XX.insert(0,'EMP')
        if len(XX)==0:
            XX=['EMP','EMP']
        if XX[0] in anslist and XX[1] in anslist and ((anslist.index(XX[1])-anslist.index(XX[0]))==1):
            ansstartindex=anslist.index(XX[1])
            while ansstartindex<len(anslist):
                res+=anslist[ansstartindex]
                ansstartindex+=1
                if ansstartindex>=len(anslist) or anslist[ansstartindex] in ['.','。','!','！','?','？','sep'] or anslist[ansstartindex]==YY:
                    break
        else:
            for i in range(len(anslist)):
                candidate = anslist[i:i + 2]
                a = set(XX)
                b = set(candidate)
                common = a & b
                if len(common) > 0:
                   ansstartindex = i+2
                   #print(anslist)
                   while ansstartindex<len(anslist):
                       #print(ansstartindex)
                       res += anslist[ansstartindex]
                       ansstartindex += 1
                       if ansstartindex>=len(anslist) or anslist[ansstartindex] in ['.', '。', '!', '！', '?', '？', 'sep'] or anslist[ansstartindex]==YY:
                           break
        if res:
            return res

    res=''.join(ans_lsts[0])
    return res

def get_ans(query_lst,query_class,ans_lsts):
    res=''
    if query_class=='HUM':
        res=HUMans(query_lst,ans_lsts)
        if res:
            return res
    elif query_class=='NUM':
        res=NUMans(query_lst,ans_lsts)
        if res:
            return res
    elif query_class=='TIME':
        res=TIMEans(query_lst,ans_lsts)
        if res:
            return res
    elif query_class=='LOC':
        res=LOCans(query_lst,ans_lsts)
        if res:
            return res
    elif query_class == 'OBJ' or query_class == 'DES' or query_class == 'UNKNOW':
        res=_get_objdesunk_ans(query_lst,ans_lsts)
        if not res:
            print('error')
        return res
    if not res:
        res=_get_objdesunk_ans(query_lst,ans_lsts)
        return res
def evaluate():
    data=utils.load_json('./data/train.json')
    classdata=utils.load_json_total('./prerpocessed/trainclassresult.json')
    Blue=0
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
            sent_lst=tokenize(sent)
            ans_lsts.append(sent_lst)
        res_hat=get_ans(query_lst,classtype,ans_lsts)
        blue=metric.bleu1(res_hat,res_tru)
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


def predict():
    outputfile=open('./prerpocessed/test_answer.json','w',encoding='utf-8')
    data=utils.load_json(spansvmrank)
    testdict=utils.load_json_total(testdictpath)
    for row in tqdm(data):
        output={}
        qid=row['qid']
        question=testdict[str(qid)]
        answer_pid=row['pid']
        query_class=row['class']
        temp=row['spanseg_query'].split(' ')
        query_list = [x for x in temp if x != '']
        ans_lists=row['spanseg_answer']
        res=get_ans(query_list,query_class,ans_lists)
        output['qid']=qid
        output['question']=question
        output['answer_pid']=answer_pid
        output['answer']=res
        result_dp = json.dumps(output,ensure_ascii=False)
        outputfile.write(result_dp + '\n')
    outputfile.close()









if __name__=='__main__':
    evaluate()
    '''评价blue 0.5079603172537755
HUM  blue 0.5359934947455969
NUM  blue 0.46654554593546177
LOC  blue 0.4211321132747889
TIME  blue 0.6599027770204183
OTHER  blue 0.5052266572906117'''



