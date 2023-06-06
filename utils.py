# encoding=utf-8
import jieba
from ltp import LTP
import torch
import json
from tqdm import trange, tqdm
def contain_chinese(check_str):
    '''

    :param check_str: 待检查字符串
    :return: 是否含中文
    '''
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False
def after_modify(seglist):
    '''

    :param seglist: 初步分词的结果
    :return: 后处理后的结果
    '''
    new_tokenlist=[]
    for word in seglist:
        if contain_chinese(word):
            new_tokenlist.append(word)
        else:
            if bool(word.isdigit()):
                try:
                    a = int(word)
                    if a <= 2023 and a >= 2020:
                        new_tokenlist.append(word)
                except ValueError:
                    new_tokenlist.append(word)
    return new_tokenlist


def tokenize_jieba(sentence,modify):
    '''

    :param sentence: 待分词句
    :return: 分词
    '''
    sentence=sentence.replace(' ','').replace('\n','')
    seg=jieba.cut(sentence,use_paddle=True)
    seg=' '.join(seg)
    seglist=seg.split(' ')
    with open('stopwords.txt', "r",encoding='utf-8') as words:
        stopwords = [i.strip() for i in words]
    stopwords.extend(['-','【','】','，','.','（','）','——','(',')','〔','〕','—','\xa0','--','©','@',':','/'])
    tokenlist=[]
    for word in seglist:
        if word not in stopwords:
            tokenlist.append(word)
    if modify:
        tokenlist = after_modify(tokenlist)  # 去掉无意义的数字、字母串
    return tokenlist

#ltp分词的效果和速度相对不及jieba的基于paddle的预训练模型，分词模块使用jieba的tokenizer

def tokenize_ltp(sentence,modify):
    '''

    :param sentence: 待分词句子
    :return:分词
    '''
    ltp = LTP("LTP/small")  # 默认加载 Small 模型
    # 将模型移动到 GPU 上
    if torch.cuda.is_available():
        # ltp.cuda()
        ltp.to("cuda")
    output = ltp.pipeline([sentence], tasks=["cws"])
    # 使用字典格式作为返回结果
    seglist=output.cws[0]
    with open('stopwords.txt', "r",encoding='utf-8') as words:
        stopwords = [i.strip() for i in words]
    stopwords.extend(['-','【','】','，','.','（','）','——','(',')','〔','〕','—'])
    tokenlist=[]
    for word in seglist:
        if word not in stopwords:
            tokenlist.append(word)
    if modify:
        tokenlist = after_modify(tokenlist)  # 去掉无意义的数字、字母串
    return tokenlist
'''def tokenize_jieba(sentence,modify):
    sentence=sentence.replace(' ','').replace('\n','')
    seg=jieba.cut(sentence,use_paddle=True)
    seg=' '.join(seg)
    seglist=seg.split(' ')
    with open('stopwords.txt', "r",encoding='utf-8') as words:
        stopwords = [i.strip() for i in words]
    stopwords.extend(['-','【','】','，','.','（','）','——','(',')','〔','〕','—','\xa0','--','©','@',':','/'])
    tokenlist=[]
    for word in seglist:
        if word not in stopwords:
            tokenlist.append(word)
    if modify:
        tokenlist = after_modify(tokenlist)  # 去掉无意义的数字、字母串
    return tokenlist'''

def load_json(file_path):
    '''

    :param file_path: 待读取data路径
    :return: list形式的数据，一个元素为一行
    '''
    pages_dic = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            p = json.loads(line)
            pages_dic.append(p)
    return pages_dic

def load_classdata(data_path):
    dt = {'id':[],'fulllabel':[],'toplabel':[],'sublabel':[],'answer':[]}
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            [label, sent] = line.strip().split('\t')
            dt['answer'].append(' '.join(jieba.cut(sent,use_paddle=True)))#问句一般比较短,不适合做太多处理'a_b_c'形式
            dt['fulllabel'].append(label)
            dt['toplabel'].append(label.split('_')[0])
            dt['sublabel'].append(label.split('_')[1])
    dt['id'] = [i for i in range(len(dt['answer']))]
    return dt

def prepoccess(path):
    file=open('./prerpocessed/seg_passages.json','w',encoding='utf-8')
    data=load_json(path)
    seg_data = {}
    #print(data[1])
    for line in tqdm(data):
        pid=line['pid']
        sents=line['document']
        seg_data[pid]=[]
        for sent in sents:
            seg_sent=tokenize_jieba(sent,modify=False)
            if len(seg_sent)!=0:
                seg_data[pid].append(seg_sent)
    json.dump(seg_data,file,ensure_ascii=False)

def load_json_total(path):
    '''
    :param path:path内json数据整个是一个完整的json格式
    :return:
    '''
    with open(path,'r',encoding='utf-8') as file:
        data=json.load(file)
    return data

def dictlize(path):
    data=load_json(path)
    file=open('./prerpocessed/passages_multi_sentences_dict.json','w',encoding='utf-8')
    result = {}
    for row in tqdm(data):
        pid=row['pid']
        doclist=row['document']
        result[pid]=doclist
    json.dump(result, file, ensure_ascii=False)
    file.close()

def dictlizetest(path):
    data = load_json(path)
    file = open('./prerpocessed/test_dict.json', 'w', encoding='utf-8')
    result = {}
    for row in tqdm(data):
        qid = row['qid']
        doclist = row['question']
        result[qid] = doclist
    json.dump(result, file, ensure_ascii=False)
    file.close()



if __name__=='__main__':
    path='./data/test.json'
    dictlizetest(path)
