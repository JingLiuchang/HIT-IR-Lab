import pandas as pd
import utils
from collections import Counter
import time
import joblib
from tqdm import trange, tqdm
import math
import heapq
import json

class BM25():
    def __init__(self,datapath):
        '''

        :param datapath: 训练文本地址
        '''
        self.k1=1.5
        self.k3=1.5
        self.b=0.75
        self.epsilon=0.1
        self.inverted_index={}#倒排检索表word:{pid:fi}
        self.total_N=0#句子总数
        self.length_of={}#句子的长度字典pid:len
        self.datapath=datapath
        self.w=0#用于处理idf值为负时

    def train(self):
        print('Loading data……')
        start=time.time()
        datas =utils.load_json(self.datapath)#list
        self.total_N=0
        length_sum=0
        for data in tqdm(datas):#遍历文档集合
            '''
            if (data['pid']+1)%100==0:
                print(data['pid'],'trained')
            '''
            self.total_N+=1#文档数+1
            paragraph=''.join(data['document'])#连接句子
            seg_p=utils.tokenize_jieba(paragraph,modify=False)
            self.length_of[data['pid']]=len(seg_p)
            length_sum+=len(seg_p)
            words_cnt=Counter(seg_p)
            for word,cnt in words_cnt.items():
                if word not in self.inverted_index.keys():
                    self.inverted_index[word]={}
                    self.inverted_index[word][data['pid']]=cnt
                else:
                    self.inverted_index[word][data['pid']]=cnt
        self.length_of['avg']=length_sum/self.total_N#平均长度
        end=time.time()

        if self.total_N%2==0:
            self.w=math.log(self.total_N-self.total_N/2+1+0.5)-math.log(self.total_N/2-1+0.5)
        else:
            self.w=math.log(self.total_N-math.floor(self.total_N/2)+0.5)-math.log(math.floor(self.total_N/2)+0.5)

        print('done training! time used:', end - start)



    def cal_secore(self,seg_query,pid):
        '''

        :param seg_query: 分词好的句子
        :param pid: 候选文章pid号
        :return: 句子和文章间的bm25得分
        '''
        #W={}
        #R={}
        score=0

        for q,q_cnt in Counter(seg_query).items():
            if q in self.inverted_index.keys():
                dq=len(self.inverted_index[q].keys())#document number that contains q
            else:
                dq=0
            idf_temp=math.log(self.total_N-dq+0.5)-math.log(dq+0.5)
            if idf_temp<=0:
                idf_temp=self.w*self.epsilon
            try:
                q_cnt_document=self.inverted_index[q][pid]
            except KeyError:
                q_cnt_document=0
            r_temp1=((self.k1+1)*q_cnt_document)/(q_cnt_document+self.k1*(1-self.b+self.b*(self.length_of[pid]/self.length_of['avg'])))
            r_temp2=((self.k3+1)*q_cnt)/(self.k3+q_cnt)
            r_temp=r_temp1*r_temp2

            score+=r_temp*idf_temp

        return score

    def bm25_topk(self,query,k):
        '''

        :param query: 问句（未分词）
        :param k: 返回得分top k的
        :return: 得分top k的答案的pid号
        '''
        score={}
        seg_q=utils.tokenize_jieba(query,modify=False)
        #print('doing inverse_index search……')
        query_pids=[]
        for q in seg_q:
            if q in self.inverted_index.keys():
                query_pids.extend(self.inverted_index[q].keys())
        query_pids=list(set(query_pids))#倒排检索的pid
        #print('doing bm25 search……')
        for pid in query_pids:
            score[pid]=self.cal_secore(seg_q,pid)
        tpk_pids=heapq.nlargest(k,score,key=lambda x:score[x])
        return tpk_pids

def eval(k,bm25):
    '''

    :param k: 返回top k个答案
    :param bm25: 模型
    :return: 打印准确
    '''
    traindatas = utils.load_json('./data/train.json')
    cnt_true = 0
    cnt = 0
    print('applying model on train.json')
    for traindata in tqdm(traindatas):
        cnt += 1
        query = traindata['question']
        target_pid = traindata['pid']
        pred_pid = bm25.bm25_topk(query, k)
        if target_pid in pred_pid:
            cnt_true += 1
    print('model P:',cnt_true / cnt)

def predict(k,bm25):
    resultfile=open('./prerpocessed/BM25result.json','w',encoding='utf-8')
    testdatas = utils.load_json('./data/test.json')
    print('applying model on test.json')
    for testdata in tqdm(testdatas):
        result={}
        query = testdata['question']
        qid = testdata['qid']
        pred_pid = bm25.bm25_topk(query, k)
        result['qid']=qid
        result['query']=' '.join(utils.tokenize_jieba(query,modify=False))
        result['pid']=pred_pid
        result_dp= json.dumps(result, ensure_ascii=False)
        resultfile.write(result_dp + '\n')
    resultfile.close()

def run(trainflag,testflag,data_path,save_path):
    '''

    :param trainflag: true if need training
    :param testflag: true if need testing
    :param data_path: 训练数据
    :param save_path: 模型保存地址
    :return:
    '''
    if trainflag:
        bm25=BM25(data_path)
        bm25.train()
        print('saving model at '+save_path)
        joblib.dump(bm25,save_path)
        print('done training, bm25 model saved at '+save_path)
    print('loading model……')
    bm25=joblib.load(save_path)
    print('bm25&inverse_index model loaded')
    if testflag:
        #0.89
        eval(1,bm25)
        #0.924
        eval(2,bm25)
        #0.925
        #eval(3,bm25)
        #0.938
        #eval(5,bm25)
        #0.944
        #eval(7,bm25)
    if predictflag:
        predict(1,bm25)

if __name__=='__main__':
    trainflag=False
    testflag=True
    predictflag=False
    data_path='./data/passages_multi_sentences.json'
    save_path='./model/bm25.pkl'
    run(trainflag,testflag,data_path,save_path)