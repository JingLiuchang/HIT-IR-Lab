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
import json

class svmrankmodel():
    def __init__(self):
        self.train_feature_path = './prerpocessed/trainfeature'
        self.test_feature_path = './prerpocessed/testfeature'
        self.dev_feature_path = './prerpocessed/devfeature'
        self.seg_passages_path = 'prerpocessed/seg_passages.json'
        self.train_path = 'data/train.json'
        self.test_path = './prerpocessed/BM25result.json'
        self.model='./model/svmrank'
        self.devresult='./prerpocessed/devresult'
        self.testresult = './prerpocessed/testresult'
        self.testclass_path='./prerpocessed/BM25classresult.json'

    #下面是特征计算部分，默认输入格式为"我 喜欢 天安门 广场"
    #句内特征
    #句内词数目
    def cal_tokennum(self,s):
        s=s.split(' ')
        return len(s)
    # 句内实词数目
    def cal_entitynum(self, s):
        s = s.replace(' ', '')
        entityflag = ['a', 'n', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v']
        word_tags = jieba.posseg.cut(s, use_paddle=True)
        cnt = 0
        for word_tag in word_tags:
            if word_tag.flag in entityflag:
                cnt += 1
        return cnt
    #句间特征
    #句间LCS
    def cal_lcs(self,s1,s2):
        s1=s1.replace(' ','')
        s2 = s2.replace(' ', '')
        length = pylcs.lcs_sequence_length(s1, s2)
        if min(len(s1),len(s2))==0:
            print('s1',s1)
            print('s2',s2)
        return length/min(len(s1),len(s2))
    #句间uni、bi共现
    def cal_gram(self,s1,s2):
        s1 = s1.split(' ')
        s2 = s2.split(' ')
        unigram = len([word for word in s1 if word in s2]) / len(s2)
        s1_list = [preword + subword for preword, subword in zip(s1[:-1], s1[1:])]
        s2_list = [preword + subword for preword, subword in zip(s2[:-1], s2[1:])]
        bigram = len([bi_word for bi_word in s1_list if bi_word in s2_list]) / (len(s2_list) + 1)
        return unigram, bigram
    #句间编辑距离
    def cal_editdistance(self,s1,s2):#pylcs库没有进行词对齐，是按照字符为单位计算的，可以尝试改进
        dis=pylcs.edit_distance(s1,s2)
        return dis
    #句间词数目差异
    def cal_tokennumbias(self,s1,s2):
        num1=self.cal_tokennum(s1)
        num2=self.cal_tokennum(s2)
        return abs(num1-num2)
    #句间tfidf余弦相似度
    def cal_tfidfcos(self,s1,s2,tfidf):
        S1=[]
        S1.append(s1)
        S2 = []
        S2.append(s2)
        s1_tfidf=tfidf.transform(S1).toarray()
        s2_tfidf=tfidf.transform(S2).toarray()
        down=norm(s1_tfidf)*norm(s2_tfidf)
        if down:
            up=np.dot(s1_tfidf,s2_tfidf.T)
            return up/down
        else:
            return 0

    def get_features(self,seg_question,seg_answer,tfidf):
        feature=[]
        feature.append('1:%d' % self.cal_tokennum(seg_answer))
        feature.append('2:%d' % self.cal_entitynum(seg_answer))
        feature.append('3:%f' % self.cal_lcs(seg_answer,seg_question))
        feature.append('4:%f' % self.cal_gram(seg_answer,seg_question)[0])
        feature.append('5:%f' % self.cal_gram(seg_answer, seg_question)[1])
        feature.append('6:%d' % self.cal_editdistance(seg_answer, seg_question))
        feature.append('7:%d' % self.cal_tokennumbias(seg_answer, seg_question))
        feature.append('8:%d' % self.cal_tfidfcos(seg_answer, seg_question,tfidf))
        return feature

    def make_testfeature(self):
        seg_passages = utils.load_json_total(self.seg_passages_path)
        testlist = utils.load_json(self.test_path)
        feature_list = []
        for row in tqdm(testlist):  # 遍历文件中的每一行query信息
            qid = row['qid']
            pid = row['pid'][0]
            seg_query = row['query']
            features = []
            vectrain = []
            for seglist in seg_passages[str(pid)]:  # 制造tf-idf训练集
                vectrain.append(' '.join(seglist))
            tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_vectorizer.fit(vectrain)
            for seg_list in seg_passages[str(pid)]:
                feature = ' '.join(self.get_features(seg_query, ' '.join(seg_list), tfidf_vectorizer))
                features.append('0 qid:%d %s' % (qid, feature))
            feature_list.append(features)
        feature_list.sort(key=lambda lst: int(lst[0].split()[1].split(':')[1]))
        with open(self.test_feature_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join([feature for features in feature_list for feature in features]))

    def make_trainfeature(self,trainrate):
        seg_passages = utils.load_json_total(self.seg_passages_path)
        trainlist = utils.load_json(self.train_path)
        feature_list = []
        # seg_passages, res_lst, feature_lst = load_seg_passages(), read_json(train_path), []
        for row in tqdm(trainlist):  # 遍历train.json文件中的每一行query信息
            # item:{"question": "罗静恩韩文名字是什么？", "pid": 6726, "answer_sentence": ["韩文:나경은"], "answer": "나경은", "qid": 2213}
            features = []
            qid = row['qid']
            pid = row['pid']
            seg_query_list = utils.tokenize_jieba(row['question'], False)
            seg_ans_lists = [utils.tokenize_jieba(sent, False) for sent in row['answer_sentence']]
            vectrain = []
            for seglist in seg_passages[str(pid)]:  # 制造tf-idf训练集
                vectrain.append(' '.join(seglist))
            tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_vectorizer.fit(vectrain)
            for seg_list in seg_passages[str(pid)]:
                feature = ' '.join(self.get_features(' '.join(seg_query_list), ' '.join(seg_list), tfidf_vectorizer))
                if seg_list in seg_ans_lists:
                    score = 3
                else:
                    score = 1
                features.append('%d qid:%d %s' % (score, qid, feature))
            feature_list.append(features)

        feature_list.sort(key=lambda lst: int(lst[0].split()[1].split(':')[1]))

        slide = int(len(feature_list) * trainrate)
        train_feature_list = feature_list[:slide]
        dev_feature_list = feature_list[slide:]
        with open(self.train_feature_path, 'w', encoding='utf-8') as trainfile:
            with open(self.dev_feature_path, 'w', encoding='utf-8') as devfile:
                trainfile.write('\n'.join([feature for feature_lst in train_feature_list for feature in feature_lst]))
                devfile.write('\n'.join([feature for feature_lst in dev_feature_list for feature in feature_lst]))


    def trainmodel(self):
        train_cmd = f'.\model\svm_rank_windows\svm_rank_learn.exe -c 200.0 {self.train_feature_path} {self.model}'
        system(train_cmd)
        predict_cmd = f'.\model\svm_rank_windows\svm_rank_classify.exe {self.dev_feature_path} {self.model} {self.devresult}'
        system(predict_cmd)

    def eval(self):
        with open(self.dev_feature_path, 'r', encoding='utf-8') as f1, open(self.devresult, 'r', encoding='utf-8') as f2:
            y_true, y_predict, right1,right2,right3 = {}, {}, 0,0,0
            for line1, line2 in zip(f1, f2):
                if len(line1) == 1:
                    break
                qid = int(line1.split()[1].split(':')[1])
                lst1, lst2 = y_true.get(qid, []), y_predict.get(qid, [])
                lst1.append((int(line1[0]), len(lst1)))
                lst2.append((float(line2.strip()), len(lst2)))
                y_true[qid], y_predict[qid] = lst1, lst2

            mmr=0
            for qid in y_true:
                truelist = y_true[qid] # 按照val大小排序
                rankscorelist= sorted(y_predict[qid], key=lambda item: item[0], reverse=True)
                targeindex=[]#turelist中全部权重为3的序号作为正确集
                '''
                flag=0
                flaghave3=0
                for item in lst1:
                    if flag==0 and item[0]==3:
                        flag=1
                        flaghave3=1
                        continue
                    if flag==1 and item[0]==3:
                        print('found',lst1)
                        print(qid)
                '''
                for item in truelist:
                    if item[0]==3:
                        targeindex.append(item[1])
                if rankscorelist[0][1] in targeindex:#预测结果中top1在正确集中
                    right1+=1
                if (rankscorelist[0][1] in targeindex) or (rankscorelist[1][1] in targeindex):#top2在正确集
                    right2+=1
                if (rankscorelist[0][1] in targeindex) or (rankscorelist[1][1] in targeindex) or (rankscorelist[2][1] in targeindex):#top3在正确集中
                    right3+=1
                for i in range(len(rankscorelist)):
                    if rankscorelist[i][1] in targeindex:
                        mmr+=1/(i+1)
                        break

            print('MRR',mmr/len(y_true))
            print('top1',right1/len(y_true))
            print('top2',right2/len(y_true))
            print('top3',right3 / len(y_true))

    def predict(self,top):  # num表示抽取的答案句数目
        predict_cmd = f'.\model\svm_rank_windows\svm_rank_classify.exe {self.test_feature_path} {self.model} {self.testresult}'
        system(predict_cmd)
        resultfile=open('./prerpocessed/svmresult.json','w',encoding='utf-8')
        with open(self.test_feature_path, 'r', encoding='utf-8') as f1, open(self.testresult, 'r', encoding='utf-8') as f2:
            labels = {}
            for line1, line2 in zip(f1, f2):
                if len(line1) == 1:
                    break
                qid = int(line1.split()[1].split(':')[1])
                if qid not in labels:
                    labels[qid] = []
                labels[qid].append((float(line2.strip()), len(labels[qid])))

            seg_passages=utils.load_json_total(self.seg_passages_path)
            testlist=utils.load_json(self.testclass_path)
            for row in tqdm(testlist):  # 遍历文件中的每一行query信息
                qid=row['qid']
                pid=row['pid'][0]
                seg_query=row['query']
                row['answer_sentence']=[]
                scoreranklist=sorted(labels[qid], key=lambda val: val[0], reverse=True)
                seg_passage=seg_passages[str(pid)]
                for result in scoreranklist[:top]:
                    index=result[1]
                    if seg_passage[index] not in row['answer_sentence']:
                        row['answer_sentence'].append(seg_passage[index])
                result_dp = json.dumps(row, ensure_ascii=False)
                resultfile.write(result_dp + '\n')
        resultfile.close()


def run():
    model=svmrankmodel()
    makefeatureflag=False
    trainflag=True
    testflag=False
    if makefeatureflag:
        model.make_trainfeature(trainrate=0.9)
        model.make_testfeature()
    if trainflag:
        model.trainmodel()
        model.eval()
    if testflag:
        model.predict(top=3)


if __name__=='__main__':
    model=svmrankmodel()
    model.eval()
    '''MMR
    0.59530216389686
    top1
    0.4253731343283582
    top2
    0.5932835820895522
    top3
    0.7164179104477612'''