from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import utils
import joblib
import jieba
from tqdm import trange, tqdm
import json
def prepare_data(trainpath,testpath,labelflag,trainflag,testflag):
    if trainflag:
        traindata = utils.load_classdata(trainpath)
        x_train = traindata['answer']
        if labelflag == 'top':
            y_train = traindata['toplabel']
        else:
            y_train = traindata['fulllabel']
    if testflag:
        testdata = utils.load_classdata(testpath)
        x_test = testdata['answer']
        if labelflag == 'top':
            y_test = testdata['toplabel']
        else:
            y_test = testdata['fulllabel']

    if trainflag and testflag:
        return x_train,y_train,x_test,y_test
    elif trainflag:
        return x_train,y_train
    else:
        return x_test,y_test



class Classification():
    def __init__(self):
        self.traindatapath = './question_classification/trian_questions.txt'
        self.testdatapath = './question_classification/test_questions.txt'
        self.labelencoder=LabelEncoder()
        self.tfidfvectorizer=TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        x_train,y_train,x_test,y_test=prepare_data(self.traindatapath,self.testdatapath,labelflag='top',trainflag=True,testflag=True)
        self.labelencoder.fit(y_train+y_test)
        self.tfidfvectorizer.fit(x_train+x_test)
        self.model=MLPClassifier(solver='lbfgs')#0.8942965779467681
        #self.model=LogisticRegression(multi_class='multinomial',max_iter=150)#也是0.89略有差别
        print('initialized')
    def train(self):
        x_train, y_train = prepare_data(self.traindatapath, self.testdatapath, labelflag='top', trainflag=True,
                                        testflag=False)
        x_train_tf=self.tfidfvectorizer.transform(x_train)
        y_train_en=self.labelencoder.transform(y_train)
        self.model.fit(x_train_tf,y_train_en)

    def eval(self):
        x_test,y_test=prepare_data(self.traindatapath, self.testdatapath, labelflag='top', trainflag=False,
                                        testflag=True)
        print(x_test)
        x_test_tf = self.tfidfvectorizer.transform(x_test)
        y_test_en = self.labelencoder.transform(y_test)

        y_hat=self.model.predict(x_test_tf)
        score = metrics.accuracy_score(y_test_en, y_hat)
        matrix = metrics.confusion_matrix(y_test_en, y_hat)
        report = metrics.classification_report(y_test_en, y_hat)
        print('准确率\n', score)
        print('召回率\n', report)


    def predict(self):
        resultfile = open('./prerpocessed/BM25classresult.json', 'w', encoding='utf-8')
        testdatas = utils.load_json('./data/test.json')
        predictx=[]
        for testdata in tqdm(testdatas):
            predictx.append(' '.join(jieba.cut(testdata['question'],use_paddle=True)))

        predictx_tf=self.tfidfvectorizer.transform(predictx)
        y_hat_en = self.model.predict(predictx_tf)
        y_hat=self.labelencoder.inverse_transform(y_hat_en)
        bm25resultlist=utils.load_json('./prerpocessed/BM25result.json')
        for i in range(len(y_hat)):
            bm25resultlist[i]['class']=y_hat[i]
            result_dp = json.dumps(bm25resultlist[i], ensure_ascii=False)
            resultfile.write(result_dp + '\n')

        resultfile.close()

    def predictontrain(self):
        resultfile = open('./prerpocessed/trainclassresult.json', 'w', encoding='utf-8')
        traindatas = utils.load_json('./data/train.json')
        predictx=[]
        for traindata in tqdm(traindatas):
            predictx.append(' '.join(jieba.cut(traindata['question'],use_paddle=True)))

        predictx_tf=self.tfidfvectorizer.transform(predictx)
        y_hat_en = self.model.predict(predictx_tf)
        y_hat=self.labelencoder.inverse_transform(y_hat_en)
        print(len(y_hat))
        print(len(traindatas))
        result = {}
        for i in range(len(y_hat)):
            result[traindatas[i]['qid']]=y_hat[i]
        json.dump(result, resultfile, ensure_ascii=False)
        resultfile.close()

if __name__=='__main__':
    #classmodel=Classification()
    #classmodel.train()
    #joblib.dump(classmodel,'./model/classification.pkl')
    classmodel=joblib.load('./model/classification.pkl')
    #classmodel.eval()
    #classmodel.predict()
    classmodel.eval()













