from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics as mt
import numpy  as np
from joblib import dump, load
from labelling import  SubsetLabeller
from encoder import FrequencyEncoder, NGramEncoder, TF_IDFEncoder, BagOfWordsEncoder
from split_data import get_train_test
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

def eval_classifier(classifier, X_test, y_test, conf_mat=False, y_pred = None):
    if y_pred is None:     
        y_pred = classifier.predict(X_test)
    print(f"accuracy: {mt.accuracy_score(y_test, y_pred):.3f}", end=';| ')
    print(f"recall: {mt.recall_score(y_test, y_pred,average='macro'):.3f}", end=';| ')
    print(f"precision: {mt.precision_score(y_test, y_pred,average='macro'):.3f}", end=';| ')
    print(f"f1: {mt.f1_score(y_test, y_pred,average='macro'):.3f}")
    if conf_mat:
        conf_mat = mt.confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(np.log(conf_mat + 1))
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("log(predictions)", rotation=-90, va="bottom")
        plt.xlabel("label prediction")
        plt.ylabel("true label")
        plt.show()
        
def cross_eval_classifier(classifier, X_train, y_train):
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring="accuracy", verbose=1)
    print(scores, np.average(scores))


def find_end_clf_combination():
    X_train_, _, y_train, _ = get_train_test('basic_split-small')

    print('labelling...')
    labeler = SubsetLabeller().load()
    y_train = labeler.label(y_train)

    encoders = [
                BagOfWordsEncoder(name_flag='-small'), 
                FrequencyEncoder(name_flag='-small'), 
                TF_IDFEncoder(name_flag='-small'),
                NGramEncoder(2, name_flag='-small'),
                NGramEncoder(3, name_flag='-small'),
                NGramEncoder(4, name_flag='-small'),
                NGramEncoder(5, name_flag='-small')
                ]
    for encoder_ in encoders:
        encoder = encoder_.load()
        print(encoder.name)
        X_train = encoder.encode(X_train_)
        # X_test = encoder.encode(X_test_)
        print('data encoded')
        cls_a = [MultinomialNB(), LogisticRegression(max_iter=1000), SVC(), RandomForestClassifier(), KNeighborsClassifier()]
        for cls in cls_a:
            cross_eval_classifier(cls, X_train, y_train)
            print('-------------------------------')
 

def train_classifiers():

    X_train_, X_test_, y_train, y_test = get_train_test('basic_split')

    lab = SubsetLabeller().load()
    y_train, y_test = lab.label(y_train), lab.label(y_test)
    
    
    print('encoding')
    enc = NGramEncoder(2).load()
    X_train = enc.encode(X_train_)
    X_test = enc.encode(X_test_)
    
    print('fitting')
    clf = LogisticRegression(max_iter=1000, verbose=1)
    clf.fit(X_train,y_train)
    dump(clf, f'classifiers/lr.clf')
    eval_classifier(clf, X_test, y_test)
    
    print('fitting')
    clf = KNeighborsClassifier()
    clf.fit(X_train,y_train)
    dump(clf, f'classifiers/knn.clf')
    eval_classifier(clf, X_test, y_test)
    
    
    print('encoding')
    enc = NGramEncoder(2).load()
    X_train_ = enc.encode(X_train_)
    X_test_ = enc.encode(X_test_)
    
    print('fitting')
    clf = SVC(verbose=True)
    clf.fit(X_train_,y_train)
    dump(clf, f'classifiers/svm.clf')
    eval_classifier(clf, X_test_, y_test, True)
    
    print('fitting')
    clf = RandomForestClassifier(n_estimators=40, verbose=1 )
    clf.fit(X_train,y_train)
    dump(clf, f'classifiers/rf.clf')
    eval_classifier(clf, X_test, y_test)
    
    print('fitting')
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    dump(clf, f'classifiers/bayes.clf')
    eval_classifier(clf, X_test, y_test)
    
    
def evaluate_classifiers():
    _, X_test_, _, y_test = get_train_test('basic_split')

    lab = SubsetLabeller().load()
    y_test =lab.label(y_test)
    
    print('encoding')
    enc = NGramEncoder(2).load()
    print('number of 2grams:', len(enc.API_gram_set))
    X_test = enc.encode(X_test_)
    
    print('loading classifier')
    clf = load('classifiers/lr.clf')
    eval_classifier(clf, X_test, y_test, True)
    
    print('loading classifier')
    clf = load('classifiers/knn.clf')
    eval_classifier(clf, X_test, y_test, True)
    
    print('encoding')
    enc = NGramEncoder(3).load()
    print('number of 3grams:', len(enc.API_gram_set))
    X_test = enc.encode(X_test_)
    
    print('loading classifier')
    clf = load('classifiers/svm.clf')
    eval_classifier(clf, X_test, y_test, True)
    
    print('loading classifier')
    clf = load('classifiers/rf.clf')
    eval_classifier(clf, X_test, y_test, True)
    
    print('loading classifier')
    clf = load('classifiers/bayes.clf')
    eval_classifier(clf, X_test, y_test, True)


             