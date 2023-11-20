from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics as mt
import numpy  as np
from joblib import dump, load
from labelling import Labeller, KMeansLabeller, EnumerativeLabeller, SubsetLabeller
from encoder import FrequencyEncoder, NGramEncoder, TF_IDFEncoder, BagOfWordsEncoder
from split_data import get_train_test
import matplotlib.pyplot as plt
from split_data import basic_split

# tunes parameters of a classifier
# parameter to be tuned in 'param_grid'
def grid_search(classifier, X, y, param_grid, cv=5, scoring='f1_macro'):
    search = GridSearchCV(classifier, param_grid=param_grid, cv=cv, scoring=scoring, verbose=2)
    search.fit(X,y)
    return search.best_estimator_

# tunes and saves the tuned classifier
def tune_classifier(classifier, X, y, param_grid, name):
    classifier = grid_search(classifier, X, y, param_grid)
    eval_classifier(classifier, X, y)
    print(f'found best params for classifier {name}: {classifier.get_params()}')
    dump(classifier, f'classifiers/{name}.clf')
    return classifier

def load_classifier(name):
    return load(f'classifiers/{name}.clf')

# tunes all chosen classifiers using grid search
# encoder and labeller method chosen by the user
def tune_all(encoder, labeller, name_flag=''):
    X_train, _, y_train, _ = get_train_test(f'basic_split{name_flag}')
    print('encoding...')
    X_train = encoder.encode(X_train)
    print('labelling...')
    y_train = labeller.label(y_train)
    print('---------Bayes-----------')
    tune_classifier(MultinomialNB(), X_train, y_train, 
                    param_grid={'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, 
                    name=f'bayes-{encoder.name}-{labeller.name}{name_flag}')
    
    print('---------LogRes-----------')
    tune_classifier(LogisticRegression(max_iter=1000, solver='liblinear'), X_train, y_train, 
                    param_grid={'C':            [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                # 'penalty':      ['l1', 'l2', 'elasticnet'],
                                'dual':         [False, True],
                                'multi_class':  ['auto', 'ovr', 'multinomial']}, 
                    name=f'LR-{encoder.name}-{labeller.name}{name_flag}')

    print('---------SVM-----------')
    tune_classifier(SVC(), X_train, y_train, 
                    param_grid={'C':      [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                'gamma':  ['scale', 'auto']}, 
                    name=f'SVM-{encoder.name}-{labeller.name}{name_flag}')
    
    print('---------RandForest-----------')
    tune_classifier(RandomForestClassifier(), X_train, y_train, 
                    param_grid={'criterion':          ["gini", "entropy", "log_loss"], 
                                'max_depth':          [2, 4, 8, 16, 32, 64],
                                'min_samples_split':  [2, 4, 8, 16, 32, 64]}, 
                    name=f'RF-{encoder.name}-{labeller.name}{name_flag}')
    
    print('---------KNN-----------')
    tune_classifier(KNeighborsClassifier(), X_train, y_train, 
                    param_grid={'n_neighbors': [2, 4, 8, 16, 32],
                                'weights':     ['uniform', 'distance'],
                                'p':           [1, 2]}, 
                    name=f'KNN-{encoder.name}-{labeller.name}{name_flag}')


# evaluates the classifier
def eval_classifier(classifier, X_test, y_test, conf_mat=False):
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




if __name__ == '__main__':
    labeler = SubsetLabeller().load()
    encoders = [
                BagOfWordsEncoder(name_flag='-small').load(), 
                FrequencyEncoder(name_flag='-small').load(), 
                NGramEncoder(2, name_flag='-small').load(),
                NGramEncoder(3, name_flag='-small').load(),
                NGramEncoder(4, name_flag='-small').load(),
                NGramEncoder(5, name_flag='-small').load(),
                TF_IDFEncoder(name_flag='-small').load()
                ]

    for encoder in encoders:
    # encoder = TF_IDFEncoder(name_flag='-small').load() # make('basic_split-small').save()
        tune_all(encoder, labeler, name_flag='-small')
    exit(0)

    X_train_, X_test_, y_train, y_test = get_train_test('basic_split-small')

    print('labelling...')
    y_train = labeler.label(y_train)
    y_test = labeler.label(y_test)

    encoders = [
                # BagOfWordsEncoder(name_flag='-small').load(), 
                # FrequencyEncoder(name_flag='-small').load(), 
                # NGramEncoder(2, name_flag='-small').load(),
                # NGramEncoder(3, name_flag='-small').load(),
                # NGramEncoder(4, name_flag='-small').load(),
                # NGramEncoder(5, name_flag='-small').load(),
                TF_IDFEncoder(name_flag='-small').load()]

    for encoder in encoders:
        print('encoding...')
        X_train = encoder.encode(X_train_)
        X_test = encoder.encode(X_test_)
        
        print('fitting classifier...')
        cls = load_classifier('LR-TF-IDF-small-Subset80-small') # RandomForestClassifier() #  KNeighborsClassifier() # LogisticRegression() # SVC() # MultinomialNB() # 
        print(cls.get_params())
        cls = cls.fit(X_train, y_train)
        print(encoder.name)
        eval_classifier(cls, X_test, y_test, conf_mat=True)



    # disp = mt.ConfusionMatrixDisplay(conf_mat)
    # disp.plot()
    # plt.show()
    # [0 2 2 4 5 2 1 3 4 7]
    # evaluate(load_classifier('KNN-Frequency-KMeans32'), X_enc, y_lab)
    # evaluate(load_classifier('LR-Frequency-KMeans32').fit(X_train,y_train), X_enc, y_lab)
    # evaluate(load_classifier('RF-Frequency-KMeans32'), X_enc, y_lab)
    # evaluate(load_classifier('SVM-Frequency-KMeans32'), X_enc, y_lab)
