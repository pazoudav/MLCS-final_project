from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy  as np
from joblib import dump, load
from labelling import Labeller, KMeansLabeller
from split_data import get_train_test
from encoder import FrequencyEncoder


def grid_search(classifier, X, y, param_grid, cv=5, scoring='f1_macro'):
    search = GridSearchCV(classifier, param_grid=param_grid, cv=cv, scoring=scoring, verbose=2)
    search.fit(X,y)
    return search.best_estimator_

def tune_classifier(classifier, X, y, param_grid, name):
    classifier = grid_search(classifier, X, y, param_grid)
    dump(classifier, f'classifiers/{name}.clf')
    return classifier

def load_classifier(name):
    return load(f'classifiers/{name}.clf')

def tune_all(encoder, labeller):
    X_train, _, y_train, _ = get_train_test('basic_split')
    print('encoding')
    X_train = encoder.encode(X_train)
    print('labelling')
    y_train = labeller.label(y_train)
    
    # print('---------Bayes-----------')
    # tune_classifier(MultinomialNB(), X_train, y_train, 
    #                 param_grid={'alpha': [1]}, 
    #                 name=f'bayes-{encoder.name}-{labeller.name}')
    # 
    # print('---------LogRes-----------')
    # tune_classifier(LogisticRegression(max_iter=1000), X_train, y_train, 
    #                 param_grid={'C':      [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
    #                             'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}, 
    #                 name=f'LR-{encoder.name}-{labeller.name}')
    
    print('---------SVM-----------')
    tune_classifier(SVC(), X_train, y_train, 
                    param_grid={'C':      [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
                                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                                'gamma':  ['scale', 'auto']}, 
                    name=f'SVM-{encoder.name}-{labeller.name}')
    
    print('---------RandForest-----------')
    tune_classifier(RandomForestClassifier(), X_train, y_train, 
                    param_grid={'criterion':          ["gini", "entropy", "log_loss"], 
                                'max_depth':          [2, 4, 8, 16, 32, 64],
                                'min_samples_split':  [2, 4, 8, 16, 32, 64]}, 
                    name=f'RF-{encoder.name}-{labeller.name}')
    
    print('---------KNN-----------')
    tune_classifier(KNeighborsClassifier(), X_train, y_train, 
                    param_grid={'n_neighbors': [2, 4, 8, 16, 32],
                                'weights':     ['uniform', 'distance'],
                                'p':           [1, 2]}, 
                    name=f'KNN-{encoder.name}-{labeller.name}')


# tune_all(FrequencyEncoder().load(), KMeansLabeller(32).load())
