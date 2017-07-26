
# 5/10/17 - Multiclass_Accuracies script
# coding: utf-8

#In[1]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn import metrics
import collections
# In[5]:

#for make_scorer
from sklearn.metrics import accuracy_score,make_scorer

#Dictionaries that hold parameters 
paramsKNN = {
    'n_neighbors':[2,5,10,50]
}
paramsDecisionTrees = {
    'criterion':['gini', 'entropy'],
    'max_depth':[5,10,20],
    'min_samples_split': [2,5,10,15],
    'min_samples_leaf':[5,10,15],
    'max_features':['sqrt'],   
}
paramsNB = {}
paramsSVM = {
    'estimator__C':[1,2, 10, 100],
    'estimator__kernel':['rbf', 'linear']
}
paramsGaussian = {}
paramsRandomForest = {
    'n_estimators': [100, 500, 1000,2000],
    'max_depth':[3,5,10,100],
    'min_samples_split':[5,10,15],
    'min_samples_leaf':[5,10,15],
    'max_features':['sqrt', 'log2']
}

paramsNeuralNet = {
  'hidden_layer_sizes':[(140,), (210,)],
  'solver':['lbfgs', 'adam']
}
paramsAdaboost = {
  'estimator__learning_rate': [1],
  'estimator__n_estimators': [50,100,500,1000],
  'estimator__base_estimator':[DecisionTreeClassifier(), RandomForestClassifier()]
}
paramsExtraTrees = {
    'n_estimators':[100,500,1000,2000],
    'max_depth':[4,10,100],
    'min_samples_split':[2,5,10,15],
    'min_samples_leaf': [2,5,10,15],
    'max_features':['sqrt', 'log2'], 
}


# In[14]:

# Dictionary of algorithms (with their parameters)

algs = collections.OrderedDict()
algs['KNN'] = paramsKNN
algs['Decision Tree'] = paramsDecisionTrees
algs['Naive Bayes'] = paramsNB
algs['SVM'] = paramsSVM
algs['Gaussian Process'] = paramsGaussian
algs['Random Forest'] = paramsRandomForest
algs['Neural Net'] = paramsNeuralNet
algs['AdaBoost'] = paramsAdaboost
algs['Extra Trees Classifier'] = paramsExtraTrees


df = pd.DataFrame()
df['classifier name'] = ['KNN', 'Decision Tree', 'Naive Bayes', 'SVM', 'Gaussian Process', 'Random Forest', 'Neural Net', 'AdaBoost', 'Extra Trees Classifier']

# In[2]:
# Sets up models with specified algorithms and parameters to run Multiclass classification
def gridSearch(dataset_name, X, y, num_iterations):
    models = collections.OrderedDict() 
    for i in range(1, num_iterations):
        name = dataset_name + str(i)
        models['KNN'] = KNeighborsClassifier()
        models['Decision Tree'] = DecisionTreeClassifier(random_state=1)
        models['Naive Bayes'] = OneVsRestClassifier(GaussianNB())
        models['SVM'] = OneVsRestClassifier(SVC(random_state=1)) 
        models['Gaussian Process'] = OneVsRestClassifier(GaussianProcessClassifier(random_state=1))
        models['Random Forest'] = RandomForestClassifier(random_state=1)
        models['Neural Net'] = MLPClassifier(random_state=1)
        models['AdaBoost'] = OneVsRestClassifier(AdaBoostClassifier(random_state=1))
        models['Extra Trees Classifier'] = ExtraTreesClassifier(random_state=1)
        run_dataset(name, X, y, models, algs) 
                      
    return df

#make_scorer
ftwo_scorer = make_scorer(accuracy_score)

# In[10]:
# Runs datasets and classifies each sample as obese, lean, overweight
def run_dataset(dataset_name, X, y, models, algs):
    iter_range = range(1,6)
    average_accuracy = 0.0
    accuracy_list = []
    print(dataset_name)
    for (name, model), (name, alg) in zip(models.items(),algs.items()):
        print(model)
	#splits data 50/50
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
	y = label_binarize(y, classes=[1,2,3])
        #clf = GridSearchCV(model, alg, cv=10)
        clf = GridSearchCV(model, alg, cv=10,scoring=ftwo_scorer)
        clf.fit(X_train, y_train)
        # print( best accuracy and associated params
        print(clf.best_params_)
        print('\n')
	y_pred_class = clf.predict(X_test)
	print("CLASSIFICATION SCORE")
	print(metrics.accuracy_score(y_test, y_pred_class))
	# print std. deviation
        accuracy_list.append(clf.best_score_)	
        #clf = clf.best_estimator_ 
    se = pd.Series(accuracy_list)
    df[dataset_name] = se.values


