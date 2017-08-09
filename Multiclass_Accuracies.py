#!/usr/bin/env python
#Local -- Multiclass_Accuracies script 


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
from sklearn.metrics import accuracy_score,make_scorer,precision_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier


#Dictionaries that hold parameters 
paramsKNN = {
    'n_neighbors':[2,5,10,20],

    #new        
   # 'n_neighbors':[10],
    'algorithm':['auto','ball_tree','kd_tree','brute'],
}
paramsDecisionTrees = {
    'criterion':['gini', 'entropy'],
    'max_depth':[5,10,20],
    'min_samples_split': [2,5,10,15],
    'min_samples_leaf':[5,10,15],
    'max_features':['sqrt','log2'],

}
paramsNB = {}
paramsSVM = {
    #'estimator__C':[1,2, 10, 100],
    'estimator__C':[1,2, 10],
    'estimator__kernel':['rbf', 'linear']
}
paramsGaussian = {}
paramsRandomForest = {

    'criterion':['gini', 'entropy'],#new
   # 'n_estimators': [10,50,100],#changed
   # 'max_depth':[3,5,10,100],
   # 'min_samples_split':[5,10,15],
   # 'min_samples_leaf':[5,10,15],
   # 'max_features':['sqrt', 'log2']
}

paramsNeuralNet = {
  'hidden_layer_sizes':[(140,), (210,)],
  'solver':['lbfgs', 'adam']
}
paramsAdaboost = {
  'estimator__learning_rate': [1],
  #'estimator__n_estimators': [50,100,500,1000],
  'estimator__base_estimator':[DecisionTreeClassifier(), RandomForestClassifier()]
}
paramsExtraTrees = {
    #'n_estimators':[10,50,70,90,100,500,1000,2000],#changed
    #'max_depth':[4,10,100],
    #'min_samples_split':[2,5,10,15],
    #'min_samples_leaf': [2,5,10,15],
    'max_features':['sqrt', 'log2'],
}


# In[14]:

# Dictionary of algorithms (with their parameters)

algs = collections.OrderedDict()
#algs['KNN'] = paramsKNN
#algs['Decision Tree'] = paramsDecisionTrees
#algs['Naive Bayes'] = paramsNB
#algs['SVM'] = paramsSVM
#algs['Gaussian Process'] = paramsGaussian
algs['Random Forest'] = paramsRandomForest
#algs['Neural Net'] = paramsNeuralNet
#algs['AdaBoost'] = paramsAdaboost
#algs['Extra Trees Classifier'] = paramsExtraTrees


df = pd.DataFrame()
#df['classifier name'] = ['KNN', 'Decision Tree', 'Naive Bayes', 'SVM', 'Gaussian Process', 'Random Forest', 'Neural Net', 'AdaBoost', 'Extra Trees Classifier']


df['classifier name'] = ['KNN']
print("Testing.........................................................M_A_0")

# In[2]:
# Sets up models with specified algorithms and parameters to run Multiclass classification
def gridSearch(dataset_name, X, y, num_iterations):
    print("Testing.........................................................M_A_1")
    models = collections.OrderedDict()
    for i in range(1, num_iterations):
        name = dataset_name + str(i)
       # models['KNN'] = KNeighborsClassifier()
       # models['Decision Tree'] = DecisionTreeClassifier(random_state=1)
       # models['Naive Bayes'] = OneVsRestClassifier(GaussianNB())
       # models['SVM'] = OneVsRestClassifier(SVC(random_state=1))
       # models['Gaussian Process'] = OneVsRestClassifier(GaussianProcessClassifier(random_state=1))
        models['Random Forest'] = RandomForestClassifier(random_state=1)
       # models['Neural Net'] = MLPClassifier(random_state=1)
       # models['AdaBoost'] = OneVsRestClassifier(AdaBoostClassifier(random_state=1))
       # models['Extra Trees Classifier'] = ExtraTreesClassifier(random_state=1)
        print("Testing.........................................................M_A_2")
        run_dataset(name, X, y, models, algs)

    return df

#make_scorer
ftwo_scorer = make_scorer(precision_score,average=None)


# In[10]:
# Runs datasets and classifies each sample as obese, lean, overweight
def run_dataset(dataset_name, X, y, models, algs): 
    print("Testing.........................................................M_A")
    iter_range = range(1,6)
    average_accuracy = 0.0
    accuracy_list = []
    print(dataset_name)
    for (name, model), (name, alg) in zip(models.items(),algs.items()):
        print(model)
        #splits data 50/50
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
        y = label_binarize(y, classes=[1,2,3])
        clf = GridSearchCV(model, alg, cv=10)

        #clf = GridSearchCV(model, alg, cv=10,scoring=ftwo_scorer)

        clf.fit(X_train, y_train)
        # print( best accuracy and associated params
        print(clf.best_params_)
        print('\n')
        y_pred_class = clf.predict(X_test)
        print("CLASSIFICATION SCORE")
#       print(metrics.precision_score(y_test, y_pred_class,average = None))

        print(metrics.accuracy_score(y_test, y_pred_class))
        #print out the number of correct prediction
        print(metrics.accuracy_score(y_test, y_pred_class,normalize=False))
        print(y_test.size)
       
        target_names = ['Normal','Overweight','Obese']
        print(alg)
        print(classification_report(y_test,y_pred_class,target_names=target_names)) 
        print(model)
        #print(y_pred_class)
        pd.options.display.max_rows = 999#show the whole data for visualization
        #print(y_test)
	#y_test_1 = y_test.append(y_pred_class)
        y_test_tran=np.array(y_test)
        y_test_pred_combined = [y_test_tran,y_pred_class]
        y_test_pred_combined = np.array(y_test_pred_combined)
        f1=open('./testfile2.txt', 'w+')
        f1.write(str(y_test.shape))
        f1.write(str(y_test.size))
        f1.write(str(y_test))
        f1.write(str(y_test_pred_combined))
        f1.write(str(y_test_pred_combined.__len__()))
        f1.write(str(y_test_pred_combined[0].__len__()))
        f1.write(str(y_test_pred_combined.shape))
        f1.close()
	
	#number of testing data
        array_length=y_test_pred_combined[0].__len__()
	#sum of the adjusted index
        sum_of_err=0
	#loop for getting the adjusted accuracy for different false
	#classification
        for i in range(0,array_length):
                test_element = y_test_pred_combined[0][i]	
                print(y_test_pred_combined[0][i])
                pred_element = y_test_pred_combined[1][i]
                print(y_test_pred_combined[1][i])
                pred_diff = test_element - pred_element
                print(pred_diff)
                pred_diff = abs(pred_diff)	
                sum_of_err = sum_of_err + pred_diff
        print("THE RESULT IS HERE")
        print(sum_of_err)	
	 #print (labels[y_pred_class])
	
	 # print std. deviation
        accuracy_list.append(clf.best_score_)
        #clf = clf.best_estimator_ 
    se = pd.Series(accuracy_list)
    df[dataset_name] = se.values

