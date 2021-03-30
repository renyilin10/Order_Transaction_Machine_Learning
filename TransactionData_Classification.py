### Analyze data ###

## Classification ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
print(sklearn.__version__)

dataset = pd.read_csv('output3.csv',keep_default_na=False)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

## split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


## Random Forest ##

## Base model
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred_RF = classifier_RF.predict(X_test)
cm_RF = confusion_matrix(y_test, y_pred_RF)
print(cm_RF)
accuracy_score(y_test, y_pred_RF) ## base model accuracy

# get importance
importance_RF = classifier_RF.feature_importances_

## create a dataframe of feature names and their importance score
df_rf = pd.DataFrame({'Feature_Names': dataset.iloc[:, 1:-1].columns, 'Importance':importance_RF})

df_rf1 = df_rf.sort_values(by ='Importance', ascending = False)

df_rf1

## Hyperparameter turning
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num =12)] 

max_features = ['auto', 'sqrt'] 

max_depth = [int(x) for x in np.linspace(2, 30, num = 10)] 
max_depth.append(None)

min_samples_split = [2,5,10] 

min_samples_leaf = [1,2,4] 

bootstrap = [True, False] 

## create the param grid
parameter_grid = {'n_estimators': n_estimators, 'max_features' : max_features, 'max_depth': max_depth,
                 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap':bootstrap}

print(parameter_grid)

## Used GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf_Model = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = rf_Model, 
                           param_grid = parameter_grid,
                           scoring = 'accuracy',
                           cv = 10,
                           verbose = 2,
                           random_state = 0,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_  ## Mean cross-validated score of the best_estimator
best_parameters = grid_search.best_params_  ## Parameter setting that gave the best results on the hold out data
print("Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters", best_parameters)


print(f'Train Accuracy - : {(grid_search.score(X_train, y_train))*100:.2f}%')
print(f'Test Accuracy - : {(grid_search.score(X_test, y_test))*100:.2f}%')


## Used RandomizedSearch CV (investigate in fewer combination of params to speed up)
from sklearn.model_selection import RandomizedSearchCV
Rangrid_search = RandomizedSearchCV(estimator = rf_Model, 
                                    param_distributions = parameter_grid,
                                    n_iter = 100,
                                    scoring = 'accuracy',
                                    cv = 10,
                                    verbose = 2,
                                    random_state = 0,
                                    n_jobs = -1)
Rangrid_search.fit(X_train, y_train)
best_accuracy2 = Rangrid_search.best_score_ 
best_parameters2 = Rangrid_search.best_params_  
print("Accuracy: {:.2f} %".format(best_accuracy2*100))
print("Best Parameters", best_parameters2)

print(f'Train Accuracy - : {(Rangrid_search.score(X_train, y_train))*100:.2f}%')
print(f'Test Accuracy - : {(Rangrid_search.score(X_test, y_test))*100:.2f}%')

## Use the best params from RandomizedSearch CV
classifier_RF_Best = RandomForestClassifier(n_estimators = 59, min_samples_split = 5, min_samples_leaf = 1, max_features = 'auto', max_depth = 11, bootstrap = True) 

classifier_RF_Best.fit(X_train, y_train)

y_pred_RF_Best = classifier_RF_Best.predict(X_test)
cm_RF_Best = confusion_matrix(y_test, y_pred_RF_Best)
print(cm_RF_Best)
accuracy_score(y_test, y_pred_RF_Best) ## best model accuracy from Rangrid_search


## Get the feature importance from the best model
importance_RF_Rangrid = classifier_RF_Best.feature_importances_

## create a dataframe of feature names and their importance score
df_rf_Rangrid = pd.DataFrame({'Feature_Names': dataset.iloc[:, 1:-1].columns, 'Importance':importance_RF_Rangrid})

df_rf1_Rangrid = df_rf_Rangrid.sort_values(by ='Importance', ascending = False)

df_rf1_Rangrid
                        

## XGBoost ## 
from xgboost import XGBClassifier
classifier_XGB = XGBClassifier()
classifier_XGB.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred_XGB = classifier_XGB.predict(X_test)
cm_XGB = confusion_matrix(y_test, y_pred_XGB)
print(cm_XGB)
accuracy_score(y_test, y_pred_XGB)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_XGB, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


## CatBoost ##
!pip install catboost

from catboost import CatBoostClassifier
classifier_CB = CatBoostClassifier()
classifier_CB.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred_CB = classifier_CB.predict(X_test)
cm_CB = confusion_matrix(y_test, y_pred_CB)
print(cm_CB)
accuracy_score(y_test, y_pred_CB)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier_CB, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


                        
                        
                        



