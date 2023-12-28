# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:28:28 2023

@author: HP
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_score, recall_score, auc,roc_curve
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

#TO DO : MULTINOMIAL LOGISTIC REGRESSION

file1 = pd.read_excel("File 1.xlsx")
file2 = pd.read_excel("File 2.xlsx")
file3 = pd.read_excel("File 3.xlsx")
file4 = pd.read_excel("File 4.xlsx")
file5 = pd.read_excel("File 5.xlsx")
file6 = pd.read_excel("File 6.xlsx")
file7 = pd.read_excel("File 7.xlsx")
file8 = pd.read_excel("File 8.xlsx")

#merging all the files
files = pd.concat([file1, file2,file3,file4,file5,file6,file7,file8])

#Creating a New DataFrame and remaning the variables
files_data = pd.DataFrame()
files_data["Urine level"] = files[["Urine level"]]
files_data["Paper level"] = files[["Paper level"]]
files_data["Feces level"] = files[["Feces Level"]]
files_data["B1P1"] = files[["Blue LED 1\nPhotodiode 1"]]
files_data["B1P2"] = files[["Blue LED 1\nPhotodiode 2"]]
files_data["G1P1"] = files[["Green LED 1\nPhotodiode 1"]]
files_data["G1P2"] = files[["Green LED 1\nPhotodiode 2"]]
files_data["R1P1"] = files[["Red LED 1\nPhotodiode 1"]]
files_data["R1P2"] = files[["Red LED 1\nPhotodiode 2"]]
files_data["B2P1"] = files[["Blue LED 2\nPhotodiode 1"]]
files_data["B2P2"] = files[["Blue LED 2\nPhotodiode 2"]]
files_data["G2P1"] = files[["Green LED 2\nPhotodiode 1"]]
files_data["G2P2"] = files[["Green LED 2\nPhotodiode 2"]]
files_data["R2P1"] = files[["Red LED 2\nPhotodiode 1"]]
files_data["R2P2"] = files[["Red LED 2\nPhotodiode 2"]]
files_data["Flush volume"] = files[["Flush volume"]]
files_data["No Discret"] = files[["number from 0 to 51"]]
files_data["Case of flush"] = files[["Case of flush"]]
files_data["PCB value"] = files[["PCB value"]]


plt.hist(files_data["Case of flush"])
# set min case of flush (everything below 4 is a 4)

#Correaltion heatmap 
results = files_data[["B1P1","B1P2","G1P1","G1P2","R1P1","R1P2","B2P1","B2P2","G2P1","G2P2","R2P1","R2P2","Case of flush"]]
sns.heatmap(results.corr(), annot=True)
plt.show()

#testing and dealing multicolinearity
X = files_data[["B1P1","B1P2","G1P1","G1P2","R1P1","R1P2","B2P1","B2P2","G2P1","G2P2","R2P1","R2P2"]]
y = files_data[['Case of flush']]
y = y.values.ravel()


#dealing with multicolinearity
vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)

#changing variables
#change X2 variables and do it again
Y1=files_data[["G2P1"]]
X1 = files_data[["R2P1"]]
model = sm.OLS(Y1, X1)
results1 = model.fit()
residuals1 = results1.resid
files_data.loc[:,"Res_G2P1"]=residuals1

X1 = files_data[["B1P1","B1P2","G1P1","G1P2","R1P1","R1P2","B2P1","B2P2","Res_G2P1","G2P2","R2P1","R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif["features"] = X1.columns
vif.round(1)

Y2=files_data[["R2P2"]]
X2 = files_data[["R2P1"]]
model = sm.OLS(Y2, X2)
results2 = model.fit()
residuals2 = results2.resid
files_data.loc[:,"Res_R2P2"]=residuals2

X2 = files_data[["B1P1","B1P2","G1P1","G1P2","R1P1","R1P2","B2P1","B2P2","Res_G2P1","G2P2","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif["features"] = X2.columns
vif.round(1)

Y3=files_data[["R1P1"]]
X3 = files_data[["R2P1"]]
model = sm.OLS(Y3, X3)
results3 = model.fit()
residuals3 = results3.resid
files_data.loc[:,"Res_R1P1"]=residuals3

X3 = files_data[["B1P1","B1P2","G1P1","G1P2","Res_R1P1","R1P2","B2P1","B2P2","Res_G2P1","G2P2","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]
vif["features"] = X3.columns
vif.round(1)

Y4=files_data[["G2P2"]]
X4 = files_data[["R2P1"]]
model = sm.OLS(Y4, X4)
results4 = model.fit()
residuals4 = results4.resid
files_data.loc[:,"Res_G2P2"]=residuals4

X4 = files_data[["B1P1","B1P2","G1P1","G1P2","Res_R1P1","R1P2","B2P1","B2P2","Res_G2P1","Res_G2P2","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X4.values, i) for i in range(X4.shape[1])]
vif["features"] = X4.columns
vif.round(1)


Y5=files_data[["G1P1"]]
X5 = files_data[["R2P1"]]
model = sm.OLS(Y5, X5)
results5 = model.fit()
residuals5 = results5.resid
files_data.loc[:,"Res_G1P1"]=residuals5

X5 = files_data[["B1P1","B1P2","Res_G1P1","G1P2","Res_R1P1","R1P2","B2P1","B2P2","Res_G2P1","Res_G2P2","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X5.values, i) for i in range(X5.shape[1])]
vif["features"] = X5.columns
vif.round(1)


Y6=files_data[["G1P2"]]
X6 = files_data[["R1P2"]]
model = sm.OLS(Y6, X6)
results6 = model.fit()
residuals6 = results6.resid
files_data.loc[:,"Res_G1P2"]=residuals6

X6 = files_data[["B1P1","B1P2","Res_G1P1","Res_G1P2","Res_R1P1","R1P2","B2P1","B2P2","Res_G2P1","Res_G2P2","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X6.values, i) for i in range(X6.shape[1])]
vif["features"] = X6.columns
vif.round(1)

Y7=files_data[["B2P2"]]
X7 = files_data[["B2P1"]]
model = sm.OLS(Y7, X7)
results7 = model.fit()
residuals7 = results7.resid
files_data.loc[:,"Res_B2P2"]=residuals7

X7 = files_data[["B1P1","B1P2","Res_G1P1","Res_G1P2","Res_R1P1","R1P2","B2P1","Res_B2P2","Res_G2P1","Res_G2P2","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X7.values, i) for i in range(X7.shape[1])]
vif["features"] = X7.columns
vif.round(1)

Y8=files_data[["B2P1"]]
X8 = files_data[["B1P1"]]
model = sm.OLS(Y8, X8)
results8 = model.fit()
residuals8 = results8.resid
files_data.loc[:,"Res_B2P1"]=residuals8

X8 = files_data[["B1P1","B1P2","Res_G1P1","Res_G1P2","Res_R1P1","R1P2","Res_B2P1","Res_B2P2","Res_G2P1","Res_G2P2","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X8.values, i) for i in range(X8.shape[1])]
vif["features"] = X8.columns
vif.round(1)


Y9=files_data[["Res_G1P1"]]
X9 = files_data[["Res_R1P1"]]
model = sm.OLS(Y9, X9)
results9 = model.fit()
residuals9 = results9.resid
files_data.loc[:,"Res_G1P1b"]=residuals9

X9 = files_data[["B1P1","B1P2","Res_G1P1b","Res_G1P2","Res_R1P1","R1P2","Res_B2P1","Res_B2P2","Res_G2P1","Res_G2P2","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X9.values, i) for i in range(X9.shape[1])]
vif["features"] = X9.columns
vif.round(1)

Y10=files_data[["Res_G2P2"]]
X10= files_data[["Res_R2P2"]]
model = sm.OLS(Y10, X10)
results10 = model.fit()
residuals10 = results10.resid
files_data.loc[:,"Res_G2P2b"]=residuals10

X10 = files_data[["B1P1","B1P2","Res_G1P1b","Res_G1P2","Res_R1P1","R1P2","Res_B2P1","Res_B2P2","Res_G2P1","Res_G2P2b","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X10.values, i) for i in range(X10.shape[1])]
vif["features"] = X10.columns
vif.round(1)

Y11=files_data[["Res_G1P1b"]]
X11 = files_data[["Res_G2P1"]]
model = sm.OLS(Y11, X11)
results11 = model.fit()
residuals11 = results11.resid
files_data.loc[:,"Res_G1P1bi"]=residuals11

X11 = files_data[["B1P1","B1P2","Res_G1P1bi","Res_G1P2","Res_R1P1","R1P2","Res_B2P1","Res_B2P2","Res_G2P1","Res_G2P2b","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X11.values, i) for i in range(X11.shape[1])]
vif["features"] = X11.columns
vif.round(1)

Y12= files_data[["Res_G2P2b"]]
X12 = files_data[["Res_G2P1"]]
model = sm.OLS(Y12, X12)
results12 = model.fit()
residuals12 = results12.resid
files_data.loc[:,"Res_G2P2bi"]=residuals12

X12 = files_data[["B1P1","B1P2","Res_G1P1bi","Res_G1P2","Res_R1P1","R1P2","Res_B2P1","Res_B2P2","Res_G2P1","Res_G2P2bi","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X12.values, i) for i in range(X12.shape[1])]
vif["features"] = X12.columns
vif.round(1)

Y13=files_data[["B1P1"]]
X13 = files_data[["R2P1"]]
model = sm.OLS(Y13, X13)
results13 = model.fit()
residuals13 = results13.resid
files_data.loc[:,"Res_B1P1"]=residuals13

X13 = files_data[["Res_B1P1","B1P2","Res_G1P1bi","Res_G1P2","Res_R1P1","R1P2","Res_B2P1","Res_B2P2","Res_G2P1","Res_G2P2bi","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X13.values, i) for i in range(X13.shape[1])]
vif["features"] = X13.columns
vif.round(1)

Y14=files_data[["B1P2"]]
X14 = files_data[["R1P2"]]
model = sm.OLS(Y14, X14)
results14 = model.fit()
residuals14 = results14.resid
files_data.loc[:,"Res_B1P2"]=residuals14

X14 = files_data[["Res_B1P1","Res_B1P2","Res_G1P1bi","Res_G1P2","Res_R1P1","R1P2","Res_B2P1","Res_B2P2","Res_G2P1","Res_G2P2bi","R2P1","Res_R2P2"]]

vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X14.values, i) for i in range(X14.shape[1])]
vif["features"] = X14.columns
vif.round(1)

#test multinomial model
X = X14
y = files_data[['Case of flush']]
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 



clf = [
    LogisticRegression(multi_class='multinomial', class_weight='balanced' , solver='newton-cg', penalty='l2',),
    ]
clf_columns = []
clf_compare = pd.DataFrame(columns = clf_columns)

row_index = 0
for alg in clf:
        
    predicted = alg.fit(X_train, y_train).predict(X_test)
    clf_name = alg.__class__.__name__
    clf_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 5)
    clf_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 5)
    row_index+=1
    
clf_compare.sort_values(by = ['Test Accuracy'], ascending = False, inplace = True)    
clf_compare

#GridSearch
parameters = {'penalty': ['none', 'l2', 'l1', 'elasticnet'],
              'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
              'multi_class': ['multinomial'],
              'class_weight':['balanced'],
              'random_state':[100],
              'max_iter':[1000]}
clf2 = GridSearchCV(LogisticRegression(), parameters, cv=5)
logreg_cv=clf2.fit(X_train,y_train)
print("Tuned hpyerparameters (best parameters):", logreg_cv.best_params_)
print('Best Penalty:', clf2.best_estimator_.get_params()['penalty'])
print("accuracy :",clf2.best_score_)

#RandomizedSearch
parameters = {'C': [0.001,0.01,0.1,1,10,100,1000],
              'penalty': ['none', 'l2', 'l1', 'elasticnet'],
              'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
              'multi_class': ['multinomial'],
              'class_weight':['balanced'],
              'max_iter':[1000]}
clf3 = RandomizedSearchCV(LogisticRegression(), parameters, cv=5, n_jobs=-1)
logreg_cv=clf3.fit(X_train,y_train)
print("Tuned hpyerparameters (best parameters):", logreg_cv.best_params_)


#results = confusion_matrix(Y,result.PCB_value)
#print(results)
clf_f = LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty='l2', class_weight='balanced', C=1.0)
pred = clf_f.fit(X_train, y_train).predict(X_test)

#adding +2 for cleanliness:
pred = pred+2
print(pred)

#making sure no values are above 11
max_accepted_value = 11
pred = np.minimum(pred, max_accepted_value)
print(pred)

#adding more for 4th first class, where the error are more spread, based on correlation matrix
for i in range(len(pred)):
    if pred[i]<4:
        pred[i]=pred[i]+2
    else:
        pass

#Measuring the model performance
#new accuracy for cleanliness
ok = 0
i=0
for i  in range(len(pred)):
    if pred[i] >= y_test[i]: 
        ok+=1
    else:
        pass
acc = (ok/len(pred)) 
acc_2f = "{:.2f}".format(acc)
print(acc_2f)

#new accuracy for cleanliness of PCB_model
ok = 0
for index, row in files_data.iterrows():
    if row['PCB value'] >= row['Case of flush']: 
        ok+=1
    else:
        pass
acc_PCB = (ok/len(files_data['PCB value'])) 
print("{:.2f}".format(acc_PCB))

#check the accuracy is higher than PCB
#comparison with PCB value, show that it is cleaner
results = confusion_matrix(y_test,pred)
sns.heatmap(results, annot=True, cmap='Blues')


# errors below and above
lower_errors = np.sum(np.where(y_test > pred, 1, 0))
print("Lower Errors: ",lower_errors)
over_errors = np.sum(np.where(y_test < pred, 1, 0))
print("Over Errors: ",over_errors)