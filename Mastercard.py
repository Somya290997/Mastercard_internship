import numpy as np
import pandas as pd
import os
import sys

pd.set_option('display.width',1000)
pd.set_option('display.max_column',40)
pd.set_option('precision',2)

import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
train=pd.read_csv('C:\\Users\\somya\\Anaconda3\\examples\\micro_delinquency\\Micro_delinquency.csv')
print(train.describe()[:])

print("\n")
print(train.describe(include="all"))

print("\n\n", train.columns)

print(train.head())
print(train.sample(5))

print("Data types for each feature : \n", train.dtypes)

print(pd.isnull(train).sum()) #checks for unusuable data

cols=train.columns
train[cols]=train[cols].apply(pd.to_numeric,errors='coerce')
print(train)

train['aon']=train['aon'].fillna(0)
print(train)
train["aon"]=train["aon"].astype(int)
print(train)

train['daily_decr30']=train['daily_decr30'].fillna(0)
train['daily_decr90']=train['daily_decr90'].fillna(0)
train['rental30']=train['rental30'].fillna(0)
train['rental90']=train['rental90'].fillna(0)

indexNames = train[train['aon'] < 0].index
train.drop(indexNames, inplace=True)

'''sbn.barplot(x="label", y="aon", data=train)
plt.show()

sbn.barplot(x="label", y="daily_decr30", data=train)
plt.show()
sbn.barplot(x="label", y="daily_decr90", data=train)
plt.show()
sbn.barplot(x="label", y="rental30", data=train)
plt.show()

sbn.barplot(x="label", y="rental90", data=train)
plt.show()
sbn.barplot(x="label", y="last_rech_date_ma", data=train)
plt.show()
sbn.barplot(x="label", y="last_rech_date_da", data=train)
plt.show()
sbn.barplot(x="label", y="last_rech_amt_ma", data=train)
plt.show()
sbn.barplot(x="label", y="cnt_ma_rech30", data=train)
plt.show()
sbn.barplot(x="label", y="fr_ma_rech30", data=train)
plt.show()
sbn.barplot(x="label", y="sumamnt_ma_rech30", data=train)
plt.show()
sbn.barplot(x="label",y="medianamnt_ma_rech30",data=train)
plt.show()
sbn.barplot(x="label",y="medianamnt_ma_rech90",data=train)
plt.show()
sbn.barplot(x="label", y="medianmarechprebal90", data=train)
plt.show()
sbn.barplot(x="label", y="cnt_ma_rech90", data=train)
plt.show()
sbn.barplot(x="label", y="fr_ma_rech90", data=train)
plt.show()
sbn.barplot(x="label", y="sumamnt_ma_rech90", data=train)
plt.show()
sbn.barplot(x="label", y="medianmarechprebal30", data=train)
plt.show()
sbn.barplot(x="label", y="cnt_da_rech30", data=train)
plt.show()
sbn.barplot(x="label", y="fr_da_rech30", data=train)
plt.show()
sbn.barplot(x="label", y="cnt_da_rech90", data=train)
plt.show()
sbn.barplot(x="label", y="fr_da_rech90", data=train)
plt.show()
sbn.barplot(x="label", y="cnt_loans30", data=train)
plt.show()
sbn.barplot(x="label", y="amnt_loans30", data=train)
plt.show()
sbn.barplot(x="label",y="maxamnt_loans30",data=train)
plt.show()
sbn.barplot(x="label",y="medianamnt_loans30",data=train)
plt.show()
sbn.barplot(x="label",y="cnt_loans90",data=train )
plt.show()
sbn.barplot(x="label", y="amnt_loans90", data=train)
plt.show()
sbn.barplot(x="label",y="medianamnt_loans90",data=train)
plt.show()
sbn.barplot(x="label",y="maxamnt_loans90",data=train)
plt.show()
sbn.barplot(x="label",y="medianamnt_loans90",data=train)
plt.show()
sbn.barplot(x="label",y="payback30",data=train )
plt.show()
sbn.barplot(x="label",y="payback90",data=train )
plt.show()'''
correlations = train.corr()
fig = plt.figure()
subFig = fig.add_subplot(111)

cax = subFig.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
hnames = train.columns
ticks = np.arange(0,37)   # It will generate values from 0....8
subFig.set_xticks(ticks)
subFig.set_yticks(ticks)
subFig.set_xticklabels(hnames)
subFig.set_yticklabels(hnames)
plt.show()

indexNames = train[train['maxamnt_loans30'] > 12].index
train.drop(indexNames, inplace=True)

indexNames = train[train['aon'] == 0 ].index
train.drop(indexNames, inplace=True)

indexNames = train[train['daily_decr30'] == 0 ].index
train.drop(indexNames, inplace=True)

indexNames = train[train['daily_decr90'] == 0 ].index
train.drop(indexNames, inplace=True)

indexNames = train[train['rental30'] == 0 ].index
train.drop(indexNames, inplace=True)

indexNames = train[train['rental90'] == 0 ].index
train.drop(indexNames, inplace=True)

indexNames = train[train['maxamnt_loans90'] > 12].index
train.drop(indexNames, inplace=True)

indexNames = train[train['amnt_loans30'] > 1080].index
train.drop(indexNames, inplace=True)
indexNames = train[train['rental30'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['rental90'] < 0].index
train.drop(indexNames, inplace=True)

indexNames = train[train['payback30'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['payback90'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['amnt_loans90'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['cnt_loans90'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['medianmarechprebal90'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['medianamnt_ma_rech90'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['fr_ma_rech90'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['cnt_ma_rech90'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['medianmarechprebal30'] < 0].index
train.drop(indexNames, inplace=True)
indexNames = train[train['sumamnt_ma_rech30'] < 0].index
train.drop(indexNames, inplace=True)


train = train.drop(['msisdn'], axis = 1)
train = train.drop(['pcircle'], axis = 1)
train = train.drop(['pdate'], axis = 1)

print(train)
print(train.dtypes)


from sklearn.model_selection import train_test_split
input_predictors = train.drop(['aon','label','medianamnt_loans90','amnt_loans30','sumamnt_ma_rech90',
                               'medianamnt_ma_rech30','cnt_ma_rech30','last_rech_date_da','cnt_da_rech30',
                               'fr_da_rech30',  'cnt_da_rech90','fr_da_rech90','medianamnt_loans30'  ], axis=1)
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
output_target=train["label"]
print(input_predictors)
x_train, x_val, y_train, y_val=train_test_split(input_predictors, output_target, test_size = 0.20, random_state = 6)

from sklearn.metrics import accuracy_score

#1 Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-1: Accuracy of LogisticRegression : ", acc_logreg)
print("\n")

#2 Naive Bayes
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-2: Accuracy of GaussianNB : ", acc_gaussian)
print("\n")

#3 Support Vector Machine
'''from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-3: Accuracy of Support Vector Machines : ", acc_svc)
print("\n")'''

#4 Linear SVC
from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-3: Accuracy of LinearSVC : ",acc_linear_svc)
print("\n")

#5 Perceptron
from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-4: Accuracy of Perceptron : ",acc_perceptron)
print("\n")

#6 Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-5: Accuracy of DecisionTreeClassifier : ", acc_decisiontree)
print("\n")

#7 Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-6: Accuracy of RandomForestClassifier : ",acc_randomforest)
print("\n")

#8 KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-7: Accuracy of k-Nearest Neighbors : ",acc_knn)
print("\n")

#9Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-8: Accuracy of Stochastic Gradient Descent : ",acc_sgd)
print("\n")

#10 Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-9: Accuracy of GradientBoostingClassifier : ",acc_gbk)
print("\n")

#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# model = LogisticRegression()
# model.fit(x_train, y_train)
# predicted = model.predict(x_val)
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
report = classification_report(y_val, y_pred)
print(report)
from sklearn.metrics import confusion_matrix
with mlflow.start_run():
    # lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    # lr.fit(x_train,y_train)
    #
    # predicted_qualities = lr.predict(x_val)
    randomforest = RandomForestClassifier()
    randomforest.fit(x_train, y_train)
    y_pred = randomforest.predict(x_val)

    (rmse, mae, r2) = eval_metrics(y_val, y_pred)

    # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    conf_matrix = confusion_matrix(y_val, y_pred)
    true_positive = conf_matrix[0][0]
    true_negative = conf_matrix[1][1]
    false_positive = conf_matrix[0][1]
    false_negative = conf_matrix[1][0]

    mlflow.log_metric("true_positive", true_positive)
    mlflow.log_metric("true_negative", true_negative)
    mlflow.log_metric("false_positive", false_positive)
    mlflow.log_metric("false_negative", false_negative)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(randomforest, "model", registered_model_name="MicroDelinquencyModel")
    else:
        mlflow.sklearn.log_model(randomforest, "model")



