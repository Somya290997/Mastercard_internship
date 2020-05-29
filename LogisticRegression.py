import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
dataset = pd.read_csv('C:\\Users\\somya\\Anaconda3\\examples\\micro_delinquency\\data1.csv')
dataset.fillna(-99999, inplace=True)

'''pd.set_option('display.width',1000)
pd.set_option('display.max_column',40)
pd.set_option('precision',2)

import matplotlib.pyplot as plt
import seaborn as sbn

import warnings
warnings.filterwarnings("ignore")

train=pd.read_csv('/home/hp/data1.csv')
train.fillna(-99999, inplace=True) 
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
output_target=train["label"]
print(input_predictors)
x_train, x_val, y_train, y_val=train_test_split(input_predictors, output_target, test_size = 0.30, random_state = 6)
'''
# input
x = dataset.iloc[:,[0,2,3,4,7,9,10,11,12,13,14,15,16,17,18,19,23,24,25,27,28,29,31,32]].values

# output
y = dataset.iloc[:, 36].values
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 0)
test_size = 0.30
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

print (xtrain[0:10, :])
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)

print ("Confusion Matrix : \n", cm)
from sklearn.metrics import accuracy_score
accuracy = round(accuracy_score(ytest, y_pred) * 100, 2)
print( "MODEL: Accuracy of LogisticRegression : ",accuracy)
print("\n")
from sklearn.metrics import classification_report


report = classification_report(ytest, y_pred)
print(report)

# Evaluate metrics
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    (rmse, mae, r2) = eval_metrics(ytest, y_pred)
    # Print out ElasticNet model metrics
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_param("validation_ratio", str(test_size))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(classifier, "model")
    conf_matrix = confusion_matrix(ytest,y_pred)
    true_positive = conf_matrix[0][0]
    true_negative = conf_matrix[1][1]
    false_positive = conf_matrix[0][1]
    false_negative = conf_matrix[1][0]

    mlflow.log_metric("true_positive", true_positive)
    mlflow.log_metric("true_negative", true_negative)
    mlflow.log_metric("false_positive", false_positive)
    mlflow.log_metric("false_negative", false_negative)


    #log_artifact(<your_plot>, "confusion_matrix")