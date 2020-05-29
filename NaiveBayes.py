from sklearn import metrics
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.naive_bayes import GaussianNB
dataset = pd.read_csv('C:\\Users\\somya\\Anaconda3\\examples\\micro_delinquency\\data1.csv')
dataset.fillna(-99999, inplace=True)
x = dataset.iloc[:,[0,2,3,4,7,9,10,11,12,13,14,15,16,17,18,19,23,24,25,27,28,29,31,32]].values
y = dataset.iloc[:, 36].values
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 0)
test_size = 0.3
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)
print (xtrain[0:10, :])
classifier = GaussianNB()
classifier.fit(xtrain, ytrain)
x_pred = ytest
y_pred = classifier.predict(xtest)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)
print(metrics.classification_report(x_pred, y_pred))
print(metrics.confusion_matrix(x_pred, y_pred))
from sklearn.metrics import accuracy_score
accuracy = round(accuracy_score(ytest, y_pred) * 100, 2)
print( "MODEL: Accuracy of NaiveBayes : ",accuracy)
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