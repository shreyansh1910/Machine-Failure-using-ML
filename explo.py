import pandas as pd
import numpy as np
from sklearn import preprocessing as scale
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    f1_score,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score as AUC

# reading .csv file
dataset = pd.read_csv("C:/Users/Shreyansh/Desktop/exploratory/device_failure.csv")
print(dataset.size)
print(dataset.head(5))
print(dataset.dtypes)
dataset.isnull().sum()
data = dataset.dropna(axis=0)
data.columns.shape
print(data["failure"].value_counts())
print(data.describe())
data.sort_values(["device", "date"], inplace=True)
data["Days"] = data.groupby("device")["date"].rank(method="dense")
data["date"] = pd.to_datetime(data["date"])
data.columns.shape
print(data["failure"].value_counts())
data.groupby(["attribute7", "attribute8"])["attribute7"].count()
Failure = data[data.failure == 1]
plt.plot(Failure.Days, Failure.failure, "o")
print("length " + str(len(Failure.Days)))
plt.title("Distribution of Failure Rate as per age(days) from installation")
plt.ylabel("Failure")
plt.xlabel("Age(No. of days since installation)")
plt.show()
Corr = data[data.columns].corr()
sns.heatmap(Corr, annot=True)
fig = plt.figure(figsize=(14, 5))
plt.plot(data.attribute9, data.attribute3, "o")
plt.title("Relationship between attributes")
plt.xlabel("X (Attribute9)")
plt.ylabel("Y (Attribute3)")
plt.show()

print(data.describe())

df_nonfailure = data[data["failure"] == 0]
df_failure = data[data["failure"] == 1]
df_nonfailure_downsample = resample(
    df_nonfailure, replace=False, n_samples=400, random_state=23
)
df_resampled = pd.concat([df_nonfailure_downsample, df_failure])

data_Outcome = df_resampled["failure"]

df_resampled = df_resampled.drop(["failure", "date", "device", "attribute8"], axis=1)
standard_sc = scale.StandardScaler()
x_std = standard_sc.fit_transform(df_resampled)
data_scaled = pd.DataFrame(x_std)
print(data_scaled.head())

xtrain, xtest, ytrain, ytest = train_test_split(
    data_scaled, data_Outcome, test_size=0.50, random_state=19
)

print("ytest", len(ytest))


def Metrics(ytest, pred):
    print(
        "accuray:",
        accuracy_score(ytest, pred),
        ",recall score:",
        recall_score(ytest, pred),
        "\n ConfusionMatrix: \n",
        confusion_matrix(ytest, pred),
    )

    average_precision = average_precision_score(ytest, pred)
    print("average_precision_score: ", average_precision_score(ytest, pred))

    precision, recall, _ = precision_recall_curve(ytest, pred)


gnb = GaussianNB()
print(len(ytrain))
modelgnb = gnb.fit(xtrain, ytrain)
pred_gnb = modelgnb.predict(xtest)
Metrics(ytest, pred_gnb)
