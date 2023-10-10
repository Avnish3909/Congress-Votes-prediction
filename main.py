import pandas as pd
import numpy as np
import ydata_profiling
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import argsort
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix, f1_score, \
    precision_score, mean_squared_error, mean_absolute_error, r2_score
from ydata_profiling import ProfileReport

df=pd.read_csv(r"C:\Users\dell\Downloads\housevotes.csv")

print(df.head())

cols=["Handicapped_infant","Water_project_cost","Adoption_budget_resolution","Physician_free_freeze","El_salvador_aid","Religous_group_in_school","Anti_satelite_test_ban","Aid_to_nicagara","Mx_missile","Immigiration","Syanfuel_corporation_cutback","Education_spending","Superfund_right_to_sue","Crime","duty_free_export","Export_south_africa","Target"]
df.columns=cols
print(df.head().to_string())
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)
print(df.describe().to_string())
ydata_profiling.ProfileReport(df)
#ProfileReport(df).to_file(output_file="Housevote report.html")
df=df.fillna(df.mode().iloc[0])
print(df.isnull().sum())
def lb(x):
    lb=LabelEncoder()
    return lb.fit_transform(x)
df=df.apply(lb)
print(df.head().to_string())

x=df.drop("Target",axis=1)
y=df["Target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


bnb=BernoulliNB(alpha=0.1)
bnb.fit(x_train,y_train)
b_pred=bnb.predict(x_test)
print(b_pred)
print(accuracy_score(y_test,b_pred))
print(recall_score(y_test,b_pred))
print(precision_score(y_test,b_pred))
print(f1_score(y_test,b_pred))
print(confusion_matrix(y_test,b_pred))
dcm=confusion_matrix(y_test,b_pred)
dconf_matrix=pd.DataFrame(dcm,columns=["Predicted 0","Predicted 1"],index=["Actual 0","Actual 1"])
plt.figure(figsize=(10,10))
sns.heatmap(dconf_matrix,annot=True,fmt="d",cmap="Oranges")
plt.title("Confusion Matrix BernoulliNB")
plt.show()

TN=dcm[0][0]
FP=dcm[0][1]
FN=dcm[1][0]
TP=dcm[1][1]
sensitivity=TP/(TP+FN)
specificity=TN/(TN+FP)
print("BnB")
print("True Negative:",TN)
print("False Positive:",FP)
print("False Negative:",FN)
print("True Positive:",TP)
print("Sensitivity:",sensitivity)
print("Specificity:",specificity)

print("\t\tClassification Report for BnB")
print(classification_report(y_test,b_pred))
lgm=LogisticRegression(C=2)
lgm.fit(x_train,y_train)
l_pred=lgm.predict(x_test)
print(l_pred)
print(accuracy_score(y_test,l_pred))
print(recall_score(y_test,l_pred))
print(precision_score(y_test,l_pred))
print(f1_score(y_test,l_pred))
print(confusion_matrix(y_test,l_pred))
dcm=confusion_matrix(y_test,l_pred)
dconf_matrix=pd.DataFrame(dcm,columns=["Predicted 0","Predicted 1"],index=["Actual 0","Actual 1"])
plt.figure(figsize=(10,10))
sns.heatmap(dconf_matrix,annot=True,fmt="d",cmap="Reds")
plt.title("Confusion Matrix LogisticRegression")
plt.show()
TN=dcm[0][0]
FP=dcm[0][1]
FN=dcm[1][0]
TP=dcm[1][1]
sensitivity=TP/(TP+FN)
specificity=TN/(TN+FP)
print("LogisticRegression")
print("True Negative:",TN)
print("False Positive:",FP)
print("False Negative:",FN)
print("True Positive:",TP)
print("Sensitivity:",sensitivity)
print("Specificity:",specificity)
print("\t\tClassification Report for LogisticRegression")
print(classification_report(y_test,l_pred))


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
t_pred=dt.predict(x_test)
print(t_pred)
print(accuracy_score(y_test,t_pred))
print(recall_score(y_test,t_pred))
print(precision_score(y_test,t_pred))
print(f1_score(y_test,t_pred))
print(confusion_matrix(y_test,t_pred))

dcm=confusion_matrix(y_test,t_pred)
dconf_matrix=pd.DataFrame(dcm,columns=["Predicted 0","Predicted 1"],index=["Actual 0","Actual 1"])
plt.figure(figsize=(10,10))
sns.heatmap(dconf_matrix,annot=True,fmt="d",cmap="Purples")
plt.title("Confusion Matrix DecisionTree")
plt.show()
TN=dcm[0][0]
FP=dcm[0][1]
FN=dcm[1][0]
TP=dcm[1][1]
sensitivity=TP/(TP+FN)
specificity=TN/(TN+FP)
print("DecisionTree")
print("True Negative:",TN)
print("False Positive:",FP)
print("False Negative:",FN)
print("True Positive:",TP)
print("Sensitivity:",sensitivity)
print("Specificity:",specificity)
print("\t\tClassification Report for DecisionTree")
print(classification_report(y_test,t_pred))


rf=RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(precision_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

dcm=confusion_matrix(y_test,y_pred)
dconf_matrix=pd.DataFrame(dcm,columns=["Predicted 0","Predicted 1"],index=["Actual 0","Actual 1"])
plt.figure(figsize=(10,10))
sns.heatmap(dconf_matrix,annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix RandomForest")
plt.show()

TN=dcm[0][0]
FP=dcm[0][1]
FN=dcm[1][0]
TP=dcm[1][1]
sensitivity=TP/(TP+FN)
specificity=TN/(TN+FP)
print("RandomForest")
print("True Negative:",TN)
print("False Positive:",FP)
print("False Negative:",FN)
print("True Positive:",TP)
print("Sensitivity:",sensitivity)
print("Specificity:",specificity)
print("\t\tClassification Report for RandomForest")
print(classification_report(y_test,y_pred))

def InitialiseEval():
    return {"model":[],"accuracy":[],"recall":[],"precision":[],"f1":[]}
def insertData(test,pred,model):
    eval_data=InitialiseEval()
    eval_data["model"].append(model)
    eval_data["accuracy"].append(accuracy_score(test,pred))
    eval_data["recall"].append(recall_score(test,pred))
    eval_data["precision"].append(precision_score(test,pred))
    eval_data["f1"].append(f1_score(test,pred))
    return eval_data
def appendData(data1,data2):
    for i in data1.keys():
        data1[i].extend(data2[i])
    return data1

eval_b=insertData(y_test,b_pred,"BnB")
eval_l=insertData(y_test,l_pred,"LogisticRegression")
eval_d=insertData(y_test,t_pred,"DecisionTree")
eval_r=insertData(y_test,y_pred,"RandomForest")
eval_all=appendData(eval_b,eval_l)
eval_all=appendData(eval_all,eval_d)
eval_all=appendData(eval_all,eval_r)

def plot_modelss(df):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].bar(df["model"], df["accuracy"], label="accuracy", color='magenta')
    axs[0, 0].set_title("Accuracy")

    axs[0, 1].bar(df["model"], df["precision"], label="precision", color='orange')
    axs[0, 1].set_title("Precision")

    axs[1, 0].bar(df["model"], df["recall"], label="recall", color='blue')
    axs[1, 0].set_title("Recall")

    axs[1, 1].bar(df["model"], df["f1"
                                  ""], label="f1_score", color='green')
    axs[1, 1].set_title("F1 Score")

    for ax in axs.flat:
        ax.set(xlabel="Model", ylabel="Score")
        ax.legend()

    plt.tight_layout()
    plt.show()

plot_modelss(eval_all)

lasso=Lasso(alpha=0.1)
lasso.fit(x_train,y_train)
print("coefficients:",lasso.coef_,"intercept:",lasso.intercept_)

print("mean_absolute_error:",mean_absolute_error(y_test,lasso.predict(x_test)))
print("mean_squared_error:",mean_squared_error(y_test,lasso.predict(x_test)))
print("R2",r2_score(y_test,lasso.predict(x_test)))




