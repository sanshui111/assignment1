# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:39:21 2024

@author: zzzzzm
"""


import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
data_path = 'D:/2024-1学年/sph6004/sph6004_assignment1_data.csv'
df = pd.read_csv(data_path)
rows, cols = df.shape
print(f"数据集有 {rows} 行和 {cols} 列。")
print("\n描述性统计信息：")
print(df.describe())
print("\n列的缺失值数量：")
print(df.isnull().sum())

for col in ['gender', 'race']:
    mode_value = df[col].mode()[0]  
    df[col].fillna(mode_value, inplace=True)   
calpercent = df.drop(columns=['gender', 'race']).isnull().mean() * 100
high_col = calpercent[calpercent > 50].index
df.drop(columns=high_col, inplace=True)
low_col = calpercent[calpercent <= 50].index
for col in low_col:
    if col != 'gender' and col != 'race':
        median_val = df[col].mean()
        df[col].fillna(median_val, inplace=True)
num_col = df.select_dtypes(include=['number']).columns
zscores = stats.zscore(df[num_col])
abs_zs = abs(zscores)
need_keep = (abs_zs < 3).all(axis=1)  
df = df[need_keep]
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object': 
        df[col] = encoder.fit_transform(df[col])
x = df.drop(columns=['id','aki'])
y = df['aki']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
x_train_scal = scaler.fit_transform(x_train)
x_test_scal = scaler.transform(x_test)
df_x_train_scal = pd.DataFrame(x_train_scal, columns=x_train.columns)
df_x_test_scal = pd.DataFrame(x_test_scal, columns=x_test.columns)
df_scalfile = pd.concat([df_x_train_scal, df_x_test_scal])
df_scalfile.to_csv('D:/2024-1学年/sph6004/scaled_data.csv', index=False)
print(df_scalfile)
data = pd.read_csv('D:/2024-1学年/sph6004/scaled_data.csv')
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(x_train_scal, y_train)
plt.figure(figsize=(10, 6))
sns.barplot(x=forest.feature_importances_, y=x.columns)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
selector = SelectFromModel(forest,threshold=0.01,prefit=True)
x_train_sel = selector.transform(x_train_scal)
x_test_sel = selector.transform(x_test_scal)
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(x_train_sel, y_train)
    y_pred = model.predict(x_test_sel)
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, zero_division=1)
    results[name] = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Support': support
    }
for name, model in models.items():
    model.fit(x_train_sel, y_train)
    y_pred = model.predict(x_test_sel)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
plt.figure(figsize=(16, 12))
for i, metric in enumerate(['Precision', 'Recall', 'F1 Score', 'Support'], 1):
    plt.subplot(2, 2, i)
    sns.barplot(x=list(results.keys()), y=[results[model][metric][1] for model in results.keys()])
    plt.title(f'{metric} for Class 1')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()
