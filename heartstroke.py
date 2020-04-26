import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("train_2v.csv")
print(df.head())

# Basic exploratory analysis
print(df.shape)
print(df.describe())

df = df.drop("id", axis=1)
print(df.dtypes)
print(df.corr('pearson'))

print(df.isnull().sum() / len(df))
df["smoking_status"].value_counts()
df["smoking_status"] = df["smoking_status"].replace(np.nan, 'None')
df["bmi"] = df["bmi"].replace(np.nan, np.nanmedian(df["bmi"]))

print(sum(df["stroke"] == 1) / len(df))  # imbalanced dataset
print(df.corr("pearson"))

# Data preprocessing

df_dummy = pd.get_dummies(df[["gender", 'Residence_type', 'smoking_status', 'work_type', "ever_married"]],
                          drop_first=True)
min_max_scaler = preprocessing.MinMaxScaler()
scaled_columns = min_max_scaler.fit_transform(df[["age", "avg_glucose_level", "bmi"]])
scaled_columns = pd.DataFrame(scaled_columns, columns=["age", "avg_glucose_level", "bmi"])
df = pd.concat([scaled_columns, df_dummy, df[["stroke", "hypertension", "heart_disease"]]], axis=1)

X_train = df.drop("stroke", axis=1)
y_train = df["stroke"]

# Look into how coefficients change wrt the choice of C, which denotes the inverse of regularization strength.
C = np.logspace(-3, 3, 30)
coefs = []
for c in C:
    clf = LogisticRegression(solver='liblinear', C=c, penalty='l1', class_weight="balanced")
    clf.fit(X_train, y_train)
    coefs.append(list(clf.coef_[0]))
coefs = np.array(coefs)

plt.figure(figsize=(10, 10))

for i in range(coefs.shape[1]):
    plt.plot(C, coefs[:, i])
plt.xscale('log')
plt.title('L1 penalty - Logistic regression')
plt.xlabel('C')
plt.ylabel('Coefficient value')
plt.legend(df.columns)
plt.savefig('coefs_lasso.png')


# Grid search to find best C value
hyperparameters = dict(C=C)

logistic = LogisticRegression(solver='liblinear', penalty='l1', class_weight="balanced")
clf = GridSearchCV(logistic, hyperparameters, cv=10, verbose=0)
best_model = clf.fit(X_train, y_train)
print('Best C:', best_model.best_estimator_.get_params()['C'])

scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
std_error = scores_std / np.sqrt(10)
one_se_index = np.argmin( np.abs(scores - (np.max(scores) - std_error[np.argmax(scores)])))
c_best = C[one_se_index]
print('1se C:', c_best)

scores = 1-scores # error
plt.figure().set_size_inches(8, 6)
plt.semilogx(C, scores)
plt.ylabel('CV error')
plt.xlabel('Inverse of regularization strength')
plt.axhline(np.min(scores), linestyle='--', color='.5')
plt.axvline(c_best, linestyle='--', color='.5')
plt.xlim([C[0], C[-1]])
plt.savefig('cv_error_lasso.png')

# Fit the model to whole training data with selected C value
clf = LogisticRegression(solver='liblinear', C=c_best, penalty='l1', class_weight="balanced")
clf.fit(X_train, y_train)

features = X_train.columns
importances = np.abs(clf.coef_)
indices = np.argsort(importances)

plt.figure(figsize=(25, 10))

feat_importances = {features[i]:importances[0,i] for i in indices[0]}
plt.barh(range(len(feat_importances)), list(feat_importances.values()), align='center')
plt.yticks(range(len(feat_importances)), list(feat_importances.keys()))
plt.xlabel("Absolute Value of Coefficients")
plt.title("Feature Importance by Lasso")
plt.savefig('lasso_importances.png')

