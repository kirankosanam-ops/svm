import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.svm import SVC

df = pd.read_csv('cleveland.csv', header=None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df.isnull().sum()

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
df['sex'] = df.sex.map({'female': 0, 'male': 1})

# data preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# print(X)

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(f'X train: \n{X_train}')
# print(f'X test: \n{X_test}')
# print(f'y train: \n{y_train}')
# print(f'y test: \n{y_test}')


# scaling/transforming data
sc = ss()
X_train = sc.fit_transform(X_train)
# print(f'X - test : \n{X_train}')
X_test = sc.transform(X_test)
# print(f'X - test : \n{X_train}')

# SVM classifier
classifier = SVC(kernel='poly', random_state=0)
classifier.fit(X_train, y_train)
y_pred_svm = classifier.predict(X_test)
# print(f'y pred: \n{y_pred}')

# confusion matrix
cm_test = confusion_matrix(y_pred_svm, y_test)
# print(f'confusion matrix test: \n{cm_test}')

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)
# print(f'confusion matrix train: \n{cm_train}')


print()
print('Accuracy = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train) * 100))

vals = [62, 0, 4, 160, 164, 0, 2, 145, 0, 6.2, 3, 3, 7]
ans = [1]
vals1 = [66, 1, 4, 120, 302, 0, 2, 151, 0, 0.4, 2, 0, 3]
ans1 = [0]

res_mapping = {0:'No', 1:'yes'}
a = sc.transform(np.array(vals).reshape(1, -1))
res = classifier.predict(a)
print(res_mapping[res[0]])


# 1, 3, 5 set -1 odd
# 2, 4 set 2 even