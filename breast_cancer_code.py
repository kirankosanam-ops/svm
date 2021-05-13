import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

data = pd.read_csv("breast_cancer.csv")
# data.head()
# data.shape

data.isnull().values.any()
data.isnull().sum()

X = data.iloc[:, 1:10].values
# features[:, 3]

y = data.iloc[:, 10].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])
# features[:, 5]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,
                                                    test_size=0.20)

# Feature scaling the features for easy calculation
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = SVC(kernel="rbf", random_state=0)
clf.fit(X_train, y_train)
label_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, label_pred)
score = clf.score(X_test, y_test)
print(score * 100)

# custome input
vals = [6, 2, 5, 3, 9, 4, 7, 2, 2]
a = sc.transform(np.array(vals).reshape(1, -1))
# print(a)
# 2 for Benign and 4 for Malignant
cancer_type = ['', '', 'Benign', '', 'Malignant']
cancer_pred = clf.predict(a)
print(cancer_type[cancer_pred[0]])
