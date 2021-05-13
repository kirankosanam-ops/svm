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

features = data.iloc[:, 1:10].values
# features[:, 3]

label = data.iloc[:, 10].values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(features[:, 5:6])
features[:, 5:6] = imputer.transform(features[:, 5:6])
# features[:, 5]

features_train, features_test, label_train, label_test = train_test_split(features, label, random_state=0,
                                                                          test_size=0.20)

# Feature scaling the features for easy calculation
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)


clf = SVC(kernel="rbf", random_state=0)
clf.fit(features_train, label_train)
label_pred = clf.predict(features_test)

cm = confusion_matrix(label_test, label_pred)
score = clf.score(features_test, label_test)
print(score * 100)

vals = [6, 2, 5, 3, 9, 4, 7, 2, 2]
a = sc.transform(np.array(vals).reshape(1, -1))
print(a)

cancer_pred = clf.predict(a)
print(cancer_pred)
