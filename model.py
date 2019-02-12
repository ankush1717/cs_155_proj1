#ACTUAL MODEL CODE
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split



data = np.loadtxt(open("train_2008.csv", "rb"), delimiter=",", skiprows=1)
test_data = np.loadtxt(open("test_2008.csv", "rb"), delimiter=",", skiprows=1)

X = data[:, 3:382]
y = data[:, 382]

X_test = test_data[:, 3:382]

clf = RandomForestClassifier(n_estimators = 190, min_samples_leaf = 20)
clf.fit(X, y)



results = clf.predict_proba(X_test)

with open('output_1.csv', 'w') as output:
    output.write('id,target\n')
    for i in range(len(test_data)):
        output.write(str(int(test_data[i][0])) + ',' + str(results[i]) + '\n')
