# KNN #

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + '\n...')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
print(X_train.shape)

import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
iris_dataframe = DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',
                           hist_kwds={'bins':20}, s=60, alpha=.8)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

import numpy as np

X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new)
print('prediction: {}'.format(prediction))
print(iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
print(y_pred)
print('accuracy: {}'.format(np.mean(y_pred==y_test)))
knn.score(X_test, y_test)