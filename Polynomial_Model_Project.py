import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from Pre_processing import *

# Load players data
data = pd.read_csv('Final_Fifa_Data_after_preprocessing.csv')

fifa_data = data.iloc[:, :]
X = data.iloc[:, 0:76]
Y = data['value']

# Feature Selection
# Get the correlation between the features
corr = fifa_data.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['value']) > 0.5]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = fifa_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
X = X[top_feature]

# feature_scaling
X = featureScaling(X, 0, 1)

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

poly_features = PolynomialFeatures(degree=3)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
start = time.time()
poly_model.fit(X_train_poly, y_train)
end = time.time()
print("training time = ", end - start)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

print('Mean Square Error for train', metrics.mean_squared_error(y_train, y_train_predicted))

print('Mean Square Error for test', metrics.mean_squared_error(y_test, prediction))

true_player_value = np.asarray(y_test)
predicted_player_value = prediction
print('True value for the first player in the test set in is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in is : ' + str(predicted_player_value))
