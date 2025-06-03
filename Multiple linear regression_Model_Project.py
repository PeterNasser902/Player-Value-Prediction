import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=False)

# Apply multiply Linear Regression on the selected features
cls = linear_model.LinearRegression()
start = time.time()
cls.fit(X_train, y_train)
end = time.time()
print("training time = ", end - start)

y_train_predicted = cls.predict(X_train)
print('Mean Square Error of train =', metrics.mean_squared_error(y_train, y_train_predicted))

prediction = cls.predict(X_test)
print('Mean Square Error of test =', metrics.mean_squared_error(y_test, prediction))

true_player_value = np.asarray(y_test)[0]
predicted_player_value = prediction[0]
print('True value for the first player in the test set in is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in is : ' + str(predicted_player_value))
