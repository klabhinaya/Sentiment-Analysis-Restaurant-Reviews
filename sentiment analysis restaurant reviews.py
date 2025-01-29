import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv")
print(df)
print(df.head())
print(df.tail())
print(df.describe())
print(df.info())

# Correlation heatmap, excluding non-numeric columns
numeric_df = df.select_dtypes(include=[np.number])
c = numeric_df.corr()
sns.heatmap(c, annot=True, cmap='Greens')
plt.show()

# Boxplot for outliers in 'Profit'
outliers = ['Profit']
plt.rcParams['figure.figsize'] = [8, 8]
sns.boxplot(data=df[outliers], orient="v", palette="Set2", width=0.7)
plt.title("Outliers Variable Distribution")
plt.ylabel("Profit Range")
plt.xlabel("Continuous Variable")
plt.show()

# Boxplot for 'Profit' by 'State'
sns.boxplot(x='State', y='Profit', data=df)
plt.show()

# Distribution plot for 'Profit'
sns.histplot(df['Profit'], bins=5, kde=True)
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Preparing the data for modeling
X = df.iloc[:, 0:4].values
y = df.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
X1 = pd.DataFrame(X)
print(X1.head())

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)
print(x_train)

# Linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)

# Model performance
testing_data_model_score = model.score(x_test, y_test)
print("Model Score/Performance on Testing data:", testing_data_model_score)

training_data_model_score = model.score(x_train, y_train)
print("Model Score/Performance on Training data:", training_data_model_score)

df_results = pd.DataFrame(data={'Predicted value': y_pred.flatten(), 'Actual Value': y_test.flatten()})
print(df_results)

# Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2Score = r2_score(y_test, y_pred)
print("R2 score of model is:", r2Score * 100)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error is:", mse * 100)

rmse = np.sqrt(mse)
print("Root Mean Squared Error is:", rmse * 100)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error is:", mae)
