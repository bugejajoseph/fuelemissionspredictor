####################################################################
##################### Multiple Linear Regression ###################
####################################################################

##################### Importing needed packages ####################

# Install required packages
import piplite

# Install the necessary packages
await piplite.install(['pandas'])
await piplite.install(['matplotlib'])
await piplite.install(['numpy'])
await piplite.install(['scikit-learn'])

# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set the inline backend for interactive plots
%matplotlib inline

##################### Downloading data #########################

# Function to download data from a URL
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

# Download the dataset and save it as FuelConsumption.csv
path = "https://<url>" # Replace the URL with the fuel consumption database
filename = "FuelConsumption.csv"
await download(path, filename)
#### alternatively use !wget -O FuelConsumption.csv <url>
# Read the data into a pandas DataFrame
df = pd.read_csv(filename)

# Display the first few rows of the dataset
df.head()

##################### Data Exploration ########################

# Summarize the data
df.describe()

# Select the relevant features for analysis
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

# Visualize the relationship between Engine Size and CO2 Emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

##################### Creating train and test dataset #####################

# Split the dataset into train and test sets (80% train, 20% test)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

##################### Train data distribution #####################

# Visualize the training data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

##################### Multiple Regression Model #####################

from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
regr = LinearRegression()

# Prepare the training data
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])

# Train the model
regr.fit(x_train, y_train)

# Print the coefficients of the model
print('Coefficients: ', regr.coef_)

##################### Prediction #####################

# Make predictions on the test data
x_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = regr.predict(x_test)

# Calculate the mean squared error (Residual sum of squares) and variance score (R^2)
print("Residual sum of squares: %.2f" % np.mean((y_hat - y_test) ** 2))
print('Variance score: %.2f' % regr.score(x_test, y_test))

##################### Exercise #####################

# Reinitialize the Linear Regression model for exercise
regr = LinearRegression()

# Prepare the training data with additional features
x_train_exercise = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y_train_exercise = np.asanyarray(train[['CO2EMISSIONS']])

# Train the model for exercise
regr.fit(x_train_exercise, y_train_exercise)

# Print the coefficients for exercise
print('Coefficients: ', regr.coef_)

# Make predictions on the test data for exercise
x_test_exercise = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y_test_exercise = np.asanyarray(test[['CO2EMISSIONS']])
y_hat_exercise = regr.predict(x_test_exercise)

# Calculate the mean squared error and variance score for exercise
print("Residual sum of squares: %.2f" % np.mean((y_hat_exercise - y_test_exercise) ** 2))
print('Variance score: %.2f' % regr.score(x_test_exercise, y_test_exercise))
