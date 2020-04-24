__author__ = 'compiler'

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
print(concrete_data.head())
print(concrete_data.shape)

# So, there are approximately 1000 samples to train our model on.
# Because of the few samples, we have to be careful not to overfit the training data.
#
# Let's check the dataset for any missing values.
print(concrete_data.describe())
print(concrete_data.isnull().sum())

# The data looks very clean and is ready to be used to build our model.
#
# Split data into predictors and target
# The target variable in this problem is the concrete sample strength.
# Therefore, our predictors will be all the other columns.

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# Let's do a quick sanity check of the predictors and the target dataframes.
print(predictors.head())
print(target.head())

# Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation.
predictors_norm = (predictors - predictors.mean()) / predictors.std()
print(predictors_norm.head())

# Let's save the number of predictors to n_cols since we will need this number when building our network.
n_cols = predictors_norm.shape[1] # number of predictors


# Build a Neural Network
# Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model
# The above function create a model that has two hidden layers, each of 50 hidden units.


# Train and Test the Network
# Let's call the function now to create our model.

# build the model
model = regression_model()

# Next, we will train and test the model at the same time using the fit method.
# We will leave out 30% of the data for validation and we will train the model for 100 epochs.
# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)



