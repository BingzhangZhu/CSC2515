'''
CSC 2515 Homework 2 Q2 Code
Collaborators: Zhimao Lin, Bingzhang Zhu
'''

from matplotlib.colors import Normalize
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from texttable import Texttable



def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features

def split_data(X, Y, train_size, test_size, seeder = 1):
    train_input_data, test_input_data, train_target_data, test_target_data = train_test_split(X, Y, train_size=train_size, test_size=test_size, shuffle=True, random_state=seeder)
    return train_input_data, test_input_data, train_target_data, test_target_data

def summarize_data(X, Y, features):

    X_df = pd.DataFrame(X, columns = features)
    print(X_df.describe())
    print("Target Data Summary")
    print(f"Min: {Y.min()}")
    print(f"Mean: {Y.mean()}")
    print(f"Max: {Y.max()}")
    print(f"Std: {Y.std()}")

def visualize(X, y, features):
    plt.figure(figsize=(15, 10))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        my_plot = plt.subplot(4, 4, i + 1)
        my_plot.scatter(X[:,i], y, marker='+')
        my_plot.set(title = f"{features[i]} vs. Median Price (in 1K)", xlabel = f"{features[i]}", ylabel = "Median Price (in 1K)")
    
    plt.tight_layout()
    plt.show()

def fit_regression(X, Y, features):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    t = Texttable()
    t.set_max_width(0)
    t.add_rows([features, regr.coef_])
    print(t.draw())
    print(f"Coefficients: {regr.coef_} and # of Coefficients {len(regr.coef_)}")
    print(f"Intercept: {regr.intercept_}")

    return regr

def test_predict(regr, X, Y):
    prediction = regr.predict(X)
    mse = mean_squared_error(Y, prediction)
    print(f"Mean squared error: {mse}")
    mae = mean_absolute_error(Y, prediction)
    print(f"Mean absolute error: {mae}")
    r2 = r2_score(Y, prediction)
    print(f"R 2 score: {r2}")

def main():
    # Load the data
    X, y, feature_list = load_data()
    train_input_data, test_input_data, train_target_data, test_target_data = split_data(X, y, 0.7, 0.3)
    
    # Summarize the data
    print(f"Features: {feature_list}")
    print(f"The number of features is {len(feature_list)}")
    print(f"The dimension of input data is {X.shape}.")
    print(f"The dimension of target data is {y.shape}.")

    print(f"There are {X.shape[0]} data points.")
    print(f"There are {y.shape[0]} targets.")

    print(f"Training data has {train_input_data.shape[0]} input points and {train_target_data.shape[0]} target points.")
    print(f"Test data has {test_input_data.shape[0]} input points and {test_target_data.shape[0]} target points.")
    summarize_data(X, y, feature_list)
    
    # Visualize the features
    visualize(X, y, feature_list)

    # Fit regression model
    linear_regression_model = fit_regression(train_input_data, train_target_data, feature_list)

    # Compute fitted values, MSE, etc.
    test_predict(linear_regression_model, test_input_data, test_target_data)


if __name__ == "__main__":
    main()

