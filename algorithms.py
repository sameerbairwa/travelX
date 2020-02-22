import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from weather_days import *


def decisionTreeRegression(new_weather,new_day,new_time):
    predict_time = pd.read_csv(r'dataset.csv')
    df = DataFrame(predict_time,columns=['Time Code','Day','Weather','Predicted Time'])

    x = df[['Time Code','Day','Weather']]
    y = df['Predicted Time']

    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, 
        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=3, max_leaf_nodes=None, 
        min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
    regressor.fit(x,y)

    new_weather = weather[new_weather]
    new_day = days[new_day]

    #print ('DecisionTree Regression Predicted time: \n', regressor.predict([[new_weather ,new_day,new_time]]))
    r = regressor.predict([[new_weather ,new_day,new_time]])
    return r[0]
def MultipleLinearRegression(new_weather,new_day,new_time):
    predict_time = pd.read_csv(r'dataset.csv')
    df = DataFrame(predict_time,columns=['Time Code','Day','Weather','Predicted Time'])
    #print(df)

    x = df[['Time Code','Day','Weather']]
    y = df['Predicted Time']

    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    
    new_weather = weather[new_weather]
    new_day = days[new_day]
    #print ('linear Regression Predicted time: \n', regr.predict([[new_weather ,new_day,new_time]]))
    r = regr.predict([[new_weather ,new_day,new_time]])
    return r[0]

def PolynomialRegression(new_weather,new_day,new_time):
    predict_time = pd.read_csv(r'dataset.csv')
    df = DataFrame(predict_time,columns=['Time Code','Day','Weather','Predicted Time'])
    #print(df)

    x = df[['Time Code','Day','Weather']]
    y = df['Predicted Time']

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    # X = x
    # Y = b0 + b1*X + b2*pow(X,2) + b3*pow(X,3)

    
    poly_reg = PolynomialFeatures(degree = 2)
    x_poly = poly_reg.fit_transform(x)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(x_poly, y)

    new_weather = weather[new_weather]
    new_day = days[new_day]

    #print ('Polynomial Regressison Predicted time: \n', lin_reg2.predict(poly_reg.fit_transform([[new_weather ,new_day,new_time]])))
    r = lin_reg2.predict(poly_reg.fit_transform([[new_weather ,new_day,new_time]]))
    return r[0]

def randomForestRegression(new_weather,new_day,new_time):
    predict_time = pd.read_csv(r'dataset.csv')
    df = DataFrame(predict_time,columns=['Time Code','Day','Weather','Predicted Time'])

    x = df[['Time Code','Day','Weather']]
    y = df['Predicted Time']

    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2)
    regressor.fit(x,y)
    
    new_weather = weather[new_weather]
    new_day = days[new_day]


    #print ('randomForest Regression Predicted time: \n', regressor.predict([[new_weather ,new_day,new_time]]))
    r = regressor.predict([[new_weather ,new_day,new_time]])

    return r[0]


#alpha = 0.01
#num_iters = 1000
#theta_init = np.zeros((3, 1))


def supportVectorRegression(new_weather,new_day,new_time):
    predict_time = pd.read_csv(r'dataset.csv')
    df = DataFrame(predict_time,columns=['Time Code','Day','Weather','Predicted Time'])

    x = df[['Time Code','Day','Weather']]
    y = df['Predicted Time']

    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(x,y)


    new_weather = weather[new_weather]
    new_day = days[new_day]

    r = sc_y.inverse_transform(regressor.predict(sc_x([[new_weather ,new_day,new_time]])))
    return r[0]
