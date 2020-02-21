import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
sys.path.insert(1, 'F:/Git_repo/travelX/weather_days.py')
from weather_days import *


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
