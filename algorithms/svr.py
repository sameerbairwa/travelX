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
