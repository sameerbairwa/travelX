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
