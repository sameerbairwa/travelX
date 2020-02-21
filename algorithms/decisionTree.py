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