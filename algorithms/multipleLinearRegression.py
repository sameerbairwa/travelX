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