import xlsxwriter
import urllib.request  as urllib2
import json
import datetime
import time
import numpy as np
import calendar
import math
import matplotlib.pyplot as plt
import pandas as pd
import xlrd
import csv
    
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def dat(dd,mm,yyyy):
    actual_date = datetime.date(yyyy, mm, dd)
    dates_arr=[]
    ct=actual_date-datetime.date.today()

    ct=ct.days
    #print(ct)
    pt=ct/4
    rn1=32+pt
    rn2=32+ct
    #print(rn1+rn2)


    for i in range(0,rn1):
        date_N_days_ago = actual_date - datetime.timedelta(days=7)
        actual_date=date_N_days_ago
        dates_arr.append(date_N_days_ago)

    actual_date = datetime.date(yyyy, mm, dd)

    for i in range(0,rn2):
        date_N_days_ago = actual_date - datetime.timedelta(days=1)
        actual_date=date_N_days_ago
        dates_arr.append(date_N_days_ago)

    final_dates_arr=[]

    for i in dates_arr:
        if i not in final_dates_arr:
            final_dates_arr.append(i)

    #print(final_dates_arr)

    #print(len(final_dates_arr))
    fin=[]

    for i in final_dates_arr:
        if i<=datetime.date.today():
            fin.append(i)   
            

    #print(len(fin))
    #print(fin)

    #print(len(dates_arr))

def fn_w(wth,time):
    l1=[0,0]
    if(wth=='Foggy'):
        l1[0]=time-(.05*time)
        l1[1]=time+(.3*time)

    elif(wth=='Partly Cloudy'):
        l1[0]=time-(.1*time)
        l1[1]=time+(.05*time)
    elif(wth=='Clear'):
        l1[0]=time-(0.1*time)
        l1[1]=time+(0.1*time)
    elif(wth=='Humid and Mostly Cloudly'):
        l1[0]=time-(.05*time)
        l1[1]=time+(.1*time)
    elif(wth=='Humid and Foggy'):
        l1[0]=time-(.03*time)
        l1[1]=time+(.35*time)
    elif(wth=='Mostly Cloudy'):
        l1[0]=time-(.05*time)
        l1[1]=time+(.07*time)
        
    elif(wth=='Humid'):
        l1[0]=time-(.05*time)
        l1[1]=time+(.05*time)
        
    elif(wth=='Humid and Partly Cloudy'):
        l1[0]=time-(.1*time)
        l1[1]=time+(.07*time)
        
    elif(wth=='Drizzle'):
        l1[0]=time-(.05*time)
        l1[1]=time+(.1*time)
    elif(wth=='Light Rain'):
        l1[0]=time-(.05*time)
        l1[1]=time+(.15*time)
    elif(wth=='Heavy Rain'):
        l1[0]=time-(.3*time)
        l1[1]=time+(.3*time)



    #print(l1[0])
    #print(l1[1])
    return l1[1],l1[0]


def fn_ti(time_interval,time,day):
    l2=[0,0]
    #print(time_interval)
    #print(time)
    if(day=='Monday'):
        if(time_interval>=0 and time_interval<=8):
            l2[0]=time-(.7*time)
            l2[1]=time
        elif(time_interval>8 and time_interval<=11):
             l2[0]=time-(.1*time)
             l2[1]=time+(.4*time)
        elif(time_interval>11 and time_interval<=17):
            l2[0]=time-(.1*time)
            l2[1]=time+(.2*time)
        elif(time_interval>17 and time_interval<=21):
            l2[0]=time-(.15*time)
            l2[1]=time+(.3*time)
        elif(time_interval>21 and time_interval<=24):
            l2[0]=time-(.2*time)
            l2[1]=time+(.2*time)
    elif(day=='Tuesday'):
        if(time_interval>=00 and time_interval<=8):
            l2[0]=time-(.7*time)
            l2[1]=time
        elif(time_interval>8 and time_interval<=11):
             l2[0]=time-(.1*time)
             l2[1]=time+(.4*time)
        elif(time_interval>11 and time_interval<=17):
            l2[0]=time-(.1*time)
            l2[1]=time+(.2*time)
        elif(time_interval>17 and time_interval<=21):
            l2[0]=time-(.15*time)
            l2[1]=time+(.3*time)
        elif(time_interval>21 and time_interval<=24):
            l2[0]=time-(.2*time)
            l2[1]=time+(.2*time)
    elif(day=='Wednesday'):
        if(time_interval>=00 and time_interval<=8):
            l2[0]=time-(.7*time)
            l2[1]=time
        elif(time_interval>8 and time_interval<=11):
             l2[0]=time-(.1*time)
             l2[1]=time+(.4*time)
        elif(time_interval>11 and time_interval<=17):
            l2[0]=time-(.1*time)
            l2[1]=time+(.2*time)
        elif(time_interval>17 and time_interval<=21):
            l2[0]=time-(.15*time)
            l2[1]=time+(.3*time)
        elif(time_interval>21 and time_interval<=24):
            l2[0]=time-(.2*time)
            l2[1]=time+(.2*time)
    elif(day=='Thursday'):
        if(time_interval>=00 and time_interval<=8):
            l2[0]=time-(.7*time)
            l2[1]=time
        elif(time_interval>8 and time_interval<=11):
             l2[0]=time-(.1*time)
             l2[1]=time+(.4*time)
        elif(time_interval>11 and time_interval<=17):
            l2[0]=time-(.1*time)
            l2[1]=time+(.2*time)
        elif(time_interval>17 and time_interval<=21):
            l2[0]=time-(.15*time)
            l2[1]=time+(.3*time)
        elif(time_interval>21 and time_interval<=24):
            l2[0]=time-(.2*time)
            l2[1]=time+(.2*time)
    elif(day=='Friday'):
        if(time_interval>=00 and time_interval<=8):
            l2[0]=time-(.7*time)
            l2[1]=time
        elif(time_interval>8 and time_interval<=11):
             l2[0]=time-(.1*time)
             l2[1]=time+(.4*time)
        elif(time_interval>11 and time_interval<=17):
            l2[0]=time-(.1*time)
            l2[1]=time+(.2*time)
        elif(time_interval>17 and time_interval<=21):
            l2[0]=time-(.15*time)
            l2[1]=time+(.3*time)
        elif(time_interval>21 and time_interval<=24):
            l2[0]=time-(.2*time)
            l2[1]=time+(.2*time)
    elif(day=='Saturday'):
        if(time_interval>=00 and time_interval<=8):
            l2[0]=time-(.7*time)
            l2[1]=time
        elif(time_interval>8 and time_interval<=11):
             l2[0]=time-(.3*time)
             l2[1]=time
        elif(time_interval>11 and time_interval<=18):
            l2[0]=time-(.3*time)
            l2[1]=time
        elif(time_interval>18 and time_interval<=21):
            l2[0]=time
            l2[1]=time+(.2*time)
        elif(time_interval>21 and time_interval<=24):
            l2[0]=time-(.3*time)
            l2[1]=time
    elif(day=='Sunday'):
        if(time_interval>=00 and time_interval<=8):
            l2[0]=time-(.7*time)
            l2[1]=time
        elif(time_interval>8 and time_interval<=11):
             l2[0]=time-(.3*time)
             l2[1]=time
        elif(time_interval>11 and time_interval<=18):
            l2[0]=time-(.3*time)
            l2[1]=time
        elif(time_interval>18 and time_interval<=21):
            l2[0]=time
            l2[1]=time+(.2*time)
        elif(time_interval>21 and time_interval<=24):
            l2[0]=time-(.3*time)
            l2[1]=time

    #print(l2[0])
    #print(l2[1])
            
    return l2[1],l2[0]


def curr_time_fetch(lat_lon_source,lat_lon_dest):
    lat_lon_source[0]=str(lat_lon_source[0])
    lat_lon_source[1]=str(lat_lon_source[1])

    lat_lon_dest[0]=str(lat_lon_dest[0])
    lat_lon_dest[1]=str(lat_lon_dest[1])

    url="https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins="+lat_lon_source[0]+","+lat_lon_source[1]+"&destinations="+lat_lon_dest[0]+","+lat_lon_dest[1]+"&key=AIzaSyCTHwCnl_VrqowvDPFeEzQXgobof1jR4aM"
    #print(url)
    json_file=urllib2.urlopen(url)
    data=json.load(json_file)
    return data['rows'][0]['elements'][0]['duration']['text']


def convertTime(s):
    s = s.split()
    if len(s) == 4:
        hours = int(s[0])
        mins = int(s[2])
        TotalMins = mins + hours*60
    else:
        TotalMins = int(s[0])
    
    return TotalMins





def lat_lon_calc(addr):
    url="https://maps.googleapis.com/maps/api/geocode/json?address="+addr+"&key=AIzaSyCTHwCnl_VrqowvDPFeEzQXgobof1jR4aM"
    #print(url)
    json_file=urllib2.urlopen(url)
    data=json.load(json_file)
    lat_lng=[]
    
    lat_lng.append(data['results'][0]['geometry']['location']['lat'])
    lat_lng.append(data['results'][0]['geometry']['location']['lng'])

    return lat_lng

def weather_details(lat_lng_list,date,time):

    date=str(date)
    time=str(time)
    
    lat_lng_list[0]=str(lat_lng_list[0])
    lat_lng_list[1]=str(lat_lng_list[1])

    url="https://api.darksky.net/forecast/4712f03bee5326bce0e86df121301a88/"+lat_lng_list[0]+","+lat_lng_list[1]+","+date+"T"+time
    #print(url)
    json_file=urllib2.urlopen(url)
    data=json.load(json_file)
    d=data['currently']['summary']
    return d

def dataset_create_2(ul,ll,n,v):
    dataset = np.zeros((n,4),dtype=int)
    ul=math.ceil(ul)
    ll=math.floor(ll)
    ul=int(ul)
    ll=int(ll)
    days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    time = {'00-01':1,'01-02':2,'02-03':3,'03-04':4,'04-05':5,'05-06':6,'06-07':7,'07-08':8,
            '08-09':9,'09-10':10,'10-11':11,'11-12':12,'12-13':13,'13-14':14,'14-15':15,
            '15-16':16,'16-17':17,'17-18':18,'18-19':19,'19-20':20,'20-21':21,'21-22':22,
            '22-23':23,'23-24':24}
    weather = {'Foggy':1,
               'Partly Cloudy':2,
               'Clear':3,
               'Mostly Cloudy':4,
               'Humid and Mostly Cloudy':5,
               'Humid and Foggy':6,
               'Humid':7,
               'Humid and Partly Cloudy':8,
               'Drizzle':9,
               'Light Rain':10,
               'Heavy Rain':11}
    #print(ll)
    #print(ul)
    if '-' in v:
        for i in range(0,n):
            dataset[i][0] = time[v]
            dataset[i][1] = np.random.randint(1,8)
            dataset[i][2] = np.random.randint(1,12)
            dataset[i][3] = np.random.randint(ll,ul)
    elif v[len(v)-3:] == 'day':
        for i in range(0,n):
            dataset[i][0] = np.random.randint(1,25)
            dataset[i][1] = days[v]
            dataset[i][2] = np.random.randint(1,12)
            dataset[i][3] = np.random.randint(ll,ul)
    else:
        for i in range(0,n):
            dataset[i][0] = np.random.randint(1,25)
            dataset[i][1] = np.random.randint(1,8)
            dataset[i][2] = weather[v]
            dataset[i][3] = np.random.randint(ll,ul)
    return dataset

def calc_day(date):
    w=datetime.datetime.strptime(date, '%d-%m-%Y')
    day=calendar.day_name[w.weekday()]

    return day

def ti(time):
    
    t=int(time[0:2])
    e=int(time[3:5])
    if(t<10):
        k="0"+str(t)
    else:
        k=str(t)
    if(e>=0 and e<60):
        e=t+1
    if(e<10):
        e="0"+str(e)
    else:
        e=str(e)
    #print(k+"-"+str(e))

    l=k+"-"+str(e)

    return l

def writer(workbook,worksheet,dataset,rr,ur):
    j=0
    #print('dne')
    #print(ur)
    #print(rr)
    for row in range(rr,ur):
        col=0
        
        worksheet.write(row, col,dataset[j][0])
        worksheet.write(row, col+1,dataset[j][1])
        worksheet.write(row, col+2,dataset[j][2])
        worksheet.write(row, col+3,dataset[j][3])
        j=j+1

        

def writing(workbook,worksheet):
    worksheet.write(0,0,'Time Code')
    worksheet.write(0,1,'Day')
    worksheet.write(0,2,'Weather')
    worksheet.write(0,3,'Predicted Time')



def simulatorfn(ul_w,ll_w,ul_ti,ll_ti,time_int,day,weather,curr_time):




    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    workbook = xlsxwriter.Workbook('trial.xlsx')
    worksheet = workbook.add_worksheet()
    time_codes= {'00-01':1,'01-02':2,'02-03':3,'03-04':4,'04-05':5,'05-06':6,'06-07':7,'07-08':8,'08-09':9,'09-10':10,'10-11':11,'11-12':12,'12-13':13,'13-14':14,'14-15':15,'15-16':16,'16-17':17,'17-18':18,'18-19':19,'19-20':20,'20-21':21,'21-22':22,'22-23':23,'23-24':24}
    

    writing(workbook,worksheet)

    time=['00-01','01-02','02-03','03-04','04-05','05-06','06-07','07-08','08-09','09-10','10-11','11-12','12-13','13-14','14-15','15-16','16-17','17-18','18-19','19-20','20-21','21-22','22-23','23-24']

    weathers=['Foggy','Partly Cloudy','Clear','Mostly Cloudy','Humid and Mostly Cloudy','Humid and Foggy','Humid','Humid and Partly Cloudy','Drizzle','Light Rain','Heavy Rain']

    
    dataset=dataset_create_2(ul_ti,ll_ti,4000,time_int)
    writer(workbook,worksheet,dataset,1,4000)
    dataset=dataset_create_2(ul_w,ll_w,2000,weather)
    writer(workbook,worksheet,dataset,4000,6000)

    i_low=6000
    i_high=i_low+50

    for i in days:
        for j in time:
            if i is not day and j is not time_int :
                time_it=ti(j)
                time_c=time_codes[time_it]
                #print('ok')
                #print(time_c)
                #print(curr_time)
                #print(i)
                ul,ll=fn_ti(time_c,curr_time,i)
                dataset=dataset_create_2(ul,ll,50,i)
                writer(workbook,worksheet,dataset,i_low,i_high)
                i_low=i_low+50
                i_high=i_high+50
    #print(i_low)
    #print(i_high)

    workbook.close()


def csv_from_excel():
    wb = xlrd.open_workbook('trial.xlsx')
    sh = wb.sheet_by_name('Sheet1')
    dataset = open('dataset.csv', 'w')
    wr = csv.writer(dataset, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    dataset.close()


def MultipleLinearRegression(new_weather,new_day,new_time):
    predict_time = pd.read_csv(r'dataset.csv')
    df = DataFrame(predict_time,columns=['Time Code','Day','Weather','Predicted Time'])
    #print(df)

    x = df[['Time Code','Day','Weather']]
    y = df['Predicted Time']

    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    weather = {'Foggy':1,
               'Partly Cloudy':2,
               'Clear':3,
               'Mostly Cloudy':4,
               'Humid and Mostly Cloudy':5,
               'Humid and Foggy':6,
               'Humid':7,
               'Humid and Partly Cloudy':8,
               'Drizzle':9,
               'Light Rain':10,
               'Heavy Rain':11} 
    
    new_weather = weather[new_weather]
    days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
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

    #x = dataset.iloc[:,:-1].values
    #y = dataset.iloc[:,-1].values

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    # X = x
    # Y = b0 + b1*X + b2*pow(X,2) + b3*pow(X,3)

    
    poly_reg = PolynomialFeatures(degree = 2)
    x_poly = poly_reg.fit_transform(x)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(x_poly, y)

    weather = {'Foggy':1,
               'Partly Cloudy':2,
               'Clear':3,
               'Mostly Cloudy':4,
               'Humid and Mostly Cloudy':5,
               'Humid and Foggy':6,
               'Humid':7,
               'Humid and Partly Cloudy':8,
               'Drizzle':9,
               'Light Rain':10,
               'Heavy Rain':11} 
    
    new_weather = weather[new_weather]
    days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    new_day = days[new_day]

    #print ('Polynomial Regressison Predicted time: \n', lin_reg2.predict(poly_reg.fit_transform([[new_weather ,new_day,new_time]])))
    r = lin_reg2.predict(poly_reg.fit_transform([[new_weather ,new_day,new_time]]))
    return r[0]


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


    weather = {'Foggy':1,
               'Partly Cloudy':2,
               'Clear':3,
               'Mostly Cloudy':4,
               'Humid and Mostly Cloudy':5,
               'Humid and Foggy':6,
               'Humid':7,
               'Humid and Partly Cloudy':8,
               'Drizzle':9,
               'Light Rain':10,
               'Heavy Rain':11} 
    
    new_weather = weather[new_weather]
    days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    new_day = days[new_day]

    r = sc_y.inverse_transform(regressor.predict(sc_x([[new_weather ,new_day,new_time]])))
    return r[0]





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


    weather = {'Foggy':1,
               'Partly Cloudy':2,
               'Clear':3,
               'Mostly Cloudy':4,
               'Humid and Mostly Cloudy':5,
               'Humid and Foggy':6,
               'Humid':7,
               'Humid and Partly Cloudy':8,
               'Drizzle':9,
               'Light Rain':10,
               'Heavy Rain':11} 
    
    new_weather = weather[new_weather]
    days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    new_day = days[new_day]


    #print ('DecisionTree Regression Predicted time: \n', regressor.predict([[new_weather ,new_day,new_time]]))
    r = regressor.predict([[new_weather ,new_day,new_time]])
    return r[0]




def randomForestRegression(new_weather,new_day,new_time):
    predict_time = pd.read_csv(r'dataset.csv')
    df = DataFrame(predict_time,columns=['Time Code','Day','Weather','Predicted Time'])

    x = df[['Time Code','Day','Weather']]
    y = df['Predicted Time']

    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_split=2)
    regressor.fit(x,y)


    weather = {'Foggy':1,
               'Partly Cloudy':2,
               'Clear':3,
               'Mostly Cloudy':4,
               'Humid and Mostly Cloudy':5,
               'Humid and Foggy':6,
               'Humid':7,
               'Humid and Partly Cloudy':8,
               'Drizzle':9,
               'Light Rain':10,
               'Heavy Rain':11} 
    
    new_weather = weather[new_weather]
    days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    new_day = days[new_day]


    #print ('randomForest Regression Predicted time: \n', regressor.predict([[new_weather ,new_day,new_time]]))
    r = regressor.predict([[new_weather ,new_day,new_time]])

    return r[0]


#alpha = 0.01
#num_iters = 1000
#theta_init = np.zeros((3, 1))


def featureNormalize(x_m):
    mu = np.zeros((1,x_m.shape[1]))
    sigma = np.zeros((1,x_m.shape[1]))
    x_norm = x_m.astype(float)
    
    for i in range(0,len(mu)+1):
        mu[:,i] = x_m[:,i].mean()
        sigma[:,i] = x_m[:,i].std()
        x_norm[:,i] = (x_m[:,i] - mu[:,i])/sigma[:,i]
    return (x_norm, mu, sigma)

def computeCost_m(x, y, theta):
    m = len(y)
    h_x = np.dot(x, theta)
    j = np.sum(np.square(h_x - y))/(2*m)
    return j

def gradientDescentMulti(X, Y, theta, alpha, num_iters):
    m = len(Y)
    p = np.copy(X)
    t = np.copy(theta)
    j = []
    #print('Running Gradient Descent')
    for i in range(0,num_iters+1):
        cost = computeCost_m(p, Y, t)
        j.append(cost)
        h_x = np.dot(p, t)
        err = h_x - Y
        for f in range(theta.size):
            t[f] = t[f] - alpha/m *(np.sum((np.dot(p[:,f].T, err))))
    #return theta and cost 
    return j, t


def ensemble(r1,r2,r3,r4,ltime):
    ltime = ltime.split(':')
    h = int(ltime[0])
    m = int(ltime[1])
    finalTime = (.25*r1  + .1*r2 +.3*r3 + .35*r4)

    if finalTime > 60:
        rem = int(finalTime % 60)
        finalTime = finalTime//60
        #print(int(finalTime) , ' hours ' , int(rem) , ' minutes')
        if(rem > m):
            h = h-1-finalTime
            m = 60 - (rem-m)
        else:
            h = h - finalTime
            m = m - rem
        
    else:
        #print(finalTime,'minutes')
        if(finalTime > m):
            h = h-1
            m = 60 - (finalTime-m)
        else:
            m = m - finalTime
    if h < 0:
        h = 24 + h
    print(int(h) ,':',int(m))

# def afterTime(s,time_int):
#     s = s.split()
#     if len(s) == 4:
#         gh = int(s[0])
#         gm = int(s[2])
#     else:
#         gh = 0
#         gm = int(s[0])

#     time_int = time_int.split(':')
#     h = int(time_int[0])
#     m = int(time_int[1])
    
#     rem = gm
#     finalTime = gh
#         #print(int(finalTime) , ' hours ' , int(rem) , ' minutes')
#     if(rem >  m):
#         h = h-1-finalTime
#         m = 60 - (rem-m)
#     else:
#         h = h - finalTime
#         m = m - rem   
        
   
#     if h < 0:
#         h = 24 + h
#     h = int(h)
#     m = int(m)
#     return (h,m)

def datecheck(date):
    date_t=datetime.datetime.strptime (date,"%d-%m-%Y")
    flag=1
    date_today=datetime.datetime.now().strftime ("%d-%m-%Y")
    date_today=datetime.datetime.now().strptime (date_today,"%d-%m-%Y")
    if(date_t<date_today):
        flag=0

    actual_date=date_today + datetime.timedelta(days=7)

    if(date_t>actual_date):
        flag=0
    return flag







if __name__ == '__main__':
    
    '''date=raw_input("Enter date of travel (dd-mm-yyyy) : ")

    time_int=raw_input("Enter Time You want to reach: ")

    
    source=raw_input("Enter Source: ")

    dest=raw_input("Enter Destination: ")'''

    date='20-12-2018'
    time_int='10:12'
 
    
    source='Jaypee Institute of information technology, Noida, Sector-62'
    dest='jaipur'
    flag=datecheck(date)
    if(flag==0):
        print('Date is out of our computation range')
        exit()


    day=calc_day(date)


    #print(day)

    lat_lon_source=[]
    lat_lon_dest=[]

    src=source.replace(" ","+")
    dst=dest.replace(" ","+")

    lat_lon_source=lat_lon_calc(src)
    lat_lon_dest=lat_lon_calc(dst)


    #print(lat_lon_source)
    #print(lat_lon_dest)

    curr_time=curr_time_fetch(lat_lon_source,lat_lon_dest)
    #print(curr_time)

    

    curr_time=convertTime(curr_time)

    date_t=datetime.datetime.strptime (date,"%d-%m-%Y")
    date_t=datetime.datetime.strftime (date_t,"%Y-%m-%d")

    
    # h,m = afterTime(curr_time_fetch(lat_lon_source,lat_lon_dest),time_int)
    # if len(str(h)) == 1:
    #     h = '0'+str(h)
    # else:
    #     h=str(h)
    # if len(str(m)) == 1:
    #     m = '0'+str(m)
    # else:
    #     m = str(m)
    #time_int = h+':'+m
    #print(time_int)
    t=time_int+':00'
    #print(t)
    

    weather=weather_details(lat_lon_source,date_t,t)
    

    #print(weather)

    #print(curr_time)

    #print(day)

    #print(time_int)

    time_codes= {'00-01':1,'01-02':2,'02-03':3,'03-04':4,'04-05':5,'05-06':6,'06-07':7,'07-08':8,'08-09':9,'09-10':10,'10-11':11,'11-12':12,'12-13':13,'13-14':14,'14-15':15,'15-16':16,'16-17':17,'17-18':18,'18-19':19,'19-20':20,'20-21':21,'21-22':22,'22-23':23,'23-24':24}
    time_it=ti(time_int)
    time_c=time_codes[time_it]
    #print("hey")
    #print(time_c)
    #print(type(time_c))

    ul_w,ll_w=fn_w(weather,curr_time)
    ul_ti,ll_ti=fn_ti(time_c,curr_time,day)

    #print(ul_w)
    #print(ll_w)
    #print("ok")
    #print(ul_ti)
    #print(ll_ti)
    #print(time_it)
    simulatorfn(ul_w,ll_w,ul_ti,ll_ti,time_it,day,weather,curr_time)
    csv_from_excel()

    r1 = MultipleLinearRegression(weather,day,time_c)

    r2 = PolynomialRegression(weather,day,time_c)

    r3 = decisionTreeRegression(weather,day,time_c)
    r4 = randomForestRegression(weather,day,time_c)
    #print(r1,r2,r3,r4)
    ensemble(r1,r2,r3,r4,time_int)

