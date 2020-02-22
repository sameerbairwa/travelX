from helperFunctions import calc_day,datecheck,lat_lon_calc,curr_time_fetch,convertTime,datetime,ti,ensemble 
from dataPreprocessing import simulatorfn
from convert_xlsx_to_csv import csv_from_excel
from weather_days import fn_w,fn_ti


from algorithms import decisionTreeRegression,PolynomialRegression, randomForestRegression, MultipleLinearRegression

from dataPreprocessing import weather_details

if __name__ == '__main__':
    
    date=input("Enter date of travel (dd-mm-yyyy) : ")

    time_int=input("Enter Time : ")
    source=input("Enter Source: ")
    dest=input("Enter Destination: ")
    #date='20-12-2018'
    #time_int='10:12'
 
    #source='Jaypee Institute of information technology, Noida, Sector-62'
    #dest='jaipur'
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

