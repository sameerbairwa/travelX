from libraries import *
from helperFunctions import ti
from weather_days import days, weather, time, fn_ti


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