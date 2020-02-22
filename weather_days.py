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
days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
#weather fetch using darksky api


time = {'00-01':1,'01-02':2,'02-03':3,'03-04':4,'04-05':5,'05-06':6,'06-07':7,'07-08':8,
            '08-09':9,'09-10':10,'10-11':11,'11-12':12,'12-13':13,'13-14':14,'14-15':15,
            '15-16':16,'16-17':17,'17-18':18,'18-19':19,'19-20':20,'20-21':21,'21-22':22,
            '22-23':23,'23-24':24}



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
