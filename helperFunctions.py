from libraries import datetime, urllib2, json, calendar

# This function check that date is computable or not, "out of range "
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

def calc_day(date):
    w=datetime.datetime.strptime(date, '%d-%m-%Y')
    day=calendar.day_name[w.weekday()]

    return day

def lat_lon_calc(addr):
    url="https://maps.googleapis.com/maps/api/geocode/json?address="+addr+"&key=AIzaSyC8Sa49uFlX3T0UDhwxU2L6Csvl1g1ihq4"
    #print(url)
    json_file=urllib2.urlopen(url)
    data=json.load(json_file)
    lat_lng=[]
    
    lat_lng.append(data['results'][0]['geometry']['location']['lat'])
    lat_lng.append(data['results'][0]['geometry']['location']['lng'])

    return lat_lng

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
   