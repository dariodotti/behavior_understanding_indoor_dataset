import visualization as vis
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import normalize

def file_parser(pathfile_binary):
    with open(pathfile_binary,'r') as f:
        file_content = f.read().split('\n')

    return file_content


def org_data_ID(pathfile_binary):
    content = file_parser(pathfile_binary)

    ##TODO: read the key from an external file that will contain the position of each sensor
    sensors_ID = {}
    key = ['entrance','cabinet_left','cabinet_right']

    #initialize dictionary
    for k in key:
        sensors_ID.setdefault(k,[])

    for i in xrange(1,len(content)):
        if content[i][:3] == 'a53':
            sensors_ID['entrance'].append(content[i])
        elif content[i][:3] == 'a50' or content[i][:3] == 'a56':
            sensors_ID['cabinet_right'].append(content[i])
        elif content[i][:3] == 'a51':
            sensors_ID['cabinet_left'].append(content[i])

    for k in sensors_ID.keys():
        if len(sensors_ID[k])%2 != 0:
            print k+' contains odd number of events: '+ str(len(sensors_ID[k]))

    return sensors_ID


def org_data_different_tasks(sensors_ID,file_AS):

    entrance_door = sensors_ID['entrance']

    entrance_time = []
    for event_door in entrance_door:
        e = event_door.split(' ')

        if e[0][9:11] == 'ON':
            t = e[2].split('-')
            entrance_time.append(datetime.strptime((t[0]+':'+t[1]+':'+t[2]),'%H:%M:%S'))

    #initialize list
    time_slices = []
    time_slices_append = time_slices.append

    ##get sensors and delete the entrance activations
    all_sensors = file_parser(file_AS)
    sensor= []
    for i,s in enumerate(all_sensors):
        if s[:3] != 'a53' and s[:3] != 'Sen':
            sensor.append(s)


    for i in range(0,len(entrance_time),2):
        temp_time_slice = []

        for s in range(0,len(sensor)):
            #print sensor[s].split(' ')
            t = sensor[s].split(' ')[2].split('-')

            if datetime.strptime((t[0]+':'+t[1]+':'+t[2]),'%H:%M:%S')> entrance_time[i] \
                and datetime.strptime((t[0]+':'+t[1]+':'+t[2]),'%H:%M:%S') < entrance_time[i+1]:

                temp_time_slice.append(sensor[s])

        time_slices_append(temp_time_slice)

    return time_slices

def night_motion():
    print 'night motion'


def nr_visit_bathroom(sensors_ID):
    object_events=sensors_ID['cabinet_left']
    print len(object_events)
    if len(object_events)%2 != 0:
        print 'odd number of events: '+ str(len(object_events))



    vis.plot_ambient_sensor_over_time(object_events)

    # #iter 2 elements each time to check if all th epairs are ON-OFF
    # for i in xrange(0,len(object_events)-1,2):
    #     #if they are not skip one
    #     if object_events[i][9:11] == object_events[i+1][9:11]:
    #         i+=1
    #
        # print object_events[i],object_events[i+1]



def feature_extraction_as(file_AS):
    #file_AS = 'C:/Users/dario.dotti/Documents/pilot_abnormal_behavior_indoor/binary/18-10-16_sensors_subject4.txt'

    sensors_ID = org_data_ID(file_AS)

    time_slices_in_tasks = org_data_different_tasks(sensors_ID,file_AS)


    activation_matrix= []
    activ_non_normalized = []

    for n_task,t in enumerate(time_slices_in_tasks):

        activation_in_time_slice = np.zeros((len(t),3))

        for i,e in enumerate(t):

            sensor_id =  e.split(' ')[0][:3]

            if sensor_id =='a50':
                activation_in_time_slice [i][0] +=1
            elif sensor_id =='a51':
                activation_in_time_slice [i][1] +=1
            elif sensor_id =='a56':
                activation_in_time_slice [i][2] +=1

        #print n_task

        #norm_matrix = normalize(activation_in_time_slice.sum(axis=0).reshape(1,-1),norm='l1',axis=1)[0]
        #activation_matrix.append(norm_matrix)


        ##transform 5 tasks in 6 tasks
        if n_task == 3:
            norm_matrix = normalize(activation_in_time_slice[:6].sum(axis=0).reshape(1,-1),norm='l1',axis=1)[0]
            activation_matrix.append(norm_matrix)
            norm_matrix = normalize(activation_in_time_slice[6:].sum(axis=0).reshape(1,-1),norm='l1',axis=1)[0]
            activation_matrix.append(norm_matrix)
        else:
            norm_matrix = normalize(activation_in_time_slice.sum(axis=0).reshape(1,-1),norm='l1',axis=1)[0]
            activation_matrix.append(norm_matrix)

        # if len(activ_non_normalized)>0:
        #     activ_non_normalized = np.vstack((activ_non_normalized,activation_in_time_slice.sum(axis=0).reshape(1,-1)))
        # else:
        #     activ_non_normalized = activation_in_time_slice.sum(axis=0).reshape(1,-1)

    #vis.plot_ambient_sensor_activation(activ_non_normalized)
    #vis.bar_plot_ambient_sensor_more_days(activ_non_normalized)



    return activation_matrix