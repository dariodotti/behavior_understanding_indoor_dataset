from lxml import etree
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime, timedelta
import cPickle

__max_depth_value = 0
__min_depth_value = 0


def xml_parser(path_to_file):

    joint_points = []
    for event, element in etree.iterparse(path_to_file, tag='tracksInfo'):
        #print element.attrib['frameID']
        frame_body_joints = np.zeros((21,3))
        for i,child in enumerate(element):
            ##store all the body joints
            if child.attrib['x'] == '-1.#INF' or child.attrib['y'] == '-1.#INF':
                continue

            frame_body_joints[i,0]=child.attrib['x']
            frame_body_joints[i,1]=child.attrib['y']
            frame_body_joints[i,2]=child.attrib['z']
            #store only one joint
            # if child.tag == 'head':
            #     if child.attrib['x'] == '-1.#INF' and child.attrib['y'] == '-1.#INF':
            #         continue
            #     joint_points.append([child.attrib['x'],child.attrib['y'],child.attrib['z'],element.attrib['trackingID']])
        joint_points.append(frame_body_joints)

        element.clear()

    return joint_points


def read_all_data(joint_points):
    #organize the data with info needed

    points_array = np.zeros((len(joint_points),4))

    for i,p in enumerate(joint_points):

        #reconversion
        #x = int((float(p[0])*512)/768)
        #y = int((float(p[1])*424)/689)

        points_array[i,0]=int(float(p[0]))
        points_array[i,1]=int(float(p[1]))
        points_array[i,2]=float(p[2])
        #tracking id
        points_array[i,3]=np.uint64(p[3])

    #set max min depth value
    global __max_depth_value
    global __min_depth_value

    __max_depth_value=  np.max(points_array[:,2])
    __min_depth_value = np.min(points_array[:,2])

    return points_array


def org_data_timeIntervals(file):
    #print file

    #get all the data
    with open(file,'r')as f:
        file_content = f.read().split('\n')

    # save max distance to kinect
    zs = map(lambda line: float(line.split(' ')[4]),file_content)
    global kinect_max_distance
    kinect_max_distance =  np.max(zs)


    #Split file according to periods of time
    content_time = map(lambda line: line.split(' ')[8],file_content)

    time_interval_hours = 0
    time_interval_minutes = 10

    init_t = datetime.strptime(content_time[0],'%H:%M:%S')
    end_t = datetime.strptime(content_time[len(content_time)-1],'%H:%M:%S')
    x = datetime.strptime('0:0:0','%H:%M:%S')
    tot_duration = (end_t-init_t)


    counter = 0
    #print 'total duration of this file: %s' %str(tot_duration)


    time_slices = []
    time_slices_append = time_slices.append

    #get data in every timeslices
    while init_t < (end_t-timedelta(hours=time_interval_hours,minutes=time_interval_minutes)):
        list_time_interval = []
        list_time_interval_append = list_time_interval.append
        #print init_t
        for t in xrange(len(content_time)):

            if datetime.strptime(content_time[t],'%H:%M:%S')>= init_t and datetime.strptime(content_time[t],'%H:%M:%S') < init_t + timedelta(hours=time_interval_hours,minutes=time_interval_minutes):
                list_time_interval_append(file_content[t])

            if datetime.strptime(content_time[t],'%H:%M:%S') > init_t + timedelta(hours=time_interval_hours,minutes=time_interval_minutes):
                break
        #print len(list_time_interval)

        ##save time interval without distinction of part of the day
        time_slices_append(list_time_interval)

        init_t= init_t+timedelta(hours=time_interval_hours,minutes=time_interval_minutes)


    return file_content,time_slices


def thread_function(k,data,ids):
    print k
    temp_track = []
    temp_track_append= temp_track.append
    map(lambda i: temp_track_append(data[i]) if ids[i] == k else False,xrange(len(data)))

    return temp_track


def read_data_tracklets(data,multiThread):
    ids =map(lambda x: x[3],data)

    keys = list(set(ids))
    #data is sorted only if multithread isnt used
    keys = sorted(keys,key=lambda x: x)
    print len(keys)

    ####MULTI-THREAD VERSION######
    if multiThread:
        cores = 6
        pool = ThreadPool(cores)
        print 'n cores: '+str(cores)

        tracklets = pool.map(lambda k: thread_function(k,data,ids), keys)

        #close the pool and wait for the work to finish
        pool.close()
        pool.join()
    ###########################
    else:
        # keys= keys[:500]
        tracklets = []
        tracklets_append = tracklets.append

        for k in keys:
            #print k
            temp_track = []
            temp_track_append= temp_track.append

            map(lambda i: temp_track_append(data[i]) if ids[i] == k else False,xrange(len(data)))

            tracklets_append(temp_track)


    return tracklets


def get_max_min_depth_value():
    return [__max_depth_value,__min_depth_value]


def load_matrix_pickle(path):

    with open(path, 'rb') as handle:
        file = cPickle.load(handle)
    return file


def save_matrix_pickle(file,path):

    with open(path, 'wb') as handle:
        cPickle.dump(file,handle,protocol=2)

