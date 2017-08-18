import cv2
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime, timedelta
from lxml import etree

import img_processing as my_img_proc

import visualization as vis
import ambient_sensors
import data_organizer as data_org

kinect_max_distance=0
subjectID = ''
scene = np.zeros((414,512,3),dtype=np.uint8)
scene += 255

def draw_joints_and_tracks(body_points,list_poly):

    ##draw slices
    # for p in range(0,len(list_poly)):
    #    cv2.rectangle(scene,(int(list_poly[p].vertices[1][0]),int(list_poly[p].vertices[1][1])),\
    #                (int(list_poly[p].vertices[3][0]),int(list_poly[p].vertices[3][1])),0,1)

    for n_frame,traj_body_joints in enumerate(body_points):
        #n_frame = n_frame+1402

        # if n_frame < 4870:
        #     continue

        temp_ing = scene.copy()
        #draw joints
        print n_frame
        if n_frame > 250:
            color = (0,0,255)
        else:
            color = (0,0,255)

        #draw line between joints
        thickness = 3
        line_color = (19,19,164)
        #first position skipped cause there are other info stored
        #torso
        cv2.line(temp_ing,(int(float(traj_body_joints[1,0])),int(float(traj_body_joints[1,1]))),(int(float(traj_body_joints[2,0])),int(float(traj_body_joints[2,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[2,0])),int(float(traj_body_joints[2,1]))),(int(float(traj_body_joints[3,0])),int(float(traj_body_joints[3,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[3,0])),int(float(traj_body_joints[3,1]))),(int(float(traj_body_joints[4,0])),int(float(traj_body_joints[4,1]))),line_color,thickness)
        #shoulder
        cv2.line(temp_ing,(int(float(traj_body_joints[5,0])),int(float(traj_body_joints[5,1]))),(int(float(traj_body_joints[6,0])),int(float(traj_body_joints[6,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[5,0])),int(float(traj_body_joints[5,1]))),(int(float(traj_body_joints[10,0])),int(float(traj_body_joints[10,1]))),line_color,thickness)
        #hips
        cv2.line(temp_ing,(int(float(traj_body_joints[4,0])),int(float(traj_body_joints[4,1]))),(int(float(traj_body_joints[14,0])),int(float(traj_body_joints[14,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[4,0])),int(float(traj_body_joints[4,1]))),(int(float(traj_body_joints[18,0])),int(float(traj_body_joints[18,1]))),line_color,thickness)
        #right arm
        cv2.line(temp_ing,(int(float(traj_body_joints[6,0])),int(float(traj_body_joints[6,1]))),(int(float(traj_body_joints[7,0])),int(float(traj_body_joints[7,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[7,0])),int(float(traj_body_joints[7,1]))),(int(float(traj_body_joints[8,0])),int(float(traj_body_joints[8,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[8,0])),int(float(traj_body_joints[8,1]))),(int(float(traj_body_joints[9,0])),int(float(traj_body_joints[9,1]))),line_color,thickness)
        #left arm
        cv2.line(temp_ing,(int(float(traj_body_joints[10,0])),int(float(traj_body_joints[10,1]))),(int(float(traj_body_joints[11,0])),int(float(traj_body_joints[11,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[11,0])),int(float(traj_body_joints[11,1]))),(int(float(traj_body_joints[12,0])),int(float(traj_body_joints[12,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[12,0])),int(float(traj_body_joints[12,1]))),(int(float(traj_body_joints[13,0])),int(float(traj_body_joints[13,1]))),line_color,thickness)

        #right leg
        cv2.line(temp_ing,(int(float(traj_body_joints[14,0])),int(float(traj_body_joints[14,1]))),(int(float(traj_body_joints[15,0])),int(float(traj_body_joints[15,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[15,0])),int(float(traj_body_joints[15,1]))),(int(float(traj_body_joints[16,0])),int(float(traj_body_joints[16,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[16,0])),int(float(traj_body_joints[16,1]))),(int(float(traj_body_joints[17,0])),int(float(traj_body_joints[17,1]))),line_color,thickness)
        #left leg
        cv2.line(temp_ing,(int(float(traj_body_joints[18,0])),int(float(traj_body_joints[18,1]))),(int(float(traj_body_joints[19,0])),int(float(traj_body_joints[19,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[19,0])),int(float(traj_body_joints[19,1]))),(int(float(traj_body_joints[20,0])),int(float(traj_body_joints[20,1]))),line_color,thickness)
        cv2.line(temp_ing,(int(float(traj_body_joints[20,0])),int(float(traj_body_joints[20,1]))),(int(float(traj_body_joints[21,0])),int(float(traj_body_joints[21,1]))),line_color,thickness)

        if n_frame > 0:
           for i,joint in enumerate(traj_body_joints):
               if i ==0:
                   continue
               cv2.circle(temp_ing,(int(float(joint[0])),int(float(joint[1]))),2,color,-1)
               if i == 3 and n_frame>0:
                   ##draw trajectories
                   cv2.circle(scene,(int(float(joint[0])),int(float(joint[1]))),2,color,-1)
               else:
                   ##draw joint
                   cv2.circle(temp_ing,(int(float(joint[0])),int(float(joint[1]))),2,color,-1)

        if n_frame < 0:
            cv2.imshow('lab',temp_ing)
            cv2.waitKey(0)
        else:
            cv2.imshow('lab',temp_ing)
            cv2.waitKey(1)


def xml_parser(path_to_file):

    joint_points = []
    for event, element in etree.iterparse(path_to_file, tag='tracksInfo'):
        #print element.attrib['frameID']
        frame_body_joints = np.zeros((22,3),dtype='S30')


        #save in the first row the frame,time,trackID
        frame_body_joints[0,0]= element.attrib['frameID']
        frame_body_joints[0,1]= element.attrib['time']
        frame_body_joints[0,2]= element.attrib['trackingID']

        i=1
        for child in element:
            ##store all the body joints
            ##TODO: find a better way to correct this error
            if child.attrib['x'] == '-1.#INF' or child.attrib['y'] == '-1.#INF':
                frame_body_joints[i,0]= 0.
                frame_body_joints[i,1]= 0.
                frame_body_joints[i,2]= 0.
            else:
                frame_body_joints[i,0]=child.attrib['x']
                frame_body_joints[i,1]=child.attrib['y']
                frame_body_joints[i,2]=child.attrib['z']
            #store only one joint
            # if child.tag == 'head':
            #     if child.attrib['x'] == '-1.#INF' and child.attrib['y'] == '-1.#INF':
            #         continue
            #     joint_points.append([child.attrib['x'],child.attrib['y'],child.attrib['z'],element.attrib['trackingID']])
            i+=1
        joint_points.append(frame_body_joints)

        element.clear()

    return joint_points


def org_xml_data_timeIntervals(skeleton_data):

    #get all time data from the list
    content_time = map(lambda line: line[0,1].split(' ')[3] ,skeleton_data)

    #date time library

    init_t = datetime.strptime(content_time[0],'%H:%M:%S')
    end_t = datetime.strptime(content_time[len(content_time)-1],'%H:%M:%S')
    x = datetime.strptime('0:0:0','%H:%M:%S')
    tot_duration = (end_t-init_t)

    #decide the size of time slices
    # size_slice= tot_duration/25
    # hours, remainder = divmod(size_slice.seconds, 3600)
    # minutes, seconds = divmod(remainder, 60)
    hours = 0
    minutes = 0
    seconds = 2
    #print hours,minutes,seconds
    my_time_slice = timedelta(hours=hours,minutes=minutes,seconds=seconds)

    print 'time slice selected: ' + str(my_time_slice)

    #initialize list
    time_slices = []
    time_slices_append = time_slices.append

    #get data in every timeslices
    while init_t < (end_t-my_time_slice):

        list_time_interval = []
        list_time_interval_append = list_time_interval.append
        #print init_t

        for t in xrange(len(content_time)):

            if datetime.strptime(content_time[t],'%H:%M:%S')>= init_t and datetime.strptime(content_time[t],'%H:%M:%S') < init_t + my_time_slice:
                list_time_interval_append(skeleton_data[t])


            if datetime.strptime(content_time[t],'%H:%M:%S') > init_t + my_time_slice:
                break
        #print len(list_time_interval)

        ##save time interval without distinction of part of the day
        time_slices_append(list_time_interval)

        init_t= init_t+my_time_slice


    return time_slices


def set_subject(subject):
    global subjectID
    subjectID = subject


def org_data_different_tasks(skeleton_data):

    file_AS = 'C:/Users/dario.dotti/Documents/Datasets/my_dataset/wandering_dataset_um/binary/18-10-16_sensors_'+ subjectID +'.txt'

    sensors_ID = ambient_sensors.org_data_ID(file_AS)
    entrance_door = sensors_ID['entrance']

    entrance_time = []
    for event_door in entrance_door:
        e = event_door.split(' ')

        if e[0][9:11] == 'ON':
            t = e[2].split('-')
            entrance_time.append(datetime.strptime((t[0]+':'+t[1]+':'+t[2]),'%H:%M:%S'))


    #get all time data from the list
    skeleton_time_info = map(lambda line: line[0,1].split(' ')[3] ,skeleton_data)

    #initialize list
    time_slices = []
    time_slices_append = time_slices.append

    for i in range(0,len(entrance_time),2):
        temp_time_slice = []
        # print entrance_time[i],entrance_time[i+1]
        for s in range(0,len(skeleton_data)):

            if datetime.strptime(skeleton_time_info[s],'%H:%M:%S') > entrance_time[i] \
                    and datetime.strptime(skeleton_time_info[s],'%H:%M:%S') < entrance_time[i+1]:

                temp_time_slice.append(skeleton_data[s])

        time_slices_append(temp_time_slice)


    return time_slices


def org_data_timeIntervals_inside_tasks(skeleton_data_in_tasks):

    data_task_and_time_slices = []

    #decide the size of time slices
    # size_slice= tot_duration/25
    # hours, remainder = divmod(size_slice.seconds, 3600)
    # minutes, seconds = divmod(remainder, 60)
    hours = 0
    minutes = 0
    seconds = 2
    #print hours,minutes,seconds
    my_time_slice = timedelta(hours=hours,minutes=minutes,seconds=seconds)

    print 'time slice selected: ' + str(my_time_slice)

    for task in range(0,len(skeleton_data_in_tasks)):
        #get all time data from the list
        content_time = map(lambda line: line[0,1].split(' ')[3] ,skeleton_data_in_tasks[task])

        #date time library

        init_t = datetime.strptime(content_time[0],'%H:%M:%S')
        end_t = datetime.strptime(content_time[len(content_time)-1],'%H:%M:%S')
        x = datetime.strptime('0:0:0','%H:%M:%S')
        tot_duration = (end_t-init_t)

        #initialize list
        time_slices = []
        time_slices_append = time_slices.append

        #get data in every timeslices
        while init_t < (end_t-my_time_slice):

            list_time_interval = []
            list_time_interval_append = list_time_interval.append
            #print init_t

            for t in xrange(len(content_time)):

                if datetime.strptime(content_time[t],'%H:%M:%S')>= init_t and datetime.strptime(content_time[t],'%H:%M:%S') < init_t + my_time_slice:
                    list_time_interval_append(skeleton_data_in_tasks[task][t])


                if datetime.strptime(content_time[t],'%H:%M:%S') > init_t + my_time_slice:
                    break
            #print len(list_time_interval)

            ##save time interval without distinction of part of the day
            time_slices_append(list_time_interval)

            init_t= init_t+my_time_slice

        data_task_and_time_slices.append(time_slices)


    return data_task_and_time_slices


# def get_coordinate_points(time_slice,joint_id):
#
#     #get all the coordinate points of head joint
#     list_points = []
#     list_points_append = list_points.append
#
#     #get x,y,z,id
#     map(lambda line: list_points_append([line[joint_id][0],line[joint_id][1]]),time_slice)
#     zs = map(lambda line: float(line[joint_id][2]),time_slice)
#     ids =map(lambda line: np.int64(line[0][2]),time_slice)
#
#     #apply filter to cancel noise
#     x_f,y_f =my_img_proc.median_filter(list_points)
#
#     return x_f,y_f,zs,ids


def occupancy_histograms_in_time_interval(my_room, list_poly, time_slices):
    # #get number of patches
    slice_col = my_img_proc.get_slice_cols()
    slice_row = my_img_proc.get_slice_rows()
    slice_depth = my_img_proc.get_slice_depth()

    my_data_temp = []
    my_data_temp_append = my_data_temp.append

    for i in xrange(0,len(time_slices)):
        ## Checking the start time of every time slice
        if(len(time_slices[i])>1):
            print 'start time: %s' %time_slices[i][0][0][1].split(' ')[3]
        else:
            print 'no data in this time slice'


        ## counter for every id should be empty
        track_points_counter = np.zeros((slice_col*slice_row*slice_depth))

        ##get x,y,z of every traj point after smoothing process
        x_filtered,y_filtered,zs,ids = get_coordinate_points(time_slices[i],joint_id=1)

        ## display traj on img
        #temp_img = copy.copy(my_room)
        #my_img_proc.display_trajectories(temp_img, list_poly, x_filtered, y_filtered)

        ## count the occurances of filtered point x,y in every patches
        for p in xrange(0,len(list_poly)):

            for ci in xrange(0,len(x_filtered)):
                ## 2d polygon
                if list_poly[p].contains_point((int(x_filtered[ci]),int(y_filtered[ci]))):
                    ## 3d cube close to the camera
                    if zs[ci] < (kinect_max_distance/2):

                        track_points_counter[p*2] = track_points_counter[p*2] + 1
                        continue
                    else: ## 3d cube far from the camera

                        track_points_counter[(p*2)+1] = track_points_counter[(p*2)+1] + 1
                        continue


        ## save the data of every group in the final matrix
        my_data_temp_append(track_points_counter)

    ## normalize the final matrix
    normalized_finalMatrix = np.array(normalize(np.array(my_data_temp),norm='l2'))
    print 'final matrix size:'
    print normalized_finalMatrix.shape

    return normalized_finalMatrix


def histograms_of_oriented_trajectories(list_poly,time_slices):

    hot_all_data_task_time_slices = []

    for i_task,task in enumerate(time_slices):
        #if i_task != 2: continue

        hot_all_data_matrix = []
        hot_all_data_matrix_append = hot_all_data_matrix.append

        print '###########task########### ',i_task
        for i in xrange(0,len(task)):
            ##Checking the start time of every time slice
            if(len(task[i])>1):
                print 'start time: %s' %task[i][0][0][1].split(' ')[3]
            else:
                print 'no data in this time slice'

                continue
            #get x,y,z of every traj point after smoothing process
            x_filtered,y_filtered,zs,ids = my_img_proc.get_coordinate_points(task[i],joint_id=1)#get all position of the head joint id =1

            #initialize histogram of oriented tracklets
            hot_matrix = []

            temp_img = scene.copy()

            for p in xrange(0,len(list_poly)):
                tracklet_in_cube_f = []
                tracklet_in_cube_c = []
                tracklet_in_cube_middle = []
                tracklet_in_cube_append_f = tracklet_in_cube_f.append
                tracklet_in_cube_append_c = tracklet_in_cube_c.append
                tracklet_in_cube_append_middle = tracklet_in_cube_middle.append

                for ci in xrange(0,len(x_filtered)):
                    #2d polygon
                    if list_poly[p].contains_point((int(x_filtered[ci]),int(y_filtered[ci]))):
                        ## 3d cube close to the camera
                        if zs[ci] < (kinect_max_distance-(1.433*2)):
                            #print 'close to kinect'
                            tracklet_in_cube_append_c([x_filtered[ci],y_filtered[ci],ids[ci]])

                            cv2.circle(temp_img,(int(x_filtered[ci]),int(y_filtered[ci])),2,(255,0,0),-1)

                        elif zs[ci] > (kinect_max_distance-(1.433*2)) and zs[ci] < (kinect_max_distance-1.433):
                            #print 'middle'
                            tracklet_in_cube_append_middle([x_filtered[ci],y_filtered[ci],ids[ci]])

                            cv2.circle(temp_img,(int(x_filtered[ci]),int(y_filtered[ci])),2,(0,255,0),-1)

                        elif zs[ci] > (kinect_max_distance-1.433): ##3d cube far from the camera
                            #print 'faraway to kinect'
                            tracklet_in_cube_append_f([x_filtered[ci],y_filtered[ci],ids[ci]])

                            cv2.circle(temp_img,(int(x_filtered[ci]),int(y_filtered[ci])),2,(0,0,255),-1)


                for three_d_poly in [tracklet_in_cube_c,tracklet_in_cube_middle,tracklet_in_cube_f]:

                    if len(three_d_poly)>0:

                        ## for tracklet in cuboids compute HOT following paper
                        hot_single_poly = my_img_proc.histogram_oriented_tracklets(three_d_poly)

                        ## compute hot+curvature
                        #hot_single_poly = my_img_proc.histogram_oriented_tracklets_plus_curvature(three_d_poly)

                    else:
                        hot_single_poly = np.zeros((24))

                    ##add to general matrix
                    if len(hot_matrix)>0:
                        hot_matrix = np.hstack((hot_matrix,hot_single_poly))
                    else:
                        hot_matrix = hot_single_poly

            #time = time_slices[i][0][0][1].split(' ')[3].split(':')
            #filename = 'C:/Users/dario.dotti/Documents/time_windows_HOT/'+subjectID+'_'+time[0]+'_'+time[1]+'_'+time[2]+'.jpg'
            #cv2.imwrite(filename,temp_img)

            ##Test cluster
            # load cluster data
            cluster_model = data_org.load_matrix_pickle(
                'C:/Users/dario.dotti/Documents/bow_experiment_data/cl_30_kmeans_model_2secWindow_newVersion.txt')
            keys_labels = data_org.load_matrix_pickle(
                'C:/Users/dario.dotti/Documents/bow_experiment_data/cluster_30_kmeans_word_newVersion.txt')

            similar_word = cluster_model.predict(np.array(hot_matrix).reshape(1, -1))
            print 's_w ',similar_word
            if similar_word[0] == 3:
                cv2.imshow('ciao',temp_img)
                cv2.waitKey(0)
                continue

            hot_all_data_matrix_append(hot_matrix)
            print len(hot_all_data_matrix)


        ## normalize the final matrix
        normalized_finalMatrix = np.array(normalize(np.array(hot_all_data_matrix),norm='l1'))

        hot_all_data_task_time_slices.append(normalized_finalMatrix)




    #print 'final matrix size:'
    #print np.array(normalized_finalMatrix).shape


    ##add extra bin with hours
    # hs = np.zeros((len(time_slices),1))
    #
    # for i,t in enumerate(time_slices):
    #
    #     if len(t) > 1:
    #         hs[i] = int(t[0][0][1].split(' ')[3].split(':')[0])
    #     else:
    #         hs[i] = hs[i-1]
    #
    #
    # normalized_finalMatrix = np.hstack((normalized_finalMatrix,hs))

    # print 'matrix with extra bin'
    # print np.array(hot_all_data_matrix).shape

    return hot_all_data_task_time_slices


def measure_joints_accuracy(skeleton_data):

    frame_step = 5

    mean_displcement_list = np.zeros((len(skeleton_data[0])-1,1))

    joint_distances = []
    joint_distances_append = joint_distances.append

    # for joint_id in xrange(1,len(skeleton_data[0])):
    #
    #     #euclidean distance between joint time[0] and joint time[framestep]
    #     eu_difference = map(lambda i: np.sqrt((int(float(skeleton_data[i+frame_step][joint_id,0]))- int(float(skeleton_data[i][joint_id,0])))**2 + \
    #         (int(float(skeleton_data[i+frame_step][joint_id,1])) - int(float(skeleton_data[i][joint_id,1])))**2) \
    #         if skeleton_data[i][joint_id,0] != 0. or skeleton_data[i+1][joint_id,0] != 0. else 0 \
    #            ,xrange(0,len(skeleton_data)-frame_step))
    #
    #     mean_displcement_list[joint_id-1] = np.sum(eu_difference)/len(eu_difference)
    #
    #     joint_distances_append(eu_difference)

    #print mean_displcement_list
##############
    #subject7_exit_entrance = [19676 ,16250,  1943]
    subject4_exit_entrance = [3867,6053,9053,11898,17584,25777]

    ##not optimized code but more understadable
    for joint_id in xrange(1,len(skeleton_data[0])):
        eu_difference = np.zeros((len(skeleton_data),1))
        for i in xrange(0,len(skeleton_data)-frame_step):
            if skeleton_data[i][joint_id,0] == 0. or skeleton_data[i+1][joint_id,0] == 0.:
                continue
            if i in subject4_exit_entrance:
                continue
            #euclidean distance between joint time[0] and joint time[framestep]
            eu_difference[i] = np.sqrt((int(float(skeleton_data[i+frame_step][joint_id,0]))- int(float(skeleton_data[i][joint_id,0])))**2 + \
            (int(float(skeleton_data[i+frame_step][joint_id,1])) - int(float(skeleton_data[i][joint_id,1])))**2)
        joint_distances_append(eu_difference)
        mean_displcement_list[joint_id-1] = np.sum(eu_difference)/len(eu_difference)

###############
    ##get filtered points
    joint_distances_filtered = []
    joint_distances_filtered_append = joint_distances_filtered.append
    for joint_id in xrange(1,len(skeleton_data[0])):

        ##store x,y for 1 joint each time over all frames
        list_points = []
        list_points_append = list_points.append
        for i in xrange(0,len(skeleton_data)):
            #if skeleton_data[i][joint_id,0] == 0.:
                #continue
            #if i in subject7_exit_entrance:
                #continue
            list_points_append((int(float(skeleton_data[i][joint_id,0])),int(float(skeleton_data[i][joint_id,1]))))

        ##apply filter
        x_f,y_f =my_img_proc.median_filter(list_points)

        eu_difference_filtered = np.zeros((len(skeleton_data),1))
        for i in xrange(0,len(x_f)-1):

            #if x_f[i+1] == 0. or x_f[i] == 0.:
                #continue
            if i in subject4_exit_entrance:
                continue

            eu_difference_filtered[i] = np.sqrt((x_f[i+1]-x_f[i])**2 + (y_f[i+1]-y_f[i])**2)
        joint_distances_filtered_append(eu_difference_filtered)
    # print mean_displcement_list

    ##get only the desired joint
    my_joint_raw = map(lambda x: x,joint_distances[:1][0])
    my_joint_filtered=map(lambda x: x,joint_distances_filtered[:1][0])

    #difference between raw and filtered features
    diff = map(lambda pair: pair[0]-pair[1] , zip(my_joint_raw,my_joint_filtered))

    #get frames where joint displacement over threshold
    threshold = 15

    frames_where_joint_displacement_over_threshold = []
    map(lambda (i,d): frames_where_joint_displacement_over_threshold.append(i) if d>threshold else False , enumerate(diff))
    print len(frames_where_joint_displacement_over_threshold)


    ##display mean distance of every joints between frames
    vis.plot_mean_joints_displacement(mean_displcement_list)

    ##display error each frame from selected joint
    vis.plot_single_joint_displacement_vs_filtered_points(my_joint_raw,my_joint_filtered)

    return frames_where_joint_displacement_over_threshold


def feature_extraction_video_traj(file_traj):

    ##divide image into patches(polygons) and get the positions of each one
    #my_room = np.zeros((414,512),dtype=np.uint8)
    global scene
    scene = cv2.imread('C:/Users/dario.dotti/Documents/Datasets/my_dataset/wandering_dataset_um/subject4_1834.jpg')

    list_poly = my_img_proc.divide_image(scene)

    ##check patches are correct
    for rect in list_poly:
        cv2.rectangle(scene, (int(rect.vertices[1][0]), int(rect.vertices[1][1])),
                      (int(rect.vertices[3][0]), int(rect.vertices[3][1])), (0, 0, 0))
    #
    # cv2.imshow('ciao',scene)
    # cv2.waitKey(0)


    ##--------------Pre-Processing----------------##
    skeleton_data = xml_parser(file_traj)

    ##reliability method
    #measure_joints_accuracy(skeleton_data)

    ##display joints
    #draw_joints_and_tracks(skeleton_data,list_poly)

    ##divide the data based on time info
    #skeleton_data_in_time_slices = org_xml_data_timeIntervals(skeleton_data)

    ##update depth values
    depth_v = []
    map(lambda x: depth_v.append(float(x[1,2])) ,skeleton_data)
    global kinect_max_distance
    kinect_max_distance = np.max(depth_v)
    #print kinect_max_distance

    ##divide the data based on task
    skeleton_data_in_time_slices = org_data_different_tasks(skeleton_data)

    ##divide data based time info per task
    skeleton_data_in_tasks_and_time_slices = org_data_timeIntervals_inside_tasks(skeleton_data_in_time_slices)

    ##--------------Feature Extraction-------------##
    print 'feature extraction'

    ## count traj points in each region and create hist
    #occupancy_histograms = occupancy_histograms_in_time_interval(my_room, list_poly, skeleton_data_in_time_slices)
    occupancy_histograms = 1

    ## create Histograms of Oriented Tracks
    HOT_data = histograms_of_oriented_trajectories(list_poly,skeleton_data_in_tasks_and_time_slices)

    #vis.bar_plot_motion_over_time(HOT_data)

    return [occupancy_histograms,HOT_data]

    #cluster_prediction = my_exp.main_experiments(HOT_data)