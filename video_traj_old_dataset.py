import cv2
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime, timedelta


import img_processing as my_img_proc

import visualization as vis

kinect_max_distance=0



def get_coordinate_points(occurance):

    xs = map(lambda line: int(float(line.split(' ')[2])),occurance)
    ys = map(lambda line: int(float(line.split(' ')[3])),occurance)
    zs = map(lambda line: float(line.split(' ')[4]),occurance)
    ids =map(lambda line: str(line.split(' ')[1]),occurance)

    list_points = []
    list_points_append = list_points.append

    map(lambda c: list_points_append((xs[c],ys[c])),xrange(0,len(xs)))

    #apply filter to cancel noise
    x_f,y_f =my_img_proc.median_filter(list_points)

    return x_f,y_f,zs,ids


def histograms_of_oriented_trajectories(list_poly,time_slices):
    hot_all_data_matrix = []
    hot_all_data_matrix_append = hot_all_data_matrix.append


    for i in xrange(0,len(time_slices)):
        ##Checking the start time of every time slice
        if(len(time_slices[i])>1):
            print 'start time: %s' %str(time_slices[i][0].split(' ')[8])
        else:
            print 'no data in this time slice'

        #get x,y,z of every traj point after smoothing process
        x_filtered,y_filtered,zs,ids = get_coordinate_points(time_slices[i])

        #initialize histogram of oriented tracklets
        hot_matrix = []

        for p in xrange(0,len(list_poly)):
            tracklet_in_cube_f = []
            tracklet_in_cube_c = []
            tracklet_in_cube_append_f = tracklet_in_cube_f.append
            tracklet_in_cube_append_c = tracklet_in_cube_c.append

            for ci in xrange(0,len(x_filtered)):
                #2d polygon
                if list_poly[p].contains_point((int(x_filtered[ci]),int(y_filtered[ci]))):
                    ## 3d cube close to the camera
                    if zs[ci] < (kinect_max_distance/2):

                        tracklet_in_cube_append_c([x_filtered[ci],y_filtered[ci],ids[ci]])

                    else: ##3d cube far from the camera

                        tracklet_in_cube_append_f([x_filtered[ci],y_filtered[ci],ids[ci]])


            for three_d_poly in [tracklet_in_cube_c,tracklet_in_cube_f]:
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


        hot_all_data_matrix_append(hot_matrix)




    ## normalize the final matrix
    normalized_finalMatrix = np.array(normalize(np.array(hot_all_data_matrix),norm='l2'))
    #print np.array(hot_all_data_matrix).shape

    ##add extra bin with hours
    hs = np.zeros((len(time_slices),1))


    for i,t in enumerate(time_slices):
        if len(t) > 1:
            hs[i] = int(t[0].split(' ')[8].split(':')[0])
        else:
            hs[i] = hs[i-1]


    normalized_finalMatrix = np.hstack((normalized_finalMatrix,hs))

    return normalized_finalMatrix


def occupancy_histograms_in_time_interval(my_room, list_poly, time_slices):
    # #get number of patches
    slice_col = my_img_proc.get_slice_cols()
    slice_row = my_img_proc.get_slice_rows()
    slice_depth = my_img_proc.get_slice_depth()

    my_data_temp = []
    my_data_temp_append = my_data_temp.append

    for i in xrange(0,len(time_slices)):

       ##Checking the start time of every time slice
        if(len(time_slices[i])>1):
            print 'start time: %s' %str(time_slices[i][0].split(' ')[8])
        else:
            print 'no data in this time slice'


        ## counter for every id should be empty
        track_points_counter = np.zeros((slice_col*slice_row*slice_depth))

        ##get x,y,z of every traj point after smoothing process
        x_filtered,y_filtered,zs,ids = get_coordinate_points(time_slices[i])

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

    ##add extra bin with hours
    hs = np.zeros((len(time_slices),1))


    for i,t in enumerate(time_slices):
        if len(t) > 1:
            hs[i] = int(t[0].split(' ')[8].split(':')[0])
        else:
            hs[i] = hs[i-1]

    normalized_finalMatrix = np.hstack((normalized_finalMatrix,hs))

    return normalized_finalMatrix


def org_OLDdata_timeIntervals(file):
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


def feature_extraction_video_traj(file_traj):
    print 'old dataset'

    ##visulaization apathy over week 19_4-29_4
    # motion_week = [12.038,9.022,7.974,9.9650,2.113,4.4285,5.7845]
    # slight_motion_week = [27.856,22.571,27.846,31.002,13.4013,10.6954,28.1096]
    # sedentary_week = [29.236,36.7410,35.1045,53.6780,35.505,43.7546,57.1622]
    #
    # vis.bar_plot_motion_in_region_over_long_time(motion_week)


    ##divide image into patches(polygons) and get the positions of each one
    my_room = np.zeros((480,640),dtype=np.uint8)
    list_poly = my_img_proc.divide_image(my_room)



    ##--------------Pre-Processing----------------##
    content,skeleton_data_in_time_slices = org_OLDdata_timeIntervals(file_traj)




    #occupancy_histograms = occupancy_histograms_in_time_interval(my_room, list_poly, skeleton_data_in_time_slices)
    occupancy_histograms = 1

    ## create Histograms of Oriented Tracks
    HOT_data = histograms_of_oriented_trajectories(list_poly,skeleton_data_in_time_slices)
    #HOT_data = 1

    vis.bar_plot_motion_over_time(HOT_data)
    #vis.pie_plot_motion_day(HOT_data)





    #return [occupancy_histograms,HOT_data]


if  __name__ == '__main__':
    feature_extraction_video_traj('C:/Users/dario.dotti/Documents/tracking_points/tracking_data_kinect2/29_4.txt')