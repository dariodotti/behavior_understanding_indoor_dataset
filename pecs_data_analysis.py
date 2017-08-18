import cv2
import numpy as np
from sklearn.preprocessing import normalize
from datetime import datetime, timedelta
from collections import Counter

import data_organizer

import video_traj
import img_processing



kinect_max_distance= 4.5
kinect_min_distance=0.5
cube_size = (kinect_max_distance-kinect_min_distance)/3


def draw_joints_and_tracks(body_points, scene):
    color = (0, 0, 255)

    # draw line between joints
    thickness = 3
    line_color = (19, 19, 164)

    ##check patches are correct
    # for i_rect, rect in enumerate(scene_patches):
    #     cv2.rectangle(scene, (int(rect.vertices[1][0]), int(rect.vertices[1][1])),
    #                   (int(rect.vertices[3][0]), int(rect.vertices[3][1])), (0, 0, 0))
    #
    #     ## write number of patch on img
    #     cv2.putText(scene, str(i_rect), (int(rect.vertices[1][0]) + 10, int(rect.vertices[1][1]) + 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    for n_frame, traj_body_joints in enumerate(body_points):
        # n_frame = n_frame+1402

        # if n_frame < 4870:
        #     continue

        temp_img = scene.copy()

        # draw joints
        print n_frame

        # first position skipped cause there are other info stored
        try:
            # torso
            cv2.line(temp_img, (int(float(traj_body_joints[4, 0])), int(float(traj_body_joints[4, 1]))),
                     (int(float(traj_body_joints[3, 0])), int(float(traj_body_joints[3, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[3, 0])), int(float(traj_body_joints[3, 1]))),
                     (int(float(traj_body_joints[2, 0])), int(float(traj_body_joints[2, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[2, 0])), int(float(traj_body_joints[2, 1]))),
                     (int(float(traj_body_joints[1, 0])), int(float(traj_body_joints[1, 1]))), line_color, thickness)
            # shoulder
            cv2.line(temp_img, (int(float(traj_body_joints[21, 0])), int(float(traj_body_joints[21, 1]))),
                     (int(float(traj_body_joints[9, 0])), int(float(traj_body_joints[9, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[21, 0])), int(float(traj_body_joints[21, 1]))),
                     (int(float(traj_body_joints[5, 0])), int(float(traj_body_joints[5, 1]))), line_color, thickness)
            # hips
            cv2.line(temp_img, (int(float(traj_body_joints[1, 0])), int(float(traj_body_joints[1, 1]))),
                     (int(float(traj_body_joints[17, 0])), int(float(traj_body_joints[17, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[1, 0])), int(float(traj_body_joints[1, 1]))),
                     (int(float(traj_body_joints[13, 0])), int(float(traj_body_joints[13, 1]))), line_color, thickness)
            # right arm
            cv2.line(temp_img, (int(float(traj_body_joints[9, 0])), int(float(traj_body_joints[9, 1]))),
                     (int(float(traj_body_joints[10, 0])), int(float(traj_body_joints[10, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[10, 0])), int(float(traj_body_joints[10, 1]))),
                     (int(float(traj_body_joints[11, 0])), int(float(traj_body_joints[11, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[11, 0])), int(float(traj_body_joints[11, 1]))),
                     (int(float(traj_body_joints[12, 0])), int(float(traj_body_joints[12, 1]))), line_color, thickness)
            # left arm
            cv2.line(temp_img, (int(float(traj_body_joints[5, 0])), int(float(traj_body_joints[5, 1]))),
                     (int(float(traj_body_joints[6, 0])), int(float(traj_body_joints[6, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[6, 0])), int(float(traj_body_joints[6, 1]))),
                     (int(float(traj_body_joints[7, 0])), int(float(traj_body_joints[7, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[7, 0])), int(float(traj_body_joints[7, 1]))),
                     (int(float(traj_body_joints[8, 0])), int(float(traj_body_joints[8, 1]))), line_color, thickness)

            # left leg
            cv2.line(temp_img, (int(float(traj_body_joints[13, 0])), int(float(traj_body_joints[13, 1]))),
                     (int(float(traj_body_joints[14, 0])), int(float(traj_body_joints[14, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[14, 0])), int(float(traj_body_joints[14, 1]))),
                     (int(float(traj_body_joints[15, 0])), int(float(traj_body_joints[15, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[15, 0])), int(float(traj_body_joints[15, 1]))),
                     (int(float(traj_body_joints[16, 0])), int(float(traj_body_joints[16, 1]))), line_color, thickness)
            # right leg
            cv2.line(temp_img, (int(float(traj_body_joints[17, 0])), int(float(traj_body_joints[17, 1]))),
                     (int(float(traj_body_joints[18, 0])), int(float(traj_body_joints[18, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[18, 0])), int(float(traj_body_joints[18, 1]))),
                     (int(float(traj_body_joints[19, 0])), int(float(traj_body_joints[19, 1]))), line_color, thickness)
            cv2.line(temp_img, (int(float(traj_body_joints[19, 0])), int(float(traj_body_joints[19, 1]))),
                     (int(float(traj_body_joints[20, 0])), int(float(traj_body_joints[20, 1]))), line_color, thickness)

            if n_frame > 0:
                for i, joint in enumerate(traj_body_joints):
                    if i == 0:
                        continue
                    cv2.circle(temp_img, (int(float(joint[0])), int(float(joint[1]))), 2, color, -1)
                    if i == 3 and n_frame > 0:
                        ##draw trajectories
                        cv2.circle(scene, (int(float(joint[0])), int(float(joint[1]))), 2, color, -1)
                    else:
                        ##draw joint
                        cv2.circle(temp_img, (int(float(joint[0])), int(float(joint[1]))), 2, color, -1)

            cv2.imshow('lab', temp_img)
            cv2.waitKey(1)

        except:
            print 'traj coordinates not available'
            continue

def sort_skeletons(task_skeleton_data):

    ids = map(lambda line: line[0][2], task_skeleton_data)
    print 'skeleton id: ', Counter(ids).most_common()
    skeletons = []
    for counter_ids in Counter(ids).most_common():
        skeleton_id = task_skeleton_data
        main_id = counter_ids[0]

        new_joints_points = []
        for i_point, points in enumerate(skeleton_id):
            if points[0][2] == main_id:
                if len(new_joints_points) == 0: print points[0]
                new_joints_points.append(points)

        skeleton_id = new_joints_points

        skeletons.append(skeleton_id)

    skeleton_data_sorted = skeletons[1]
    data_organizer.save_matrix_pickle(skeleton_data_sorted,
        'C:/Users/dario.dotti/Documents/pecs_data_review/skeletons_confusion_behavior_08082017_test.txt')
    return skeleton_data_sorted


def org_data_in_timeIntervals(skeleton_data, timeInterval_slice):
    #get all time data from the list dropping the decimal
    content_time = map(lambda line: line[0,1].split(' ')[1].split('.')[0],skeleton_data)

    #date time library

    init_t = datetime.strptime(content_time[0],'%H:%M:%S') #+ ' ' + timeInterval_slice[3]
    end_t = datetime.strptime(content_time[len(content_time)-1],'%H:%M:%S')
    x = datetime.strptime('0:0:0','%H:%M:%S')
    tot_duration = (end_t-init_t)

    #decide the size of time slices
    # size_slice= tot_duration/12
    # hours, remainder = divmod(size_slice.seconds, 3600)
    # minutes, seconds = divmod(remainder, 60)
    hours = timeInterval_slice[0]
    minutes = timeInterval_slice[1]
    seconds = timeInterval_slice[2]

    my_time_slice = timedelta(hours=hours,minutes=minutes,seconds=seconds)

    print 'time slice selected: ' + str(my_time_slice)

    #initialize list
    time_slices = []
    time_slices_append = time_slices.append

    c = (end_t-my_time_slice)
    #get data in every timeslices
    while init_t < (end_t-my_time_slice):
        list_time_interval = []
        list_time_interval_append = list_time_interval.append

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

def histograms_of_oriented_trajectories(list_poly, time_slices):
    #print kinect_max_distance, kinect_min_distance

    hot_all_data_matrix = []
    hot_all_data_matrix_append = hot_all_data_matrix.append

    for i in xrange(0, len(time_slices)):
        ##Checking the start time of every time slice
        if (len(time_slices[i]) > 1):
            print 'start time: %s' % str(time_slices[i][0][0][1])
        else:
            print 'no data in this time slice'
            continue

        # get x,y,z of every traj point after smoothing process
        x_filtered, y_filtered, zs, ids = img_processing.get_coordinate_points(time_slices[i], joint_id=3)

        # initialize histogram of oriented tracklets
        hot_matrix = []

        for p in xrange(0, len(list_poly)):
            tracklet_in_cube_f = []
            tracklet_in_cube_c = []
            tracklet_in_cube_middle = []
            tracklet_in_cube_append_f = tracklet_in_cube_f.append
            tracklet_in_cube_append_c = tracklet_in_cube_c.append
            tracklet_in_cube_append_middle = tracklet_in_cube_middle.append

            for ci in xrange(0, len(x_filtered)):
                if np.isinf(x_filtered[ci]) or np.isinf(y_filtered[ci]): continue

                # 2d polygon
                if list_poly[p].contains_point((int(x_filtered[ci]), int(y_filtered[ci]))):
                    ## 3d cube close to the camera
                    if zs[ci] <= (kinect_min_distance + cube_size):

                        tracklet_in_cube_append_c([x_filtered[ci], y_filtered[ci], ids[ci]])


                    elif zs[ci] > (kinect_min_distance + cube_size) and zs[ci] < (
                        kinect_min_distance + (cube_size * 2)):  #
                        tracklet_in_cube_append_middle([x_filtered[ci], y_filtered[ci], ids[ci]])

                    elif zs[ci] >= kinect_min_distance + (cube_size * 2):  ##3d cube far from the camera
                        tracklet_in_cube_append_f([x_filtered[ci], y_filtered[ci], ids[ci]])

            print len(tracklet_in_cube_c), len(tracklet_in_cube_middle), len(tracklet_in_cube_f)

            for three_d_poly in [tracklet_in_cube_c, tracklet_in_cube_middle, tracklet_in_cube_f]:
                if len(three_d_poly) > 0:

                    ## for tracklet in cuboids compute HOT following paper
                    hot_single_poly = img_processing.histogram_oriented_tracklets(three_d_poly)

                    ## compute hot+curvature
                    # hot_single_poly = my_img_proc.histogram_oriented_tracklets_plus_curvature(three_d_poly)

                else:
                    hot_single_poly = np.zeros((24))

                ##add to general matrix
                if len(hot_matrix) > 0:
                    hot_matrix = np.hstack((hot_matrix, hot_single_poly))
                else:
                    hot_matrix = hot_single_poly

        hot_all_data_matrix_append(hot_matrix)

    ## normalize the final matrix
    normalized_finalMatrix = np.array(normalize(np.array(hot_all_data_matrix), norm='l2'))

    ##add extra bin containing time


    ##return patinet id
    patient_id = ids[0]

    print 'HOT final matrix size: ', normalized_finalMatrix.shape

    return normalized_finalMatrix, patient_id

def main_pecs_data():
    ##get raw data for displaying
    task_skeleton_data = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/pecs_data_review/skeletons_repetitive_behavior_08082017.txt')
    ##if data contains multiple skeletons here I sort them cronologically
    sort_skeletons(task_skeleton_data)



    my_room = np.zeros((424,512,3),dtype=np.uint8)
    my_room += 255
    list_poly = img_processing.divide_image(my_room)

    #draw_joints_and_tracks(task_skeleton_data,my_room)
    #return 0

    skeleton_data_in_time_slices = org_data_in_timeIntervals(task_skeleton_data,[0,0,2])

    HOT_data, patient_ID = histograms_of_oriented_trajectories(list_poly, skeleton_data_in_time_slices)
    data_organizer.save_matrix_pickle(HOT_data,
        'C:/Users/dario.dotti/Documents/pecs_data_review/HOT_repetitive_behavior_08082017.txt')



if __name__ == '__main__':
    main_pecs_data()