import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn import decomposition
from collections import Counter
from math import atan2,degrees
from sklearn.preprocessing import normalize

import img_processing
import data_organizer
import hierarchical_ae_learning_methods as h_ae




def plot_dist():
    ##plot horizontal distances
    # x_dist_ = np.array(dist_x)
    # plt.scatter(range(len(x_dist_)),x_dist_[:,0],marker='+',c='r')
    # plt.scatter(range(len(x_dist_)), x_dist_[:, 1], marker='^',c='g')
    # plt.scatter(range(len(x_dist_)), x_dist_[:, 2])
    # plt.show()

    ##plot vertical distances
    y_dist_ = np.array(dist_y)
    plt.scatter(range(len(y_dist_)), y_dist_[:, 0], marker='+', c='r')
    plt.scatter(range(len(y_dist_)), y_dist_[:, 1], marker='^', c='g')
    plt.show()


dist_x = []
dist_y = []

def get_dist_arms(shoulder_left_x, shoulder_left_y, shoulder_right_x, shoulder_right_y, elbow_left_x,
                         elbow_left_y, elbow_right_x, elbow_right_y, wrist_left_x,wrist_left_y, wrist_right_x, wrist_right_y):

    ### horizontal distance
    # dist_shoulder = np.sqrt(((shoulder_left_x[0] - shoulder_right_x[0]) ** 2) + ((shoulder_left_y[0] - shoulder_right_y[0]) ** 2))
    # dist_elbow = np.sqrt(((elbow_left_x[0] - elbow_right_x[0]) ** 2) + ((elbow_left_y[0] - elbow_right_y[0]) ** 2))
    # dist_wrist = np.sqrt(((wrist_left_x[0] - wrist_right_x[0]) ** 2) + ((wrist_left_y[0] - wrist_right_y[0]) ** 2))

    #print dist_shoulder,dist_elbow,dist_wrist

    #dist_x.append([dist_shoulder, dist_elbow, dist_wrist])

    ### vertical distances
    dist_shoulder_left = np.sqrt(
        ((shoulder_left_x[0] - wrist_left_x[0]) ** 2) + ((shoulder_left_y[0] - wrist_left_y[0]) ** 2))
    dist_shoulder_right = np.sqrt(
        ((shoulder_right_x[0] - wrist_right_x[0]) ** 2) + ((shoulder_right_y[0] - wrist_right_y[0]) ** 2))

    dist_y.append([dist_shoulder_left,dist_shoulder_right])


def draw_arms(temp_img,shoulder_left_x, shoulder_left_y, shoulder_right_x, shoulder_right_y, elbow_left_x,
                         elbow_left_y, elbow_right_x, elbow_right_y, wrist_left_x,wrist_left_y, wrist_right_x, wrist_right_y,center_x,center_y):


    ## draw arm right
    cv2.line(temp_img, (shoulder_right_x, shoulder_right_y),
             (elbow_right_x, elbow_right_y), (255, 0, 0), 2)
    cv2.line(temp_img, (elbow_right_x, elbow_right_y),
             (wrist_right_x, wrist_right_y), (255, 0, 0), 2)

    ## draw arm left
    cv2.line(temp_img, (shoulder_left_x, shoulder_left_y),
             (elbow_left_x, elbow_left_y), (255, 0, 0), 2)
    cv2.line(temp_img, (elbow_left_x, elbow_left_y),
             (wrist_left_x, wrist_left_y), (255, 0, 0), 2)

    ## draw center
    cv2.circle(temp_img, (center_x, center_y), 3, (0, 0, 255), -1)

    cv2.imshow('ciao', temp_img)
    cv2.waitKey(0)


def clockwise_slope(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def lines_intersect(A, B, C, D):
    return clockwise_slope(A, C, D) != clockwise_slope(B, C, D) and clockwise_slope(A, B, C) != clockwise_slope(A, B, D)



def extract_arms_pos(shoulder_left_x, shoulder_left_y, shoulder_right_x, shoulder_right_y, elbow_left_x,
                          elbow_left_y, elbow_right_x, elbow_right_y, wrist_left_x,wrist_left_y, wrist_right_x, wrist_right_y, head_x, head_y,spineBase_x,spineBase_y):

    imgs = []

    ##get 10 fps
    for i_coord in range(0,len(shoulder_left_x),int(len(shoulder_left_x)/15)):
        #print i_coord


        ### to remove noise  i check if the spine lines pass through the shoulder line ##

        if not lines_intersect((shoulder_left_x[i_coord], shoulder_left_y[i_coord]), (shoulder_right_x[i_coord], shoulder_right_y[i_coord]), \
                        (head_x[i_coord], head_y[i_coord]),
                        (spineBase_x[i_coord], spineBase_y[i_coord])):
            #print 'no intersection'
            continue


        ###Find center X ###
        highest_point_index = np.where([shoulder_right_x[i_coord], elbow_right_x[i_coord], wrist_right_x[i_coord], shoulder_left_x[i_coord], elbow_left_x[i_coord], wrist_left_x[i_coord]] \
            == np.min([shoulder_right_x[i_coord], elbow_right_x[i_coord], wrist_right_x[i_coord], shoulder_left_x[i_coord], elbow_left_x[i_coord], wrist_left_x[i_coord]]))[0][0]

        highest = [shoulder_right_x[i_coord], elbow_right_x[i_coord], wrist_right_x[i_coord], shoulder_left_x[i_coord], elbow_left_x[i_coord], wrist_left_x[i_coord]][highest_point_index]

        lowest_point_index = np.where([shoulder_right_x[i_coord], elbow_right_x[i_coord], wrist_right_x[i_coord], shoulder_left_x[i_coord], elbow_left_x[i_coord], wrist_left_x[i_coord]] \
            == np.max([shoulder_right_x[i_coord], elbow_right_x[i_coord], wrist_right_x[i_coord], shoulder_left_x[i_coord], elbow_left_x[i_coord], wrist_left_x[i_coord]]))[0][0]
        lowest = [shoulder_right_x[i_coord], elbow_right_x[i_coord], wrist_right_x[i_coord], shoulder_left_x[i_coord], elbow_left_x[i_coord], wrist_left_x[i_coord]][lowest_point_index]

        center_x = highest + int((lowest - highest) / 2)


        ####Find center Y ##
        highest_point_index = np.where([shoulder_right_y[i_coord],elbow_right_y[i_coord],wrist_right_y[i_coord],shoulder_left_y[i_coord],elbow_left_y[i_coord],wrist_left_y[i_coord]] \
                 == np.min([shoulder_right_y[i_coord],elbow_right_y[i_coord],wrist_right_y[i_coord],shoulder_left_y[i_coord],elbow_left_y[i_coord],wrist_left_y[i_coord]]))[0][0]
        highest = [shoulder_right_y[i_coord], elbow_right_y[i_coord], wrist_right_y[i_coord], shoulder_left_y[i_coord], elbow_left_y[i_coord], wrist_left_y[i_coord]][highest_point_index]

        lowest_point_index = np.where([shoulder_right_y[i_coord], elbow_right_y[i_coord], wrist_right_y[i_coord], shoulder_left_y[i_coord], elbow_left_y[i_coord], wrist_left_y[i_coord]] \
            == np.max([shoulder_right_y[i_coord], elbow_right_y[i_coord], wrist_right_y[i_coord], shoulder_left_y[i_coord], elbow_left_y[i_coord],
                       wrist_left_y[i_coord]]))[0][0]
        lowest = [shoulder_right_y[i_coord], elbow_right_y[i_coord], wrist_right_y[i_coord], shoulder_left_y[i_coord], elbow_left_y[i_coord], wrist_left_y[i_coord]][lowest_point_index]

        center_y = highest+int((lowest - highest) /2)


        ### Draw ##
        temp_img = np.zeros((424, 512, 3), dtype=np.uint8)
        temp_img += 255

        draw_arms(temp_img,shoulder_left_x[i_coord], shoulder_left_y[i_coord], shoulder_right_x[i_coord], shoulder_right_y[i_coord], elbow_left_x[i_coord],
                  elbow_left_y[i_coord], elbow_right_x[i_coord], elbow_right_y[i_coord], wrist_left_x[i_coord], wrist_left_y[i_coord], wrist_right_x[i_coord], wrist_right_y[i_coord],center_x,center_y)


        ##find difference between the current center and the new img center, use this difference to convert everything
        feature_img = np.zeros((120, 120))

        diff_x = abs(center_x - (feature_img.shape[1]/2))
        diff_y = abs(center_y - (feature_img.shape[0]/2))

        limb_pos_x = []
        for limb_x in [shoulder_right_x[i_coord], elbow_right_x[i_coord], wrist_right_x[i_coord], shoulder_left_x[i_coord], elbow_left_x[i_coord], wrist_left_x[i_coord], head_x[i_coord], spineBase_x[i_coord]]:
            limb_pos_x.append(int(limb_x-diff_x))

        limb_pos_y = []
        for limb_y in [shoulder_right_y[i_coord],elbow_right_y[i_coord],wrist_right_y[i_coord],shoulder_left_y[i_coord],elbow_left_y[i_coord],wrist_left_y[i_coord], head_y[i_coord], spineBase_y[i_coord]]:
            limb_pos_y.append(int(limb_y - diff_y))


        # feature_img = np.zeros((120, 130, 3), dtype=np.uint8)
        # feature_img+=255
        # draw_arms(feature_img,limb_pos_x[3],limb_pos_y[3],limb_pos_x[0],limb_pos_y[0],limb_pos_x[4],limb_pos_y[4],limb_pos_x[1],limb_pos_y[1],\
        #           limb_pos_x[5],limb_pos_y[5],limb_pos_x[2],limb_pos_y[2],(feature_img.shape[1]/2),(feature_img.shape[0]/2) )


        ##  draw on black img the arms with value 255
        for i_limb in [0,1,3,4]:
            points_on_line = h_ae.createLineIterator(np.array([limb_pos_x[i_limb],limb_pos_y[i_limb]])\
                    ,np.array([limb_pos_x[i_limb + 1],limb_pos_y[i_limb + 1]]),feature_img)

            for p in points_on_line:
                ##if we want to display on img

                feature_img[int(p[1]), int(p[0])] = 0.99

                if int(p[0])+2 < feature_img.shape[1] and int(p[1])+2 < feature_img.shape[0]:
                    ##right
                    feature_img[int(p[1]) + 1, int(p[0])] = 0.99
                    feature_img[int(p[1]) + 2, int(p[0])] = 0.99
                    ##left
                    feature_img[int(p[1]) - 1, int(p[0])] = 0.99
                    feature_img[int(p[1]) - 2, int(p[0])] = 0.99
                    ##up
                    feature_img[int(p[1]), int(p[0]) - 1] = 0.99
                    feature_img[int(p[1]), int(p[0]) - 2] = 0.99
                    ##down
                    feature_img[int(p[1]), int(p[0]) + 1] = 0.99
                    feature_img[int(p[1]), int(p[0]) + 2] = 0.99


        ### add shoulders
        points_on_line = h_ae.createLineIterator(np.array([limb_pos_x[3], limb_pos_y[3]]) \
                                                 , np.array([limb_pos_x[0], limb_pos_y[0]]), feature_img)
        for p in points_on_line:
            ##if we want to display on img

            feature_img[int(p[1]), int(p[0])] = 0.99

            if int(p[0]) + 2 < feature_img.shape[1] and int(p[1]) + 2 < feature_img.shape[0]:
                ##right
                feature_img[int(p[1]) + 1, int(p[0])] = 0.99
                feature_img[int(p[1]) + 2, int(p[0])] = 0.99
                ##left
                feature_img[int(p[1]) - 1, int(p[0])] = 0.99
                feature_img[int(p[1]) - 2, int(p[0])] = 0.99
                ##up
                feature_img[int(p[1]), int(p[0]) - 1] = 0.99
                feature_img[int(p[1]), int(p[0]) - 2] = 0.99
                ##down
                feature_img[int(p[1]), int(p[0]) + 1] = 0.99
                feature_img[int(p[1]), int(p[0]) + 2] = 0.99


        ### add spine ##
        points_on_line = h_ae.createLineIterator(np.array([limb_pos_x[6], limb_pos_y[6]]) \
                                                 , np.array([limb_pos_x[7], limb_pos_y[7]]), feature_img)
        for p in points_on_line:
            ##if we want to display on img

            feature_img[int(p[1]), int(p[0])] = 0.99

            if int(p[0]) + 2 < feature_img.shape[1] and int(p[1]) + 2 < feature_img.shape[0]:
                ##right
                feature_img[int(p[1]) + 1, int(p[0])] = 0.99
                feature_img[int(p[1]) + 2, int(p[0])] = 0.99
                ##left
                feature_img[int(p[1]) - 1, int(p[0])] = 0.99
                feature_img[int(p[1]) - 2, int(p[0])] = 0.99
                ##up
                feature_img[int(p[1]), int(p[0]) - 1] = 0.99
                feature_img[int(p[1]), int(p[0]) - 2] = 0.99
                ##down
                feature_img[int(p[1]), int(p[0]) + 1] = 0.99
                feature_img[int(p[1]), int(p[0]) + 2] = 0.99


        cv2.imshow('feature_img',feature_img)
        cv2.waitKey(0)

        if len(imgs) > 0: imgs = np.vstack((imgs, feature_img.reshape((1, -1))))
        else: imgs = feature_img.reshape((1, -1))


    return imgs


AE_weights_level_1 = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/deep_AE_900_225_weights_008hd1_noNoise.txt')
hd_weights = AE_weights_level_1[0][0]
bias_1_level1 = AE_weights_level_1[1]

#pca = decomposition.PCA(n_components=100)  # 2-dimensional PCA whiten=True, svd_solver='randomized'
pca = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/clustering_posture_5sec/100pca_deep900225AE_5sec_data.txt')
cluster_model = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/clustering_posture_5sec/linearSVM_agglomerative15c_5sec_100pca.txt')


def compute_hot_f(xs,ys):
    orientation_intervals = [[range(0, 45)], [range(45, 90)], [range(90, 135)], [range(135, 180)], [range(180, 225)],
                             [range(225, 270)], \
                             [range(270, 315)], [range(315, 360)]]
    magnitude_intervals = [[range(0, 4)], [range(4, 10)], [range(10, 200)]]

    hot_matrix = np.zeros((len(orientation_intervals), len(magnitude_intervals)))

    step = 10


    for i in xrange(0, len(xs) - step):

        dx = abs(float(xs[i + step]) - float(xs[i]))
        dy = abs(float(ys[i + step]) - float(ys[i]))

        orientation = int(degrees(atan2(dy, dx)) % 360)
        magn = int(np.sqrt((np.power(dx, 2) + np.power(dy, 2))))

        # list_magn.append(magn)

        for c_interval, o_interval in enumerate(orientation_intervals):
            if orientation in o_interval[0]:
                if magn in magnitude_intervals[0][0]:
                    hot_matrix[c_interval][0] += 1
                    break
                elif magn in magnitude_intervals[1][0]:
                    hot_matrix[c_interval][1] += 1
                    break
                elif magn in magnitude_intervals[2][0]:
                    hot_matrix[c_interval][2] += 1
                    break
        ##control whether the values are in the intervals
        if hot_matrix.sum() == 0:
            print 'orientation or magn not in the intervals'
            print orientation, magn

    return normalize(hot_matrix.reshape((len(orientation_intervals)*len(magnitude_intervals))))


def subject_in_key_areas(head_x, head_y, head_z, key_areas):

    boxes2D_pos = key_areas[0]
    boxes3D_pos = key_areas[1]

    in_key_areas = np.zeros((1,len(boxes2D_pos)))

    for i_coord in range(0, len(head_x), int(len(head_x) / 15)):
        for i_b,b in enumerate(boxes2D_pos):
            if b.contains_point((int(head_x[i_coord]),int(head_y[i_coord]))):
                if abs(head_z[i_b] - boxes3D_pos[i_b]) < 0.2:
                    #print i_b, head_z[i_b], boxes3D_pos[i_b]
                    in_key_areas[0][i_b] += 1

    #print in_key_areas
    return in_key_areas/20





def extract_word_posture(participant_data,key_areas,goal):
    #scene = np.zeros((414, 512, 3), dtype=np.uint8)
    #scene += 255

    task_feature_img = []
    task_features =[]

    for i_task, task in enumerate(participant_data[:4]):
        print 'task: ', i_task
        if len(task) == 0: continue

        n_sec_data = []
        n_task_features = []


        for n_slice in range(0, len(task)):
            if len(task[n_slice]) <= 1 : continue
            #print 'n_slice ', n_slice

            flat_list = [item for item in task[n_slice]]

            ##### arms  ########
            shoulder_left_x, shoulder_left_y, zs, ids = img_processing.get_coordinate_points(flat_list, joint_id=10)
            shoulder_right_x, shoulder_right_y, zs, ids = img_processing.get_coordinate_points(flat_list, joint_id=6)

            elbow_left_x, elbow_left_y, zs, ids = img_processing.get_coordinate_points(flat_list, joint_id=11)
            elbow_right_x, elbow_right_y, zs, ids = img_processing.get_coordinate_points(flat_list, joint_id=7)

            wrist_left_x, wrist_left_y, zs, ids = img_processing.get_coordinate_points(flat_list, joint_id=12)
            wrist_right_x, wrist_right_y, zs, ids = img_processing.get_coordinate_points(flat_list, joint_id=8)

            #### spinal ###
            head_x, head_y, head_z, ids = img_processing.get_coordinate_points(flat_list, joint_id=1)
            spineBase_x,spineBase_y, z, ids = img_processing.get_coordinate_points(flat_list, joint_id=4)
            ##

            # get_dist_arms(shoulder_left_x, shoulder_left_y, shoulder_right_x, shoulder_right_y, elbow_left_x,
            #               elbow_left_y, elbow_right_x, elbow_right_y, wrist_left_x,wrist_left_y, wrist_right_x, wrist_right_y)


            if len(shoulder_left_x)< 30: continue


            ### AE features ###
            feature_imgs = extract_arms_pos(shoulder_left_x, shoulder_left_y, shoulder_right_x, shoulder_right_y, elbow_left_x,
                          elbow_left_y, elbow_right_x, elbow_right_y, wrist_left_x,wrist_left_y, wrist_right_x, wrist_right_y,head_x, head_y,spineBase_x,spineBase_y)

            ### Other features ###
            ## compute hot ##
            hot = compute_hot_f(head_x, head_y)

            ## check whether the participant is in key areas ##
            in_key_areas = subject_in_key_areas(head_x, head_y, head_z, key_areas)

            n_task_features.append([hot,in_key_areas])

            if goal == 'train_AE':
                ### if we want to extract the data to train AE
                for img in feature_imgs:
                    if len(task_feature_img)>0: task_feature_img= np.vstack((task_feature_img,img))
                    else: task_feature_img = img
            elif goal == 'test_AE':
                ### if we want to test AE for denoising and descrtize the posture ##

                for img in feature_imgs:
                    ## decompose it with trained AE and concatanate
                    if len(n_sec_data)>0:
                        ##shallow AE
                        #n_sec_data = np.hstack((n_sec_data, img_processing.sigmoid_function((np.dot(img, hd_weights)) + bias_1_level1)))
                        ##deep AE
                        act = img_processing.sigmoid_function((np.dot(img, hd_weights)) + bias_1_level1)
                        n_sec_data = np.hstack(
                            (n_sec_data, img_processing.sigmoid_function((np.dot(act, AE_weights_level_1[0][1])) + AE_weights_level_1[2])))

                    else:
                        ##shallow AE
                        #n_sec_data = img_processing.sigmoid_function((np.dot(img, hd_weights)) + bias_1_level1)
                        ##deep AE
                        act = img_processing.sigmoid_function((np.dot(img, hd_weights)) + bias_1_level1)
                        n_sec_data =img_processing.sigmoid_function((np.dot(act, AE_weights_level_1[0][1])) + AE_weights_level_1[2])

                    ## when we reach desired time (now 10 seconds)
                    if n_sec_data.shape[0] == (AE_weights_level_1[0][1].shape[1]*50):

                        ## PCA and then clustering
                        task_feature_img.append(cluster_model.predict(pca.transform(n_sec_data.reshape(1, -1)))[0])
                        n_sec_data = []

    print task_feature_img
    print Counter(task_feature_img)
    print task_feature_img.shape
    return task_feature_img


def main_posture_extr():


    skeleton_data_in_tasks_and_time_slices = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/skeleton_data_in_tasks_time_slices_30fps_ordered.txt')

    scene = np.zeros((424,512),dtype=np.uint8)
    # scene += 255
    boxes, zs, scene = data_organizer.get_areas_boxes(scene)

    participants_features = []
    for i_p in xrange(9,len(skeleton_data_in_tasks_and_time_slices)):
        task_feature_img = extract_word_posture(skeleton_data_in_tasks_and_time_slices[i_p], [boxes,zs],goal='test_AE')
        participants_features.append(task_feature_img)

    ## plot dist ##
    #plot_dist()

    #data_organizer.save_matrix_pickle(participants_features, 'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/posture_upperBody_10fps_noNoise.txt')



if __name__ == '__main__':
    main_posture_extr()