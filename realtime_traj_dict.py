import numpy as np
import cv2
import math
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import img_processing
import data_organizer
import video_traj
import hierarchical_ae_learning_methods as hs
import AE_rec





# AE_weights_level_1 = data_organizer.load_matrix_pickle(
#         'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/ae/head_joint_id1/144weights_l1_hd1002.txt')
#
# cluster_model_l1 = data_organizer.load_matrix_pickle(
#         'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/20_cluster_model_layer1.txt')

def encode_features_using_AE_layer1_cluster_activation(feature_matrix,layer):

    ##check visually the reconstruction
    #AE_rec.AE_reconstruction_level1(feature_matrix, AE_weights_level_1)


    for test_traj in feature_matrix:
        if sum(test_traj) != 0:

            ##compute AE reconstruction
            hd1_space = np.dot(test_traj, AE_weights_level_1[0][0])
            activations = AE_rec.sigmoid_function(hd1_space + AE_weights_level_1[1])

        else:
            activations= np.zeros((1,AE_weights_level_1[0][0].shape[1]))
            activations = activations[0]


    if layer == 'layer2':
        return activations
    elif layer == 'layer1':
        label = cluster_model_l1.predict(activations.reshape((1,-1)))[0]
        return label


def extract_traj_word_spatio_temporal_grid(participant_data, n_layer):

    create_activation_layer2 = 1
    create_bayes_vector = 0


    #scene = cv2.imread('C:/Users/dario.dotti/Documents/Datasets/my_dataset/wandering_dataset_um/exp_scene_depth.jpg')
    scene = np.zeros((414, 512, 3), dtype=np.uint8)
    scene += 255


    training_bayes_vector = []


    for i_task, task in enumerate(participant_data):
        print 'task: ',i_task
        if len(task)==0: continue

        if n_layer == 1:
            matrix_features = []
        elif n_layer == 2:
            matrix_activations = []
            matrix_orig_points = []
            training_bayes_vector_task = []


        for n_slice in range(0,len(task)):
            print 'n_slice ', n_slice
            flat_list = [item for item in task[n_slice]]

            video_traj.draw_joints_and_tracks(flat_list, [])

            # get x,y,z of every traj point after smoothing process
            x_f, y_f, z, ids = img_processing.get_coordinate_points(flat_list, joint_id=1)

            ########### start hierarchical autoencoder learning #######################

            size_mask = 18
            # print step
            max_step = np.sqrt(np.power(((size_mask - 3) - 0), 2) + np.power(((size_mask - 3) - 0), 2)) * 1.3

            start_t = 0
            first_point_traj = [x_f[start_t], y_f[start_t]]

            temp_scene = scene.copy()

            labels_history = []
            directions_history = []
            activation_history = []
            orig_points_history = []


            for i_p in xrange(1, len(x_f)):

                ##accumulate traj points until the distance between the first point and current point is enough for the grid
                d = np.sqrt(((x_f[i_p] - first_point_traj[0]) ** 2) + ((y_f[i_p] - first_point_traj[1]) ** 2))

                ##if the distance is enough compute the grid starting from the first point until the current point

                if abs(d - max_step) < 8:

                    xs_untilNow = x_f[start_t:i_p]
                    ys_unilNow = y_f[start_t:i_p]

                    # print len(xs_untilNow), len(ys_unilNow)

                    if len(xs_untilNow) > 30:
                        # ##update the beginning of the trajectory
                        start_t = i_p - 1
                        first_point_traj = [x_f[start_t], y_f[start_t]]
                        continue

                    ##get directions of the traj chunck using first and last point
                    # direction = get_direction_traj([x_f[start_t],y_f[start_t]],[x_f[i_p],y_f[i_p]])
                    directions = hs.get_directions_traj(xs_untilNow, ys_unilNow)
                    if directions[0] == -180: directions[0] = 180
                    directions_history.append(directions[0])

                    ##create grid according to the direction of the trajectory
                    rects_in_grid = hs.create_grid(xs_untilNow, ys_unilNow, size_mask, directions, temp_scene)

                    ##compute the features from traj chuncks in rect
                    traj_features, orig_points = hs.transform_traj_in_pixel_activation(rects_in_grid, xs_untilNow,
                                                                                       ys_unilNow, size_mask, max_step)

                    if n_layer ==1:
                    #########store final matrix#################
                        if len(matrix_features) > 0:
                            matrix_features = np.vstack((matrix_features, traj_features.reshape((1, -1))))
                        else:
                            matrix_features = traj_features.reshape((1, -1))

                    elif n_layer == 2:

                        orig_points_history.append(orig_points)

                        activation = encode_features_using_AE_layer1_cluster_activation(traj_features, 'layer2')

                        activation_history.append(activation)

                        if len(activation_history) == 3 :

                            cv2.imshow('scene', temp_scene)
                            cv2.waitKey(0)

                            if create_activation_layer2:
                                ##extract features for AE layer2
                                matrixt_activation_l2, original_points_l2 = hs.create_vector_activations_layer_2(
                                    directions_history, activation_history, orig_points_history)

                                ##save activations for layer2
                                if len(matrix_activations) > 0: matrix_activations = np.vstack((matrix_activations, matrixt_activation_l2))
                                else: matrix_activations = matrixt_activation_l2

                                ##save original for layer2
                                if len(matrix_orig_points) > 0: matrix_orig_points = np.vstack((matrix_orig_points, original_points_l2))
                                else: matrix_orig_points = original_points_l2

                            elif create_bayes_vector:
                                for a in activation_history:
                                    label = cluster_model_l1.predict(a.reshape((1, -1)))[0]
                                    labels_history.append(label)

                                ###create vector with two info label
                                vector_bayes = hs.create_vector_for_bayesian_probability(
                                    labels_history, directions_history, 3)

                                ## saving data for cluster prediction
                                if len(training_bayes_vector_task) > 0: training_bayes_vector_task = np.vstack((training_bayes_vector_task, vector_bayes))
                                else: training_bayes_vector_task = vector_bayes




                            ##refresh history
                            orig_points_history = []
                            directions_history = []
                            activation_history = []
                            orig_points_history = []
                            labels_history = []


                    ##GENERAL FOR ALL THE LAYERS## Update the beginning of the trajectory
                    start_t = i_p - 1
                    first_point_traj = [x_f[start_t], y_f[start_t]]

            #training_bayes_vector.append(training_bayes_vector_task)
    #print matrix_activations.shape

    ##save matrix activation
    # data_organizer.save_matrix_pickle(matrix_activations,
    #                                   'C:/Users/dario.dotti/Documents/data_for_personality_exp/computed_matrix/matrix_activations_l2.txt')
    # ## save original points
    # data_organizer.save_matrix_pickle(matrix_orig_points,
    #                                   'C:/Users/dario.dotti/Documents/data_for_personality_exp/computed_matrix/matrix_orig_points_l2.txt')

    print np.array(training_bayes_vector).shape

    # data_organizer.save_matrix_pickle(training_bayes_vector,
    #                                   'C:/Users/dario.dotti/Documents/data_for_personality_exp/computed_matrix/bayes_vector.txt')


def extract_traj_word_temporal_window(participant_data, n_layer):
    scene = cv2.imread('C:/Users/dario.dotti/Documents/Datasets/my_dataset/wandering_dataset_um/exp_scene_depth.jpg')
    #scene = np.zeros((414, 512, 3), dtype=np.uint8)
    #scene += 255

    size_mask = 20

    #max_step = np.sqrt(np.power(((size_mask - 3) - 0), 2) + np.power(((size_mask - 3) - 0), 2)) * 1.3
    max_step = 23

    matrix_features_participant = []
    matrix_original_points_participants = []

    for i_task, task in enumerate(participant_data):
        print 'task: ', i_task
        if len(task) == 0: continue

        create_activation_layer2 = 1

        labels_history = []
        directions_history = []
        activation_history = []
        orig_points_history = []
        temp_scene = scene.copy()

        matrix_features_task = []
        matrix_activations_task = []
        matrix_orig_points_task = []

        temp_scene = scene.copy()

        for n_slice in range(0, len(task)):
            if len(task[n_slice]) <= 1 : continue
            print 'n_slice ', n_slice

            flat_list = [item for item in task[n_slice]]

            video_traj.draw_joints_and_tracks(flat_list, [])

            # get x,y,z of every traj point after smoothing process
            x_f, y_f, z, ids = img_processing.get_coordinate_points(flat_list, joint_id=1)

            # for point in range(len(x_f)):
            #     cv2.circle(temp_scene,(x_f[point],y_f[point]),1,(0,0,255),-1)
            # cv2.imshow('ciao', temp_scene)
            # cv2.waitKey(0)

            directions = hs.get_directions_traj(x_f, y_f)
            if directions[0] == -180: directions[0] = 180
            directions_history.append(directions[0])

            distances = [np.sqrt(((x_f[0] - x_f[i_p]) ** 2) + ((y_f[0] - y_f[i_p]) ** 2)) for i_p in range(1, len(y_f))]
            index_max = int(np.where(distances == np.max(distances))[0][0])
            size_mask_in_temporalWindow = distances[index_max]

            ##create grid according to the direction of the trajectory
            rects_in_grid = hs.create_grid(x_f, y_f, size_mask_in_temporalWindow, directions, temp_scene)

            origin_mask = [rects_in_grid[0].vertices[1][0], rects_in_grid[0].vertices[1][1]]

            x_converted = [x - origin_mask[0] for x in x_f]
            y_converted = [y - origin_mask[1] for y in y_f]


            ### Get the max distance in the list ####
            print distances[index_max]
            if distances[index_max] > size_mask/2:

                OldRange = (distances[index_max])
                NewRange = 20

                New_x = [(((x) * NewRange) / OldRange) + 0 for x in x_converted]
                New_y = [(((y) * NewRange) / OldRange) + 0 for y in y_converted]

                ##create grid according to the direction of the trajectory
                rects_in_grid = hs.create_grid(New_x, New_y, size_mask, directions, temp_scene)

                ##compute the features from traj chuncks in rect
                traj_features,orig_points = hs.transform_traj_in_pixel_activation_temporal(rects_in_grid[0], New_x, New_y, size_mask, 10)

            else:

                traj_features, orig_points = hs.transform_traj_in_pixel_activation(rects_in_grid, x_f, y_f, size_mask, 10)
                if np.sum(traj_features) == 0: continue


            if n_layer == 1:
                if len(matrix_features_task) > 0:
                    matrix_features_task = np.vstack((matrix_features_task, traj_features.reshape((1, -1))))
                else:
                    matrix_features_task = traj_features.reshape((1, -1))

            elif n_layer == 2:


                orig_points_history.append(orig_points)

                activation = encode_features_using_AE_layer1_cluster_activation(traj_features, 'layer2')

                activation_history.append(activation)

                if len(activation_history) == 3:

                    # cv2.imshow('scene', temp_scene)
                    # cv2.waitKey(0)

                    if create_activation_layer2:
                        ##extract features for AE layer2
                        # matrixt_activation_l2, original_points_l2 = hs.create_vector_activations_layer_2(
                        #     directions_history, activation_history, orig_points_history)
                        matrixt_activation_l2 = activation_history[0]
                        for i_act in range(1,len(activation_history)):
                            matrixt_activation_l2 = np.hstack((matrixt_activation_l2, activation_history[i_act]))

                        original_points_l2 = orig_points_history[0]
                        for i_org_p in range(1,len(orig_points_history)):
                            original_points_l2 = np.hstack((original_points_l2, orig_points_history[i_org_p]))

                        ##save activations for layer2
                        if len(matrix_activations_task) > 0: matrix_activations_task = np.vstack((matrix_activations_task, matrixt_activation_l2))
                        else: matrix_activations_task = matrixt_activation_l2

                        ##save original for layer2
                        if len(matrix_orig_points_task) > 0: matrix_orig_points_task = np.vstack((matrix_orig_points_task, original_points_l2))
                        else: matrix_orig_points_task = original_points_l2

                    ##refresh history
                    orig_points_history = []
                    directions_history = []
                    activation_history = []
                    orig_points_history = []
                    labels_history = []


        matrix_features_participant.append(matrix_activations_task)
        matrix_original_points_participants.append(matrix_orig_points_task)


    return matrix_features_participant,matrix_original_points_participants


def get_distances_between_points(participant_data):

    ##get max distances in every time slice
    n = 1
    list_max_dist =[]
    list_dist_two_points = []
    for task in participant_data:
        for slice in task[:4]:
            if len(slice) <=1: continue
            list_distances = []
            flat_list = [item for item in slice]
            x_f, y_f, z, ids = img_processing.get_coordinate_points(flat_list, joint_id=1)
            a =1
            distances = [np.sqrt(((x_f[0] - x_f[i_p]) ** 2) + ((y_f[0] - y_f[i_p]) ** 2)) for i_p in range(1, len(y_f))]
            index_max = int(np.where(distances == np.max(distances))[0][0])


            dist_two_points = [np.sqrt(((x_f[i_p] - x_f[i_p+1]) ** 2) + ((y_f[i_p] - y_f[i_p+1]) ** 2)) for i_p in range(0, len(y_f)-1)]
            index_max_two_p = int(np.where(dist_two_points == np.max(dist_two_points))[0][0])

            list_dist_two_points.append(dist_two_points[index_max_two_p])
            list_max_dist.append(distances[index_max])

    #return np.array(list_max_dist).reshape(-1,1)

    # ###Eliminate outlier (bigger than double median), and set the size window
    best_ten =  np.sort(list_max_dist)[::-1][:10]
    #print 'AAAA'
    print best_ten
    med = np.median(best_ten)
    new_data = map(lambda x: x if x < (med*1.5) else False, list_max_dist)
    best_ten = np.sort(new_data)[::-1][:10]
    #print best_ten
    w_s =  int(np.max(new_data))
    print 'window size: ',w_s

    best_ten = np.sort(list_dist_two_points)[::-1][:10]
    #print best_ten
    med = np.median(best_ten)
    new_data = map(lambda x: x if x < (med * 3) else False, list_dist_two_points)
    best_ten = np.sort(new_data)[::-1][:10]
    #print best_ten
    max_step = int(np.max(new_data))
    print 'max_step: ', max_step




    return w_s,max_step


def visualize_cluster(data):

    my_kmean = KMeans(n_clusters=4,n_jobs=-1,init='k-means++')
    cluster_model = my_kmean.fit(data)


    plt.plot(range(0,len(data)), data[:, 0], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = my_kmean.cluster_centers_
    plt.scatter(range(0,len(centroids)), centroids[:, 0],
                marker='x', s=169, linewidths=3,
                color='b', zorder=10)
    plt.show()



def main_realtime_traj_dict():
    #slices = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/traj_org_by_ID_10fps.txt')

    skeleton_data_in_tasks_and_time_slices = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/skeleton_data_in_tasks_time_slices_30fps.txt')

    ### Staatistics on data #####
    # matrix_dist = []
    # for participant in skeleton_data_in_tasks_and_time_slices:
    #
    #     max_dist,max_step = get_distances_between_points(participant)
    #
    #     matrix_dist.append(max_dist)
    #
    # matrix_dist = np.array(matrix_dist).reshape(-1,1)
    # hs.determine_number_k_kMeans(matrix_dist)
    # visualize_cluster(matrix_dist)


    max_dist = 140
    max_step = 15

    final_matrix = []
    final_orig_points = []
    for participant in skeleton_data_in_tasks_and_time_slices:

        #extract_traj_word_spatio_temporal_grid(participant, n_layer=1)
        feature_participant,orig_point_participant = extract_traj_word_temporal_window(participant, n_layer=1)
        final_matrix.append(feature_participant)
        final_orig_points.append(orig_point_participant)

    print len(final_matrix),len(final_orig_points)

    # data_organizer.save_matrix_pickle(final_matrix, 'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/feature_matrix_participant_task_l2_new.txt')
    # data_organizer.save_matrix_pickle(final_orig_points,
    #                                   'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/orig_points_participant_task_l2_new.txt')








if __name__ == '__main__':
        main_realtime_traj_dict()