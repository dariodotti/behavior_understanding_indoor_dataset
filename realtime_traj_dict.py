import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt


import img_processing
import data_organizer
import video_traj
import hierarchical_ae_learning_methods as hs




def extract_traj_word_first_layer(slices):

    ###define max distance between N consecutive points
    # list_distances = []
    # n = 1
    # for slice in slices:
    #     x_f, y_f,z, ids = img_processing.get_coordinate_points(slice,joint_id=1)
    #     map(lambda i_points: list_distances.append(
    #         np.sqrt(np.power(x_f[i_points + n] - x_f[i_points], 2) + np.power(y_f[i_points + n] - y_f[i_points], 2))),
    #         xrange(len(x_f) - n))
    #
    # plt.scatter(range(0,len(list_distances)),list_distances)
    # plt.show()


    # scene = cv2.imread('C:/Users/dario.dotti/Documents/Datasets/my_dataset/wandering_dataset_um/exp_scene_depth.jpg')
    scene = np.zeros((414, 512, 3), dtype=np.uint8)
    scene += 255

    matrix_features = []
    matrix_orig_points = []

    for i_slices, slice in enumerate(slices[10:]):
        # print i_slices+10
        if i_slices % 50 == 0: print i_slices, datetime.now().time()

        # video_traj.draw_joints_and_tracks(slice, [])

        # get x,y,z of every traj point after smoothing process
        x_f, y_f, z, ids = img_processing.get_coordinate_points(slice, joint_id=1)

        ########### start hierarchical autoencoder learning #######################

        size_mask = 18
        # print step
        max_step = np.sqrt(np.power(((size_mask - 3) - 0), 2) + np.power(((size_mask - 3) - 0), 2)) * 1.3

        start_t = 0
        first_point_traj = [x_f[start_t], y_f[start_t]]

        temp_scene = scene.copy()

        labels_history = []
        directions_history = []

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

                directions_history.append(directions[0])

                ##create grid according to the direction of the trajectory
                rects_in_grid = hs.create_grid(xs_untilNow, ys_unilNow, size_mask, directions, temp_scene)

                ##compute the features from traj chuncks in rect
                traj_features, orig_points = hs.transform_traj_in_pixel_activation(rects_in_grid, xs_untilNow,
                                                                                   ys_unilNow, size_mask, max_step)

                #########store final matrix#################
                if len(matrix_features) > 0:
                    matrix_features = np.vstack((matrix_features, traj_features.reshape((1, -1))))
                else:
                    matrix_features = traj_features.reshape((1, -1))

                ##store original points
                # if len(matrix_orig_points) > 0:
                #     matrix_orig_points = np.vstack((matrix_orig_points, orig_points.reshape((1, -1))))
                # else:
                #     matrix_orig_points = orig_points.reshape((1, -1))
                ############

                # ##update the beginning of the trajectory
                start_t = i_p - 1
                first_point_traj = [x_f[start_t], y_f[start_t]]

    print matrix_features.shape
    ##save matrix
    data_organizer.save_matrix_pickle(matrix_features,
                                      'C:/Users/dario.dotti/Documents/data_for_personality_exp/computed_matrix/matrix_features_l1.txt')


def main_realtime_traj_dict():
    slices = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/traj_org_by_ID_10fps.txt')














if __name__ == '__main__':
        main_realtime_traj_dict()