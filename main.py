import numpy as np
import cv2
import os
import scipy.io


import data_organizer as my_data_org
import ambient_sensors as ambient_sensor_analysis
import video_traj



def get_video_features():
    #read and parse file with recorded data
    with open('C:/Users/dario.dotti/Desktop/data_recordings_master/file_to_analize_master_recordings.txt','r') as f:#C:/Users/dario.dotti/Documents/Datasets/my_dataset/wandering_dataset_um/file_to_analyze_6_labels_ordered.txt
        files = f.read().split('\n')
        print 'number of recorded files: '+str(len(files))

    matrix_allData_HOT = []
    for file in files:
        print file

        filename = os.path.basename(file)

        video_traj.set_subject(filename.split('_')[0])

        traj_features = video_traj.feature_extraction_video_traj(file)

        matrix_allData_HOT.append(traj_features[1])
        # if len(matrix_allData_HOT)>0:
        #     matrix_allData_HOT = np.vstack((matrix_allData_HOT,traj_features[1]))
        # else:
        #     matrix_allData_HOT = np.array(traj_features[1])

    print len(matrix_allData_HOT)
    #scipy.io.savemat('C:/Users/dario.dotti/Documents/hot_spatial_grid_4x4.mat',mdict={'spatial_grid_4x4': matrix_allData_HOT})
    #my_data_org.save_matrix_pickle(matrix_allData_HOT,'C:/Users/dario.dotti/Desktop/data_recordings_master/master_skeleton_data_in_tasks_time_slices_30fps.txt') #C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/skeleton_data_in_tasks_time_slices_30fps_ordered


def get_ambient_sensor_features():
    #read and parse file with recorded data
    with open('C:/Users/dario.dotti/Documents/file_to_analyze_AS_5_labels.txt','r') as f:
        files = f.read().split('\n')
        print 'number of recorded files: '+str(len(files))


    matrix_allData_as = []
    for file in files:
        print file

        activation_matrix = ambient_sensor_analysis.feature_extraction_as(file)
        print np.array(activation_matrix).shape
        if len(matrix_allData_as)>0:
            matrix_allData_as = np.vstack((matrix_allData_as,activation_matrix))
        else:
            matrix_allData_as = activation_matrix

    my_data_org.save_matrix_pickle(matrix_allData_as,'C:/Users/dario.dotti/Documents/AS_activation_5_labels_transformed.txt')


def main():
    scene = cv2.imread('C:/Users/dario.dotti/Documents/Datasets/my_dataset/wandering_dataset_um/subject4_1834.jpg')#KinectScreenshot-Color-12-25-18 - Copy

    ##get features from video trajectories
    get_video_features()

    ##ambient sensor
    #get_ambient_sensor_features()


    # ambient_sensor_analysis.org_data_different_tasks(file_AS)
    #sensors_ID = ambient_sensor_analysis.org_data_ID(file_AS)

    #ambient_sensor_analysis.nr_visit_bathroom(sensors_ID)









if __name__ == '__main__':
    main()