import video_traj as traj_analysis
import ambient_sensor as ambient_sensor_analysis

import video_traj_oldDataset

import visualization as my_vis



def retieve_data():
    print 'starting ABD system'
    ##-------------Retrieve Data------------------##
    file_traj = 'C:/Users/dario.dotti/Documents/pilot_abnormal_behavior_indoor/joints/subject4_points.xml'
    old_file_traj = 'C:/Users/dario.dotti/Documents/tracking_points/tracking_data_kinect2/2_5.txt'
    file_AS = 'C:/Users/dario.dotti/Documents/pilot_abnormal_behavior_indoor/binary/18-10-16_sensors_subject4.txt'

    ##------------Call Specific Function----------##

    ##get feature from trajectories
    #video_traj_feature = traj_analysis.feature_extraction_video_traj(file_traj)
    #video_traj_feature = video_traj_oldDataset.feature_extraction_video_traj(old_file_traj)


    ##get features from binary sensors
    sensors_ID = ambient_sensor_analysis.org_data_ID(file_AS)

    #ambient_sensor_analysis.nr_visit_bathroom(sensors_ID)

    ##Cluster without the extra bin
    #cl_pred_occupancy = my_exp.cluster_kmeans(video_traj_feature[0][:,:96],3)
    #cl_pred_HOT = my_exp.cluster_kmeans(video_traj_feature[1],3)

    ##visulaization daily motion
    #my_vis.bar_plot_occupancy_selectedAreas_over_time(video_traj_feature[0])

    #my_vis.bar_plot_motion_over_time(video_traj_feature[1])

    #my_vis.pie_plot_motion_day(video_traj_feature[1])

    ##visulaization apathy
    motion_week = [14.8787,22.3841,16.0710,16.4378,24.2042,28.3984,12.3949]
    my_vis.bar_plot_motion_in_region_over_long_time(motion_week)






if __name__ == '__main__':
    retieve_data()