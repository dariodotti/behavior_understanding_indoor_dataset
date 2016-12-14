from sklearn.linear_model import LogisticRegression
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import threading

import data_organizer
import video_traj

#scene = np.ones((414,512),dtype=np.uint8)
scene = cv2.imread('C:/Users/dario.dotti/Documents/pilot_abnormal_behavior_indoor/subject4_1834.jpg')
threadLock = threading.Lock()



def plot_classifier_confidence(task,cluster_model,keys_labels,lr):
    ##plotting confidence classifier  on bow

    plt.axis([-1, 80, -10, 20])
    plt.ion()
    x_axis=0

    hist = np.zeros((1,len(keys_labels)))


    ##every two second create a sample cluster it and create an hist
    for n_slice,two_mins_slice in enumerate(task):

        similar_word= cluster_model.predict(np.array(two_mins_slice).reshape(1,-1))

        index = np.where(similar_word == keys_labels)[0]

        hist[0][index] +=1
        pred =  lr.predict(hist)
        print pred
        conf_confusion = np.max([lr.decision_function(hist)[0][0],lr.decision_function(hist)[0][1],lr.decision_function(hist)[0][2]])
        conf_repetitive = lr.decision_function(hist)[0][3]
        conf_adl = np.max([lr.decision_function(hist)[0][4],lr.decision_function(hist)[0][5]])
        #print conf_confusion,conf_repetitive,conf_adl

        plt.plot(x_axis,conf_confusion,'r^',label='Conf')
        plt.plot(x_axis,conf_repetitive,'b^',label='Ripet')
        plt.plot(x_axis,conf_adl,'g^',label='Adl')
        #draw the legend only once
        if n_slice==0:
            plt.legend(loc='upper left')

        plt.pause(2)
        x_axis +=1


    #plt.pause(100)


def draw_joints_and_tracks(body_points,dd):



    color = (0,0,255)
    for n_frame,traj_body_joints in enumerate(body_points):
        # Get lock to synchronize threads
        threadLock.acquire()


        temp_img = scene.copy()

        #draw line between joints
        thickness = 3
        line_color = (19,19,164)
        #first position skipped cause there are other info stored
        #torso
        cv2.line(temp_img,(int(float(traj_body_joints[1,0])),int(float(traj_body_joints[1,1]))),(int(float(traj_body_joints[2,0])),int(float(traj_body_joints[2,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[2,0])),int(float(traj_body_joints[2,1]))),(int(float(traj_body_joints[3,0])),int(float(traj_body_joints[3,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[3,0])),int(float(traj_body_joints[3,1]))),(int(float(traj_body_joints[4,0])),int(float(traj_body_joints[4,1]))),line_color,thickness)
        #shoulder
        cv2.line(temp_img,(int(float(traj_body_joints[5,0])),int(float(traj_body_joints[5,1]))),(int(float(traj_body_joints[6,0])),int(float(traj_body_joints[6,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[5,0])),int(float(traj_body_joints[5,1]))),(int(float(traj_body_joints[10,0])),int(float(traj_body_joints[10,1]))),line_color,thickness)
        #hips
        cv2.line(temp_img,(int(float(traj_body_joints[4,0])),int(float(traj_body_joints[4,1]))),(int(float(traj_body_joints[14,0])),int(float(traj_body_joints[14,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[4,0])),int(float(traj_body_joints[4,1]))),(int(float(traj_body_joints[18,0])),int(float(traj_body_joints[18,1]))),line_color,thickness)
        #right arm
        cv2.line(temp_img,(int(float(traj_body_joints[6,0])),int(float(traj_body_joints[6,1]))),(int(float(traj_body_joints[7,0])),int(float(traj_body_joints[7,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[7,0])),int(float(traj_body_joints[7,1]))),(int(float(traj_body_joints[8,0])),int(float(traj_body_joints[8,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[8,0])),int(float(traj_body_joints[8,1]))),(int(float(traj_body_joints[9,0])),int(float(traj_body_joints[9,1]))),line_color,thickness)
        #left arm
        cv2.line(temp_img,(int(float(traj_body_joints[10,0])),int(float(traj_body_joints[10,1]))),(int(float(traj_body_joints[11,0])),int(float(traj_body_joints[11,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[11,0])),int(float(traj_body_joints[11,1]))),(int(float(traj_body_joints[12,0])),int(float(traj_body_joints[12,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[12,0])),int(float(traj_body_joints[12,1]))),(int(float(traj_body_joints[13,0])),int(float(traj_body_joints[13,1]))),line_color,thickness)

        #right leg
        cv2.line(temp_img,(int(float(traj_body_joints[14,0])),int(float(traj_body_joints[14,1]))),(int(float(traj_body_joints[15,0])),int(float(traj_body_joints[15,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[15,0])),int(float(traj_body_joints[15,1]))),(int(float(traj_body_joints[16,0])),int(float(traj_body_joints[16,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[16,0])),int(float(traj_body_joints[16,1]))),(int(float(traj_body_joints[17,0])),int(float(traj_body_joints[17,1]))),line_color,thickness)
        #left leg
        cv2.line(temp_img,(int(float(traj_body_joints[18,0])),int(float(traj_body_joints[18,1]))),(int(float(traj_body_joints[19,0])),int(float(traj_body_joints[19,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[19,0])),int(float(traj_body_joints[19,1]))),(int(float(traj_body_joints[20,0])),int(float(traj_body_joints[20,1]))),line_color,thickness)
        cv2.line(temp_img,(int(float(traj_body_joints[20,0])),int(float(traj_body_joints[20,1]))),(int(float(traj_body_joints[21,0])),int(float(traj_body_joints[21,1]))),line_color,thickness)

        for i,joint in enumerate(traj_body_joints):
            if i == 0:
                continue
            elif i == 1:
                ##draw trajectories
                cv2.circle(scene,(int(float(joint[0])),int(float(joint[1]))),2,color,-1)
            else:
                cv2.circle(temp_img,(int(float(joint[0])),int(float(joint[1]))),2,color,-1)



        cv2.imshow('skeleton',temp_img)
        cv2.waitKey(30)

        # Free lock to release next thread
        threadLock.release()



def main_demo():

    bow_data = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_demo/BOW_16subject_2sec.txt')
    labels_bow_data = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_demo/BOW_labels_16subject_2sec.txt')

    lr = LogisticRegression()
    lr.fit(bow_data,np.ravel(labels_bow_data))

    ##take data every 2 seconds and start classify it
    HOT_16_subject_6_tasks = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_demo/HOT_matrix_16_subject_6_tasks.txt')

    print np.array(HOT_16_subject_6_tasks).shape
    ##cluster data
    cluster_model= data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/cl_model_2secWindow_band03.txt')
    labels_cluster = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_demo/cluster_labels.txt')
    labels_cluster_counter = Counter(labels_cluster).most_common(40)
    keys_labels = map(lambda x: x[0], labels_cluster_counter)

    ##choose the subject and task to show
    subj = 3 #n_subject order: 4,5,6,7,10,11,12,13,14,19,20,15,3,16,17,18
    task = 0 #n_task order confusion: 0,1,2 repetitive:3 house_activity: 4,5

    body_joints = video_traj.xml_parser('C:/Users/dario.dotti/Documents/pilot_abnormal_behavior_indoor/joints/subject7_points.xml')

    ###Running and displaying processes at the same time
    dd=0
    display_joints_traj = threading.Thread(target=draw_joints_and_tracks,args=(body_joints,dd))
    display_confidence_classifier = threading.Thread(target=plot_classifier_confidence,args=(HOT_16_subject_6_tasks[subj][task],cluster_model,keys_labels,lr))

    display_joints_traj.start()
    display_confidence_classifier.start()


    #display_joints_traj.join()
    #display_confidence_classifier.join()




















if __name__ == '__main__':
    main_demo()