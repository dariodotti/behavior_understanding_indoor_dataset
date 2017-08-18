from sklearn.linear_model import LogisticRegression
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt

from collections import Counter
import threading
import time

import multiprocessing
from datetime import datetime as dt

from Tkinter import *

import PIL
from PIL import ImageTk, Image


import data_organizer
import video_traj
import ambient_sensors


#scene = np.ones((414,512),dtype=np.uint8)
scene = cv2.imread('C:/Users/dario.dotti/Pictures/taj_dataset_wandering/WANDERING_PECS/pecs_room.png')

#Create a window
window=Tk()
window.title('Abnormal Behavior Detector')


def draw_joints_and_tracks(body_points,current_time_shared):
    # make the thread wait for the other
    time.sleep(5.5)

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

        temp_img = scene.copy()

        # get recording time and make it as current time
        current_time = dt.strptime(traj_body_joints[0,1].split('.')[0], '%Y-%m-%d %H:%M:%S')
        current_time_shared.put(current_time)

        # draw joints
        #print n_frame

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

            if n_frame > 1:
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

                cv2.imshow('hospital room', temp_img)
                cv2.waitKey(1)
            else:
                cv2.imshow('hospital room', temp_img)
                cv2.waitKey(0)


        except:
            print 'traj coordinates not available'
            continue


def plot_classifier_confidence(task,cluster_model,keys_labels,lr,q):
    ##plotting confidence classifier  on bow
    time.sleep(8.5)
    hist = np.zeros((1,len(keys_labels)))

    x_axis = 0

    ##every two second create a sample cluster it and create an hist
    for n_slice,two_mins_slice in enumerate(task):

        similar_word= cluster_model.predict(np.array(two_mins_slice).reshape(1,-1))

        index = np.where(similar_word == keys_labels)[0]

        hist[0][index] +=1
        pred =  lr.predict(hist)

        print pred

        conf_confusion = np.max([lr.decision_function(hist)[0][0],lr.decision_function(hist)[0][2]])
        conf_repetitive = lr.decision_function(hist)[0][3]
        conf_adl = np.max([lr.decision_function(hist)[0][1],lr.decision_function(hist)[0][4],lr.decision_function(hist)[0][5]])


        q.put(['cc',conf_confusion,conf_repetitive,conf_adl,n_slice,x_axis])
        x_axis+=1

        time.sleep(2)

    #plt.pause(100)


def basic_plot():#Function to create the base plot, make sure to make global the lines, axes, canvas and any part that you would want to update later

    global ax_conf,ax_as,canvas,rect_open,rect_close,warning_img,emergency_img,normal_img,notification_icon,notification_text

    ##initialize figures
    main_fig = plt.figure()
    #ax_as = main_fig.add_subplot(212)
    ax_conf = main_fig.add_subplot(111)


    ##canvas in the main window
    canvas = FigureCanvasTkAgg(main_fig, master=window)
    canvas.show()
    ##in case of widget
    #canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    ##pack place the plot automatically, using place we can specify x,y
    #canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
    canvas._tkcanvas.place(x=80,y=20)


    ##inizialize plot of confidence of classifier
    ax_conf.axis([0,60,-10,20])
    ax_conf.plot(0,0)
    ax_conf.plot(0,0)
    ax_conf.plot(0,0)
    ax_conf.set_title('classifier confidence')


    ##initialize bar plot of ambient sensor
    #sensor = ['maindoor','toilet','livingroom']
    #ind = np.arange(len(sensor))
    #width = 0.2

    #ax_as.axis([-0.5,3,0,10])
    #ax_as.set_xticks(ind+width)
    #ax_as.set_xticklabels(sensor)

    #fake data except maindoor
    #activation_open =[0,5,6]
    #activation_close=[0,5,5]

    #bar charts
    #rect_open = ax_as.bar(ind, activation_open,width,color='red')
    #rect_close = ax_as.bar(ind+width, activation_close,width,color='blue')
    #ax_as.legend((rect_open[0],rect_close[1]),('door open','door close'),fontsize=9)
    #ax_as.set_title('ambient sensor')

    ##initialize notification icons and text
    warning_img = PIL.ImageTk.PhotoImage(PIL.Image.open('C:/Users/dario.dotti/Documents/data_for_demo/icon_warning_call_relative.png'))
    emergency_img = PIL.ImageTk.PhotoImage(PIL.Image.open('C:/Users/dario.dotti/Documents/data_for_demo/icon_emergency_call_doctors.png'))
    normal_img = PIL.ImageTk.PhotoImage(PIL.Image.open('C:/Users/dario.dotti/Documents/data_for_demo/house_daily_activity.png'))

    notification_icon = Label(window, image=warning_img)
    notification_text = Label(window, text='Calling the stakeholders')

    notification_title = Label(window, text='NOTIFICATION')
    notification_title.place(x=350, y=510)


def update_figures_in_threads(q):

    try:#Try to check if there is data in the queue
        result=q.get_nowait()

        if result !='Q':

            if result[0] == 'cc':
                print result

                ax_conf.plot(result[5],result[1],'r^',label='confusion')
                ax_conf.plot(result[5],result[2],'b^',label='repetitive')
                ax_conf.plot(result[5],result[3],'g^',label='normal activity')

                #draw the legend only once
                if result[4]==0:
                    ax_conf.legend(loc='upper left',fontsize=9)

                ##show notification images

                if max([result[1], result[2], result[3]]) == result[3]:
                    ##normal activity
                    ##show img without waiting for the threshold
                    update_notification_icons('normal_act')

                else:
                    ##confusion or ripetitive
                    ##show img only if higher than threshold
                    #if 2<result[1]< 3 or 2<result[2]< 3: update_notification_icons('warning')
                    if result[1]> 3 or result[2]> 3: update_notification_icons('emergency')



                canvas.draw()
                window.after(10, update_figures_in_threads, q)


            #elif result[0] == 'as':
                #rect_open[0].set_height(result[1][0])
                #rect_close[0].set_height(result[2][0])

                #canvas.draw()
                #window.after(10, update_figures_in_threads, q)
    except:
        ##no new input so refresh
        window.after(100, update_figures_in_threads, q)


def update_notification_icons(label_img):
    ##refreshing notification icons

    print label_img
    if label_img == 'warning':

        notification_icon.configure(image=warning_img)
        notification_icon.image = warning_img
        notification_icon.place(x=320, y=550)

        ##text
        notification_text.configure(text='Calling the stakeholders')
        notification_text.place(x=330, y=670)

    elif label_img == 'emergency':

        notification_icon.configure(image=emergency_img)
        notification_icon.image = emergency_img
        notification_icon.place(x=320, y=550)

        ##text
        notification_text.configure(text='Calling the doctors')
        notification_text.place(x=330, y=670)

    elif label_img == 'normal_act':

        notification_icon.configure(image=normal_img)
        notification_icon.image = normal_img
        notification_icon.place(x=320, y=550)

        ##text
        notification_text.configure(text='Normal Activity')
        notification_text.place(x=330, y=670)






def main_demo_pecs():
    ##get raw data for displaying
    task_skeleton_data = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/pecs_data_review/skeletons_repetitive_behavior_02082017.txt')

    ##HOT features
    HOT_16_subject_6_tasks = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/pecs_data_review/HOT_repetitive_behavior_02082017.txt')

    ##BOW computed on HOT
    # bow_data = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/bow_experiment_data/test_PECS/BOW_3_kmeans_16subject_2sec_without_outlier.txt')
    # labels_bow_data = data_organizer.load_matrix_pickle(
    #     'C:/Users/dario.dotti/Documents/bow_experiment_data/test_PECS/BOW_3_kmeans_labels_16subject_2sec_without_outlier.txt')
    bow_data = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/bow_experiment_data/BOW_30_kmeans_16subject_2sec.txt')
    labels_bow_data = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/bow_experiment_data/BOW_30_kmeans_labels_16subject_2sec.txt')

    lr = LogisticRegression()
    lr.fit(bow_data, np.ravel(labels_bow_data))

    ##cluster data
    # cluster_model = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/bow_experiment_data/test_PECS/cl_30_kmeans_model_2secWindow_without_outliers.txt')
    # labels_cluster = data_organizer.load_matrix_pickle(
    #     'C:/Users/dario.dotti/Documents/bow_experiment_data/test_PECS/cluster_3_kmeans_word__without_outliers.txt')
    cluster_model = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/bow_experiment_data/cl_30_kmeans_model_2secWindow_newVersion.txt')
    labels_cluster = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/bow_experiment_data/cluster_30_kmeans_word_newVersion.txt')
    key_labels = map(lambda x: x[0], labels_cluster)

    ##shared variable between threads
    q = multiprocessing.Queue()
    current_time_shared = multiprocessing.Queue()

    ##launch different processes in the same time
    display_joints_traj = multiprocessing.Process(target=draw_joints_and_tracks,
                                                  args=(task_skeleton_data, current_time_shared))
    display_confidence_classifier = multiprocessing.Process(target=plot_classifier_confidence, args=(
        HOT_16_subject_6_tasks, cluster_model, key_labels, lr, q))


    ##Start threads
    display_joints_traj.start()
    display_confidence_classifier.start()


    ##call plot initializer
    basic_plot()
    update_figures_in_threads(q)


    ##launch main window loop
    window.geometry('800x700')
    window.mainloop()





















if __name__ == '__main__':
    main_demo_pecs()