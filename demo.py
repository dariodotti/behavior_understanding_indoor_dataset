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
scene = cv2.imread('C:/Users/dario.dotti/Documents/pilot_abnormal_behavior_indoor/subject4_1834.jpg')

#Create a window
window=Tk()
window.title('Abnormal Behavior Detector')


def plot_classifier_confidence(task,cluster_model,keys_labels,lr,q):
    ##plotting confidence classifier  on bow

    hist = np.zeros((1,len(keys_labels)))

    x_axis = 0

    ##every two second create a sample cluster it and create an hist
    for n_slice,two_mins_slice in enumerate(task):

        similar_word= cluster_model.predict(np.array(two_mins_slice).reshape(1,-1))

        index = np.where(similar_word == keys_labels)[0]

        hist[0][index] +=1
        pred =  lr.predict(hist)

        #print pred

        conf_confusion = np.max([lr.decision_function(hist)[0][0],lr.decision_function(hist)[0][1],lr.decision_function(hist)[0][2]])
        conf_repetitive = lr.decision_function(hist)[0][3]
        conf_adl = np.max([lr.decision_function(hist)[0][4],lr.decision_function(hist)[0][5]])


        q.put(['cc',conf_confusion,conf_repetitive,conf_adl,n_slice,x_axis])
        x_axis+=1

        time.sleep(2)

    #plt.pause(100)


def draw_joints_and_tracks(body_points,current_time_shared):
    #make the thread wait for the other
    time.sleep(1.5)

    color = (0,0,255)
    for n_frame,traj_body_joints in enumerate(body_points):

        temp_img = scene.copy()

        #get recording time and make it as current time
        time_info_joint = traj_body_joints[0,1].split(' ')
        date = time_info_joint[4].split('&#')[0]+'-'+'10-'+time_info_joint[2]
        #global current_time
        current_time = dt.strptime(date+' '+time_info_joint[3],'%Y-%m-%d %H:%M:%S')

        current_time_shared.put(current_time)


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

        ##display like recorded time 30 fps
        cv2.imshow('skeleton',temp_img)
        cv2.waitKey(33)


def show_binary_sensor(sensor_data, signal_entrance_door,q,current_time_shared):

    #fake data except maindoor
    activation_open =[0,5,6]
    activation_close=[0,5,5]


    for i,s in enumerate(sensor_data):


        #wait until the current time reach the time the sensor was activated
        while True:

            try:
                current_time = current_time_shared.get()
            except:
                continue

            time_diff= current_time-s

            if  np.abs(time_diff.total_seconds()) <= 10:

                if signal_entrance_door[i][9:12] == 'OFF':
                    activation_open[0]+=1
                else:
                    activation_close[0]+=1

                q.put(['as',activation_open,activation_open])

                break


def basic_plot():#Function to create the base plot, make sure to make global the lines, axes, canvas and any part that you would want to update later

    global ax_conf,ax_as,canvas,rect_open,rect_close,warning_img,emergency_img,notification_icon,notification_text

    ##initialize figures
    main_fig = plt.figure()
    ax_as = main_fig.add_subplot(212)
    ax_conf = main_fig.add_subplot(211)


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
    sensor = ['maindoor','toilet','livingroom']
    ind = np.arange(len(sensor))
    width = 0.2

    ax_as.axis([-0.5,3,0,10])
    ax_as.set_xticks(ind+width)
    ax_as.set_xticklabels(sensor)

    #fake data except maindoor
    activation_open =[0,5,6]
    activation_close=[0,5,5]

    #bar charts
    rect_open = ax_as.bar(ind, activation_open,width,color='red')
    rect_close = ax_as.bar(ind+width, activation_close,width,color='blue')
    ax_as.legend((rect_open[0],rect_close[1]),('door open','door close'),fontsize=9)
    ax_as.set_title('ambient sensor')

    ##initialize notification icons and text
    warning_img = PIL.ImageTk.PhotoImage(PIL.Image.open('C:/Users/dario.dotti/Documents/data_for_demo/icon_warning_call_relative.png'))
    emergency_img = PIL.ImageTk.PhotoImage(PIL.Image.open('C:/Users/dario.dotti/Documents/data_for_demo/icon_emergency_call_doctors.png'))
    notification_icon = Label(window, image=warning_img)
    notification_text = Label(window, text='Calling the stakeholders')

    notification_title = Label(window, text='NOTIFICATION')
    notification_title.place(x=350, y=510)

def update_figures_in_threads(q):

    try:#Try to check if there is data in the queue
        result=q.get_nowait()

        if result !='Q':

            if result[0] == 'cc':

                ax_conf.plot(result[5],result[1],'r^',label='confusion')
                ax_conf.plot(result[5],result[2],'b^',label='repetitive')
                ax_conf.plot(result[5],result[3],'g^',label='normal activity')

                #draw the legend only once
                if result[4]==0:
                    ax_conf.legend(loc='upper left',fontsize=9)

                ##show notification images
                if 1>result[1]>0 or 1>result[2]>0:
                    update_notification_icons('warning')

                elif result[1]>1 or result[2]>1:
                    update_notification_icons('emergency')


                canvas.draw()
                window.after(10, update_figures_in_threads, q)


            elif result[0] == 'as':
                rect_open[0].set_height(result[1][0])
                rect_close[0].set_height(result[2][0])

                canvas.draw()
                window.after(10, update_figures_in_threads, q)
    except:
        ##no new input so refresh
        window.after(100, update_figures_in_threads, q)


def update_notification_icons(label_img):
    ##refreshing notification icons


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


def main_demo():

    ##get raw data for displaying
    body_joints = video_traj.xml_parser('C:/Users/dario.dotti/Documents/pilot_abnormal_behavior_indoor/joints/subject7_points.xml')
    ##HOT features organized per subjects and tasks
    HOT_16_subject_6_tasks = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_demo/HOT_matrix_16_subject_6_tasks.txt')

    ##BOW computed on HOT
    bow_data = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_demo/BOW_16subject_2sec.txt')
    labels_bow_data = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_demo/BOW_labels_16subject_2sec.txt')

    lr = LogisticRegression()
    lr.fit(bow_data,np.ravel(labels_bow_data))

    ##cluster data
    cluster_model= data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/cl_model_2secWindow_band03.txt')
    labels_cluster = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_demo/cluster_labels.txt')
    labels_cluster_counter = Counter(labels_cluster).most_common(40)
    keys_labels = map(lambda x: x[0], labels_cluster_counter)

    ##load binary data for displaying
    entrance_door_str = ambient_sensors.org_data_ID('C:/Users/dario.dotti/Documents/pilot_abnormal_behavior_indoor/binary/18-10-16_sensors_subject7.txt')['entrance']

    entrance_door = []
    signal_entrance_door=[]
    #converting entrance door from string to time
    for i,s in enumerate(entrance_door_str):

        date = s.split(' ')[1]
        time = s.split(' ')[2].split('-')

        entrance_door.append(dt.strptime(date+' '+time[0]+':'+time[1]+':'+time[2],'%y-%m-%d %H:%M:%S'))
        signal_entrance_door.append(s.split(' ')[0])

    subj = 3 #n_subject order: 4,5,6,7,10,11,12,13,14,19,20,15,3,16,17,18
    task = 0 #n_task order confusion: 0,1,2 repetitive:3 house_activity: 4,5


    ##shared variable between threads
    q = multiprocessing.Queue()
    current_time_shared=multiprocessing.Queue()

    ##launch different processes in the same time
    display_joints_traj = multiprocessing.Process(target=draw_joints_and_tracks,args=(body_joints,current_time_shared))
    display_confidence_classifier = multiprocessing.Process(target=plot_classifier_confidence,args=(HOT_16_subject_6_tasks[subj][task],cluster_model,keys_labels,lr,q))
    display_ambient_sensor = multiprocessing.Process(target=show_binary_sensor,args=(entrance_door,signal_entrance_door,q,current_time_shared))


    display_joints_traj.start()
    display_confidence_classifier.start()
    display_ambient_sensor.start()


    ##call plot initializer
    basic_plot()
    update_figures_in_threads(q)


    ##launch main window loop
    window.geometry('800x700')
    window.mainloop()






if __name__ == '__main__':
    main_demo()