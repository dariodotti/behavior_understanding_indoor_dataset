import matplotlib.pyplot as plt
import numpy as np


def bar_plot_occupancy_selectedAreas_over_time(data):
    last_col = np.array(data).shape[1]-1
    hs = data[:,last_col:]
    data = data[:,:last_col]

    ##definition of semantic areas
    ##TODO: retrieve the data from external file
    door_areas=[23,25,39,41,55,56,57,72,73]
    desks_areas= [35,37,45,51,53,61]

    working_h = [9,10,11,12,13,14,15,16,17]

    class_freq_per_hour = np.zeros([len(working_h),2])


    for i,hist_allcubes in enumerate(data):
        array_pos = int(hs[i])-working_h[0]

        for n_cube,value_singlecube in enumerate(hist_allcubes):
            if n_cube in door_areas:
                class_freq_per_hour[array_pos,0]= class_freq_per_hour[array_pos,0]+value_singlecube
            elif n_cube in desks_areas:
                class_freq_per_hour[array_pos,1]=class_freq_per_hour[array_pos,1]+value_singlecube

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.25
    ind = np.arange(9,9+len(working_h))
    rect1 = ax.bar(ind,class_freq_per_hour[:,0],width,color='red')
    rect2 = ax.bar(ind+width,class_freq_per_hour[:,1],width)

    ## add a legend
    ax.legend( (rect1[0], rect2[0]), ('Door area', 'Desks area'),fontsize=11 )

    # axes and labels
    ax.set_xlim(9-width,9+len(ind)+width)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(ind)

    plt.show()


def bar_plot_motion_over_time(data):
    last_col = np.array(data).shape[1]-1
    hs = data[:,last_col:]
    data = data[:,:last_col]

    ##definition of semantic areas
    ##TODO: retrieve the data from external file
    #door_areas=[23,25,39,41,55,56,57,72,73]
    #desks_areas= [35,37,45,51,53,61]

    working_h = [9,10,11,12,13,14,15,16,17,18]

    magn_per_hour = np.zeros([len(working_h),3])

    for i,hist_allcubes in enumerate(data):
        hist_allcubes = hist_allcubes.reshape((np.array(data).shape[1]/24,24))
        array_pos = int(hs[i])-working_h[0]

        for n_cube,value_singlecube in enumerate(hist_allcubes):
            value_singlecube = value_singlecube.reshape((8,3))
            magns_cube = value_singlecube.sum(axis= 0)

            #print magns_cube[0],magns_cube[1],magns_cube[2]

            magn_per_hour[array_pos,0] = magn_per_hour[array_pos,0]+magns_cube[0]
            magn_per_hour[array_pos,1] = magn_per_hour[array_pos,1]+magns_cube[1]
            magn_per_hour[array_pos,2] = magn_per_hour[array_pos,2]+magns_cube[2]


    print np.sum(magn_per_hour,axis=0)

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.25
    ind = np.arange(9,9+len(working_h))
    rect1 = ax.bar(ind,magn_per_hour[:,0],width,color='red')
    rect2 = ax.bar(ind+width,magn_per_hour[:,1],width)
    rect3 = ax.bar(ind+(width*2),magn_per_hour[:,2],width,color='green')

    ## add a legend
    ax.legend( (rect1[0], rect2[0],rect3[0]), ('stationary', 'slight mov', 'mov'),fontsize=11 )

    # axes and labels
    ax.set_xlim(9-width,9+len(ind)+width)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(ind)

    plt.show()


def pie_plot_motion_day(data):
    last_col = np.array(data).shape[1]-1
    hs = data[:,last_col:]
    data = data[:,:last_col]

    desks_areas= [35,37,45,51,53,61]


    #motion divided in 3 groups
    motion = np.zeros((1,3))
    time_counter = 0
    for i,hist_allcubes in enumerate(data):
        hist_allcubes = hist_allcubes.reshape((np.array(data).shape[1]/24,24))

        for n_cube,value_singlecube in enumerate(hist_allcubes):
            value_singlecube = value_singlecube.reshape((8,3))
            magns_cube = value_singlecube.sum(axis= 0)


            motion[0,0] = motion[0,0]+magns_cube[0]
            motion[0,1] = motion[0,1]+magns_cube[1]
            motion[0,2] = motion[0,2]+magns_cube[2]
            # if n_cube in desks_areas:
            #     motion[0,0] = motion[0,0]+magns_cube[0]
            #     motion[0,1] = motion[0,1]+magns_cube[1]
            #     motion[0,2] = motion[0,2]+magns_cube[2]


    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels = 'stationary', 'slight mov', 'fast mov'
    colors = ['yellowgreen', 'gold', 'lightskyblue']
    pie_slice_size = [float(i)/np.sum(motion[0]) for i in motion[0]]



    ax.pie(pie_slice_size,labels=labels, colors=colors,autopct='%1.1f%%', shadow=True)
    plt.axis('equal')
    plt.show()

    return pie_slice_size


def bar_plot_motion_in_region_over_long_time(motion_week):
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']


    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.5
    ind = np.arange(0,len(days))
    plt.bar(ind,motion_week,width,color='blue')

    ax.set_xticklabels(days)
    ax.set_xticks(ind+0.2)


    plt.show()


def plot_ambient_sensor_over_time(sensor_data):
    print 'plots ambient sensors over time'

    fig = plt.figure()
    ax = fig.add_subplot(111)

    time_in_seconds_off=[]
    time_in_seconds_on = []
    markers=[]
    colors=[]
    for e in sensor_data:
        raw_time=e.split(' ')[2]
        minutes = raw_time.split('-')[1]
        seconds = raw_time.split('-')[2]

        ##converts in seconds

        if e[9:12] == 'OFF':
            time_in_seconds_off.append((int(minutes)*60) + int(seconds))

            markers.append((5,2))
            colors.append('red')
        else:
            time_in_seconds_on.append((int(minutes)*60) + int(seconds))
            markers.append((5,2))
            colors.append('blue')

    width = 5
    fix_y= np.ones((len(time_in_seconds_off)))

    rect_open =ax.bar(np.array(time_in_seconds_off),fix_y,width,color='r')

    fix_y= np.ones((len(time_in_seconds_on)))
    rect_close = ax.bar(np.array(time_in_seconds_on)+width,fix_y,width,color='b')

    ax.set_ylim(0,1.2)

    ax.legend( (rect_open[0], rect_close[0]), ('Door open', 'Door close'),fontsize=11 )

    #ax.set_xticklabels()

    # for x,y,m,c in zip(time_in_seconds_off+time_in_seconds_on,[1,1,1,1,1,1,1,1,1,1,1],markers,colors):
    #     plt.scatter(x,y,marker=m,s=80,c=c)
    #
    plt.show()


def plot_mean_joints_displacement(mean_displacement_list):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ##TODO: joints from hetra are called by ID and not by name
    joints = ['head','neck','spine mid','spine base','spine shoulder','shoulder R','elbow R','wrist R','hand R','shoulder L','elbow L',\
              'wrist L','hand L','hip R', 'knee R','ankle R','foot R','hip L','knee L','ankle L','foot L']

    for mean_d,joint_name in zip(mean_displacement_list,joints):

        ax.scatter(1,mean_d)

        ax.annotate(joint_name, xy=(1,mean_d), xycoords='data',size=8)

    plt.show()


def plot_single_joint_displacement_vs_filtered_points(my_joint_raw,my_joint_filtered):

    ##plot as frequency
    plt.plot(my_joint_raw,color='b',label='raw points')
    plt.plot(my_joint_filtered,color='r',label='filtered points')
    plt.title('subject 7')
    plt.show()


def plot_ambient_sensor_activation_day(activation_matrix):

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)


    width = 0.3

    sensor = ['maindoor','toilet','livingroom']

    ind = np.arange(0,len(sensor))
    plt.bar(ind,np.sum(activation_matrix,axis=0),width,color='blue')

    ax.set_xticklabels(sensor)
    ax.set_xticks(ind+0.15)


    plt.show()


def bar_plot_ambient_sensor_more_days(activation_matrix):

    days = ['Mon','Tue','Wed','Thu','Fri','Sat']

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)


    width = 0.3

    sensor = ['maindoor','toilet','livingroom']

    ind = np.arange(0,len(days))
    rect1 = plt.bar(ind,activation_matrix[:,0],width,color='blue')
    rect2 = plt.bar(ind+width,activation_matrix[:,1],width,color='green')
    rect3 = plt.bar(ind+(width*2),activation_matrix[:,2],width,color='red')

    ## add a legend
    ax.legend( (rect1[0], rect2[0],rect3[0]), ('maindoor', 'toilet', 'livingroom'),fontsize=11 )

    # axes and labels
    #ax.set_xlim(9-width,9+len(ind)+width)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(days)



    plt.show()