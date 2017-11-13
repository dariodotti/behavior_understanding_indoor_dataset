import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.path as mplPath
from math import atan2,degrees,isnan
from scipy.ndimage.filters import gaussian_filter
from collections import defaultdict
from datetime import datetime

import main as main_camera017



#Divide image into regions seen as polygons
#Private variables
__slice_col = 4
__slice_row = 4
__slice_depth = 3


def get_slice_cols():
    return __slice_col


def get_slice_rows():
    return __slice_row

def get_slice_depth():
    return __slice_depth


def divide_image(my_room):
    #How to draw polygons: 1 left bottom
                        # 2 left top
                        # 3 right top
                        # 4 right bottom
    slice_cols = __slice_col
    slice_rows = __slice_row


    list_poly = []

    #check whether the img is divisble by the given slices
    # if my_room.shape[0]%slice_rows != 0 or my_room.shape[1]%slice_cols != 0:
    #     print 'img not divisible by the given number of slices'
    #     return list_poly

    n_row = range(106,my_room.shape[0],my_room.shape[0]/slice_rows)

    n_col = range(0,my_room.shape[1],my_room.shape[1]/slice_cols)


    for r in range(0,len(n_row)):
        for c in range(0,len(n_col)):

            poly = mplPath.Path(np.array([[n_col[c],n_row[r]+(my_room.shape[0]/slice_rows)],\
                                    [n_col[c],n_row[r]],\
                                    [n_col[c]+(my_room.shape[1]/slice_cols),n_row[r]],\
                                    [n_col[c]+(my_room.shape[1]/slice_cols),n_row[r]+(my_room.shape[0]/slice_rows)]]))

            list_poly.append(poly)


            ##check poly are correct
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # patch = patches.PathPatch(poly, facecolor='orange', lw=2)
            # ax.add_patch(patch)
            #
            # ax.set_xlim(0,my_room.shape[1])
            # ax.set_ylim(0,my_room.shape[0])
            #
            # plt.show()



    return list_poly

def get_coordinate_points(occurance, joint_id):


    xs = map(lambda line: float(line[joint_id][0]), occurance)
    ys = map(lambda line: float(line[joint_id][1]), occurance)
    zs = map(lambda line: float(line[joint_id][2]),occurance)
    ids =map(lambda line: np.int64(line[0][2]),occurance)

    x_f = gaussian_filter(xs, 3)
    y_f = gaussian_filter(ys,3)

    xs_int = map(lambda x_i: int(x_i) ,x_f)
    ys_int = map(lambda y_i: int(y_i) ,y_f)

    return xs_int,ys_int,zs,ids

def traj_points_filter(list_points):

    x = np.zeros(len(list_points))
    y = np.zeros(len(list_points))

    for n in range(0,len(list_points)):
        x[n] = list_points[n][0]
        y[n] = list_points[n][1]


    #y1 = signal.medfilt(y,3)
    #x1 = signal.medfilt(x,3)

    y1 = gaussian_filter(y,3)
    x1 = gaussian_filter(x, 3)

    # # plot the results
    # plt.subplot(2,1,1)
    # plt.plot(x,y,'o')
    # plt.title('input trajectory')
    #
    #
    # plt.subplot(2,1,2)
    # plt.plot(x1,y1,'o')
    # plt.title('filtered trajectory')
    # plt.xlabel('time')
    #
    # plt.show()

    return x1,y1


def histogram_oriented_tracklets(cube):
    ##remember to change intervals according to the dataset!!!

    orientation_intervals=[[range(0,45)],[range(45,90)],[range(90,135)],[range(135,180)],[range(180,225)],[range(225,270)],\
                           [range(270,315)],[range(315,360)]]
    magnitude_intervals = [[range(0,2)],[range(2,10)],[range(10,200)]]

    hot_matrix = np.zeros((len(orientation_intervals),len(magnitude_intervals)))

    cube = np.array(cube)

    #list_magn = []
    if len(list(set(cube[:,2]))) == 1:

        #k = cube[0,2]

        step = 2


        for i in xrange(0,len(cube)-step):

            dx =abs(float(cube[i+step,0]) - float(cube[i,0]))
            dy = abs(float(cube[i+step,1]) - float(cube[i,1]))

            orientation = int(degrees(atan2(dy,dx))%360)
            magn = int(np.sqrt((np.power(dx,2)+np.power(dy,2))))

            #list_magn.append(magn)

            for c_interval,o_interval in enumerate(orientation_intervals):
                if orientation in o_interval[0]:
                    if magn in magnitude_intervals[0][0]:
                        hot_matrix[c_interval][0] +=1
                        break
                    elif magn in magnitude_intervals[1][0]:
                        hot_matrix[c_interval][1] +=1
                        break
                    elif magn in magnitude_intervals[2][0]:
                        hot_matrix[c_interval][2] +=1
                        break
            ##control whether the values are in the intervals
            if hot_matrix.sum() == 0:
                print 'orientation or magn not in the intervals'
                print orientation,magn

    ##visualize magnitude values
    # if len(list_magn)>0:
    #     print np.max(list_magn)
    #     plt.plot(range(0,len(list_magn)),list_magn)
    #     plt.show()


    return hot_matrix.reshape((len(orientation_intervals)*len(magnitude_intervals)))


def get_velocity_curvature_acceleration(cube,need_correction):

    step = 2
    if len(cube) > step:

        cube = np.array(cube)

        if len(list(set(cube[:,2]))) >= 1:

            k = cube[0,2]


            v_x = np.zeros((len(cube)-step))
            v_y = np.zeros((len(cube)-step))
            velocity_array = np.zeros((len(cube)-step))

            for i in xrange(0,len(cube)-step):
                if need_correction:
                    dx =abs(float(cube[i+step,0]) - float(cube[i,0]))*2
                    dy = abs(float(cube[i+step,1]) - float(cube[i,1]))*2
                else:
                    dx =abs(float(cube[i+step,0]) - float(cube[i,0]))
                    dy = abs(float(cube[i+step,1]) - float(cube[i,1]))

                magn = np.sqrt((np.power(dx,2)+np.power(dy,2)))


                v_x[i] = dx/step
                v_y[i] = dy/step


                velocity_array[i]= int(magn/step)


            acceleration_array = np.zeros((len(velocity_array)))
            a_x = np.zeros((len(velocity_array)))
            a_y =  np.zeros((len(velocity_array)))

            for i in range(0,len(velocity_array)-1):
                a_x[i] = v_x[i+1] - v_x[i]
                a_y[i] = v_y[i+1] - v_y[i]

                acceleration_array[i]= int((velocity_array[i+1]-velocity_array[i]))/step


            velocity_curvature_acceleration_matrix = np.hstack((np.mean(velocity_array),np.mean(acceleration_array)))

            curvature_array = np.zeros((len(velocity_array)))
            for i in range(0,len(velocity_array)-1):
                num=abs((v_x[i]*a_y[i])-(v_y[i]*a_x[i]))
                den = np.power(np.power(v_x[i],2)+np.power(v_y[i],2),3/2)
                #avoid nan value due to 0/0
                if  num != 0 and den  != 0:

                    curvature_array[i] = np.mean(num / den)

                else:
                    curvature_array[i] = 0

            velocity_curvature_acceleration_matrix= np.hstack((velocity_curvature_acceleration_matrix,np.mean(curvature_array)))

            for value in velocity_curvature_acceleration_matrix:
                if isnan(value):
                    print 'WARNING nan vlaue in velocity-accelaration-curvature function'

    else:
        velocity_curvature_acceleration_matrix = [0,0,0]

    #print velocity_curvature_acceleration_matrix

    return velocity_curvature_acceleration_matrix


def histogram_oriented_tracklets_plus_curvature(cube):

    orientation_intervals=[[range(0,45)],[range(45,90)],[range(90,135)],[range(135,180)],[range(180,225)],[range(225,270)],\
                           [range(270,315)],[range(315,360)]]
    magnitude_intervals = [[range(0,10)],[range(10,40)],[range(40,81)]]

    curvature_intervals = [0.1,0.8,2]

    chot_matrix = np.zeros((len(orientation_intervals),len(magnitude_intervals)+len(curvature_intervals)+1))

    cube = np.array(cube)

    if len(list(set(cube[:,2]))) == 1:
        #k = cube[0,2]
        step = 2


        for i in xrange(0,len(cube)-step-1):

            dx =abs(float(cube[i+step,0]) - float(cube[i,0]))
            dy = abs(float(cube[i+step,1]) - float(cube[i,1]))

            orientation = int(degrees(atan2(dy,dx))%360)
            magn = int(np.sqrt((np.power(dx,2)+np.power(dy,2))))

            #compute curvature
            vx = dx/step
            vy = dy/step
            ax = (abs(float(cube[i+step+1,0]) - float(cube[i,0]))/step) - vx
            ay = (abs(float(cube[i+step+1,1]) - float(cube[i,1]))/step) - vy

            num=abs((vx*ay)-(vy*ax))
            den = np.power(np.power(vx,2)+np.power(vy,2),3/2)
            #avoid nan value due to 0/0
            if  num != 0 and den  != 0:

                curvature = np.mean(num / den)

            else:
                curvature = 0

            #print curvature

            for c_interval,o_interval in enumerate(orientation_intervals):
                if orientation in o_interval[0]:
                    if magn in magnitude_intervals[0][0]:
                        chot_matrix[c_interval][0] +=1
                    elif magn in magnitude_intervals[1][0]:
                        chot_matrix[c_interval][1] +=1
                    elif magn in magnitude_intervals[2][0]:
                        chot_matrix[c_interval][2] +=1

            #add curvature to the hist
            if curvature <curvature_intervals[0]:
                chot_matrix[c_interval][3] += 1
            elif curvature >= curvature_intervals[0] and curvature < curvature_intervals[1]:
                chot_matrix[c_interval][4] += 1
            elif curvature >= curvature_intervals[1] and curvature < curvature_intervals[2]:
                chot_matrix[c_interval][5] += 1
            else:
                chot_matrix[c_interval][6] += 1

            ##control whether the values are in the intervals
            if chot_matrix.sum() == 0:
                print 'orientation or magn not in the intervals'
                print orientation,magn


    return chot_matrix.reshape((len(orientation_intervals)*(len(magnitude_intervals)+len(curvature_intervals)+1)))


def display_trajectories(temp_img, list_poly, x, y):

    ## draw slices
    for p in range(0,len(list_poly)):
       cv2.rectangle(temp_img,(int(list_poly[p].vertices[1][0]),int(list_poly[p].vertices[1][1])),\
                   (int(list_poly[p].vertices[3][0]),int(list_poly[p].vertices[3][1])),0,1)

    # draw filtered points
    for p in range(0, len(x)):
       cv2.circle(temp_img, (int(x[p]), int(y[p])), 2, int(zs[p] * scale), -1)

    cv2.imshow('lab_room',temp_img)
    cv2.waitKey(0)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

##### my Meanshift #########

def my_gaussian_kernel(d, bw):
    return np.exp(-0.5*(d/bw)**2) / (bw*np.sqrt(2*np.pi))

def my_mean_shift(data,iterations,kernel_bandwith):

    X = np.copy(data)# Need to copy the data, don't want to modify the originals

    for it in xrange(iterations):
        print it
        for i, x in enumerate(X):
            #if i% 1000 ==0: print datetime.now().time()

            #dist = np.sqrt(((x-X)**2).sum(1))
            weight = my_gaussian_kernel(np.sqrt(np.sum((x-X)**2,axis=1)),kernel_bandwith)#2.5

            X[i] = np.sum(np.dot(np.expand_dims(weight,1).T,X),axis=0) / np.sum(weight)

    return X # we return the shifted points


def my_mean_shift_vect(data,iterations,kernel_bandwith):

    X = np.copy(data).astype(dtype='float16') # Need to copy the data, don't want to modify the originals

    for it in xrange(iterations):

        expd = np.expand_dims(X, 2)  # need another dimension...
        tiled = np.tile(expd, X.shape[0])  # ...to tile up the vectors

        dist = np.sqrt(((np.transpose(X) - tiled) ** 2).sum(1))
        weight = my_gaussian_kernel(dist, kernel_bandwith)  # 2.5

        summed_weight = np.sum(weight, 0)  # the array of the summed weights, for normalize the weighted vectors
        shifted_pts = np.dot(weight,X) / np.expand_dims(summed_weight, 1)  # the normalized vectors



    return shifted_pts





def bin_points(X, bin_size, min_bin_freq):
    bin_sizes = defaultdict(int)
    for point in X:
        binned_point = np.cast[np.int32](point / bin_size)
        bin_sizes[tuple(binned_point)] += 1

    bin_seeds = np.array([point for freq,point in enumerate(bin_sizes) if freq >= min_bin_freq], dtype=np.float32)
    bin_seeds = bin_seeds * bin_size
    return bin_seeds
##############