from math import atan2,pi
import cv2
import numpy as np
import matplotlib.path as mplPath
import matplotlib.pyplot as plt

import img_processing

def myround_ofdirection(x, base=45):
    return int(base * round(float(x)/base))


def get_directions_traj(xs,ys):

    list_orientation = []

    dx = float(xs[0]) - float(xs[int(len(xs)/2)])
    dy = float(ys[0]) - float(ys[int(len(ys)/2)])

    ##take care of points that are too close
    if dx <=1 and dy <= 1:
        dx = float(xs[0]) - float(xs[int(len(xs)-1)])
        dy = float(ys[0]) - float(ys[int(len(ys)-1)])

        ##I round the angle to 8 key directions: 0 left, 45 up-left, 90 up, 135 up-right, 180 right, -135 down-right, -90 down, -45 down-left
        list_orientation.append(myround_ofdirection(atan2(dy, dx) / pi * 180))
    else:

        ##I round the angle to 8 key directions: 0 left, 45 up-left, 90 up, 135 up-right, 180 right, -135 down-right, -90 down, -45 down-left
        list_orientation.append(myround_ofdirection(atan2(dy, dx) / pi * 180))

        dx = float(xs[int(len(xs)/2)]) - float(xs[len(xs)-1])
        dy = float(ys[int(len(ys)/2)]) - float(ys[len(ys)-1])

        ##I round the angle to 8 key directions: 0 left, 45 up-left, 90 up, 135 up-right, 180 right, -135 down-right, -90 down, -45 down-left
        t = myround_ofdirection(atan2(dy, dx) / pi * 180)
        list_orientation.append(t)

    return list_orientation


def create_grid(xs, ys, size_mask, directions, scene):
    ## The reason i dont use the already existing function is because this grid doesnt have to be centered at one point but
    ## the corner has to start next to the first point of the trajectory
    margin = 1

    first_chunck_direction = directions[0]

    ###Drawing the central rect and the one next to it

    if first_chunck_direction == -45:

        up_right_corner = [xs[0] + margin, ys[0] - margin]

        first_rect = mplPath.Path(
            np.array([[up_right_corner[0] - size_mask, up_right_corner[1] + size_mask],
                      [up_right_corner[0] - size_mask, up_right_corner[1]],
                      up_right_corner,
                      [up_right_corner[0], up_right_corner[1] + size_mask]]))



    elif first_chunck_direction == -135:

        up_left_corner = [xs[0] - margin, ys[0] - margin]

        first_rect = mplPath.Path(
            np.array([[up_left_corner[0], up_left_corner[1] + size_mask],
                      up_left_corner,
                      [up_left_corner[0]+size_mask, up_left_corner[1]],
                      [up_left_corner[0]+size_mask, up_left_corner[1] + size_mask]]))




    elif first_chunck_direction == 45:

        down_right_corner = [xs[0] + margin, ys[0] + margin]

        first_rect = mplPath.Path(
            np.array([[down_right_corner[0]-size_mask,down_right_corner[1]],
                      [down_right_corner[0]-size_mask,down_right_corner[1]-size_mask],
                      [down_right_corner[0],down_right_corner[1]-size_mask],
                      down_right_corner]))



    elif first_chunck_direction == 135:

        down_left_corner = [xs[0] - margin, ys[0] + margin]

        first_rect = mplPath.Path(
            np.array([down_left_corner,
                      [down_left_corner[0],down_left_corner[1]-size_mask],
                      [down_left_corner[0]+size_mask,down_left_corner[1]-size_mask],
                      [down_left_corner[0]+size_mask,down_left_corner[1]]]))



    ####if direction is straight, i center the rect on the points

    elif first_chunck_direction == -90:

        top_left_corner = [xs[0] - int(size_mask/2),ys[0]-margin]
        top_right_corner = [xs[0] + int(size_mask/2),ys[0]-margin]

        first_rect = mplPath.Path(
            np.array([[top_left_corner[0],top_left_corner[1]+size_mask],
                      top_left_corner,
                      top_right_corner,
                      [top_right_corner[0],top_right_corner[1]+size_mask]]))


    elif first_chunck_direction == 90:

        down_left_corner = [xs[0] - int(size_mask/2),ys[0] + margin]
        down_right_corner = [xs[0] + int(size_mask/2),ys[0] + margin]

        first_rect = mplPath.Path(
            np.array([down_left_corner,
                      [down_left_corner[0],down_left_corner[1] - size_mask],
                      [down_right_corner[0],down_right_corner[1] - size_mask],
                      down_right_corner]))

    elif first_chunck_direction == 180 or first_chunck_direction == -180 :
        top_left_corner = [xs[0] - margin,ys[0] - int(size_mask/2)]
        down_left_corner = [xs[0] - margin,ys[0] + int(size_mask/2)]

        first_rect = mplPath.Path(
            np.array([down_left_corner,
                      top_left_corner,
                      [top_left_corner[0]+size_mask, top_left_corner[1]],
                      [top_left_corner[0]+size_mask,top_left_corner[1]+size_mask]]))

    elif first_chunck_direction == 0:

        top_right_corner = [xs[0] + margin, ys[0] - int(size_mask / 2)]
        down_right_corner = [xs[0] + margin, ys[0] + int(size_mask / 2)]

        first_rect = mplPath.Path(
            np.array([[down_right_corner[0] - size_mask, down_right_corner[1]],
                      [top_right_corner[0] - size_mask, top_right_corner[1]],
                      top_right_corner,
                      down_right_corner]))




    list_rect = [first_rect]#,side_rect,second_side_rect]


    # for rect in list_rect:
    #     cv2.rectangle(scene, (int(rect.vertices[1][0]), int(rect.vertices[1][1])),
    #                   (int(rect.vertices[3][0]), int(rect.vertices[3][1])), (0, 0, 0))
    #
    # for i_p, p in enumerate(xrange(len(xs))):
    #     cv2.circle(scene, (int(xs[p]), int(ys[p]) ), 1, (255, 0, 0), -1)
    #
    # cv2.imshow('scene', scene)
    # cv2.waitKey(0)

    return list_rect


def createLineIterator(P1, P2, img):

    # Produces and array that consists of the coordinates and intensities of each pixel in a line between two points
    #
    # Parameters:
    #     -P1: a numpy array that consists of the coordinate of the first point (x,y)
    #     -P2: a numpy array that consists of the coordinate of the second point (x,y)
    #     -img: the image being processed
    #
    # Returns:
    #     -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])

   #define local variables for readability
   imageH = img.shape[0]
   imageW = img.shape[1]
   P1X = P1[0]
   P1Y = P1[1]
   P2X = P2[0]
   P2Y = P2[1]

   #difference and absolute difference between points
   #used to calculate slope and relative location between points
   dX = P2X - P1X
   dY = P2Y - P1Y
   dXa = np.abs(dX)
   dYa = np.abs(dY)

   #predefine numpy array for output based on distance between points
   itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
   itbuffer.fill(np.nan)

   #Obtain coordinates along the line using a form of Bresenham's algorithm
   negY = P1Y > P2Y
   negX = P1X > P2X
   if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
   elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
   else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

   #Remove points outside of image
   colX = itbuffer[:,0]
   colY = itbuffer[:,1]
   itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

   #Get intensities from img ndarray
   itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

   return itbuffer


def transform_traj_in_pixel_activation(rect_list, x_untilNow, y_untilNow, size_mask, step):
    #size_mask= 18
    #a = [0, 0]
    #b = [size_mask - 5, size_mask - 5]
    #step = np.sqrt(np.power((b[0] - a[0]), 2) + np.power((b[1] - a[1]), 2))


    traj_features = []
    orig_points = []


    for rect in rect_list:

        points_in_mask = []
        map(lambda ci: points_in_mask.append([int(x_untilNow[ci]), int(y_untilNow[ci])]) if rect.contains_point(
            (int(x_untilNow[ci]), int(y_untilNow[ci]))) else False, xrange(len(x_untilNow)))


        origin_mask = [rect.vertices[1][0],rect.vertices[1][1]]

        mask_img = np.zeros((size_mask, size_mask), dtype=np.uint8)
        mask_matrix = np.zeros((size_mask, size_mask))


        if len(points_in_mask) >= 2:
            for i in xrange(len(points_in_mask) - 1):
                distance_metric_value = np.sqrt(np.power(points_in_mask[i + 1][0] - points_in_mask[i][0], 2) + np.power(
                    points_in_mask[i + 1][1] - points_in_mask[i][1], 2)) / step

                ##convert img points to mask coordinate systems
                x_1 = points_in_mask[i + 1][0] - origin_mask[0]
                y_1 = points_in_mask[i + 1][1] - origin_mask[1]
                x = points_in_mask[i][0] - origin_mask[0]
                y = points_in_mask[i][1] - origin_mask[1]

                ##get all pixels lying on the line that pass between two points
                points_on_line = createLineIterator(np.array([[x], [y]]), \
                                                                np.array([[x_1], [y_1]]), mask_img)

                ##fill these pixel values with average distance between two points
                for p in points_on_line:
                    ##if we want to display on img

                    mask_img[int(p[1]),int(p[0])] = 255
                    # # print distance_metric_value
                    # if int(p[1]) + 1 < size_mask - 1 and int(p[0]) < size_mask - 1:
                    #     ##right
                    #     mask_img[int(p[1]) + 1, int(p[0])] = 255
                    #     ##left
                    #     mask_img[int(p[1]) - 1, int(p[0])] = 255
                    #     ##up
                    #     mask_img[int(p[1]), int(p[0]) - 1] = 255
                    #     ##down
                    #     mask_img[int(p[1]), int(p[0]) + 1] = 255


                    ##real value
                    mask_matrix[int(p[1]), int(p[0])] = distance_metric_value
                    # print distance_metric_value
                    if int(p[1]) + 1 < size_mask and int(p[0]) +  1 < size_mask:
                        if int(p[0]) - 1 <0:
                            p[0] = 1
                        if int(p[1]) - 1 <0:
                            p[1] = 1

                        ##right
                        mask_matrix[int(p[1]) + 1, int(p[0])] = distance_metric_value
                        ##left
                        mask_matrix[int(p[1]) - 1, int(p[0])] = distance_metric_value
                        ##up
                        mask_matrix[int(p[1]), int(p[0]) - 1] = distance_metric_value
                        ##down
                        mask_matrix[int(p[1]), int(p[0]) + 1] = distance_metric_value




            ##if we want to display on img
            #plt.imshow(mask_matrix.squeeze(), cmap=plt.cm.gray)
            #plt.show()

        else:
            mask_matrix = mask_matrix.reshape((1, -1))


        ##store final matrix
        if len(traj_features) > 0:
            traj_features = np.vstack((traj_features, mask_matrix.reshape((1, -1))))
        else:
            traj_features = mask_matrix.reshape((1, -1))

        ##store original points
        if len(orig_points)>0:
            orig_points = np.vstack((orig_points,mask_img.reshape((1,-1))))
        else:
            orig_points = mask_img.reshape((1,-1))


    return traj_features,orig_points


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def AE_reconstruction_level1(raw_features,AE_weights_level_1):

    hd_weights = AE_weights_level_1[0][0]
    bias_1 = AE_weights_level_1[1]
    bias_2 = AE_weights_level_1[2]

    ##compute AE reconstruction
    hd1_space = np.dot(raw_features, hd_weights)
    activations = sigmoid_function(hd1_space+ bias_1)


    rec_space = np.dot(activations, hd_weights.T)
    rec = sigmoid_function(rec_space+ bias_2)



    img_size = 18
    ##calculate rec error over the data
    rec_mse_local = []
    map(lambda (i, row):rec_mse_local.append( np.sum(np.power(row-raw_features[i],2))),enumerate(rec))
    print 'all data mse: ', np.sum(rec_mse_local)/len(rec)


    #print count
    ##visualize original vs reconstruction image

    for i, row in enumerate(rec):
        #if np.sum(np.power(row-raw_features[i],2)) < 3: continue

        fig = plt.figure()

        temp_orig = raw_features[i].reshape((img_size,img_size))
        ##print mean pixel value
        # a = np.where(temp_orig > 0.001)
        # mean_orig = 0
        # for i_c in range(0,len(a[0])):
        #     mean_orig = mean_orig + temp_orig[a[0][i_c], a[1][i_c]]
        # print mean_orig/len(a[0])
        ####

        img_orig = raw_features[i].reshape((img_size, img_size, 1))
        a = fig.add_subplot(1, 2, 1)
        a.set_title('original')
        imgplot = plt.imshow(img_orig.squeeze(), cmap=plt.cm.gray)

        temp = row.reshape((img_size, img_size))
        ## print mean pixel value
        a = np.where(temp > 0.1)
        mean_rec = 0
        for i_c in range(0, len(a[0])):
            mean_rec = mean_rec+ temp[a[0][i_c], a[1][i_c]]
        print mean_rec/len(a[0])
        ###

        img_rec = row.reshape((img_size, img_size, 1))
        a = fig.add_subplot(1, 2, 2)
        a.set_title('reconstruction')
        imgplot = plt.imshow(img_rec.squeeze(), cmap=plt.cm.gray)

        plt.show()