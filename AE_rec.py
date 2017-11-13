import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import operator
import os
from sklearn.cluster import KMeans,SpectralClustering,MeanShift,AgglomerativeClustering
from collections import Counter
from scipy.spatial.distance import cdist, pdist
from sklearn import decomposition
from scipy.ndimage.filters import gaussian_filter
from sklearn import svm
import random
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle, islice


import data_organizer
import hierarchical_ae_learning_methods as hs
import img_processing


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def AE_showWeights_level2(AE_weights_level_2, AE_weights_level_1):
    hd_weights_level1 = AE_weights_level_1[0][0]
    bias_1_level1 = AE_weights_level_1[1]
    bias_2_level1 = AE_weights_level_1[2]

    hd_weights_level2 = AE_weights_level_2[0][0]
    bias_1_level2 = AE_weights_level_2[1]
    bias_2_level2 = AE_weights_level_2[2]

    ##prepare weights from layer 2
    imgs = np.eye(hd_weights_level2.shape[0])
    imgs = np.dot(hd_weights_level2.T, imgs)

    print imgs.shape

    size_main_img = 56
    n_img = int(np.sqrt(imgs.shape[0]))
    main_img = np.zeros(((size_main_img + 1) * n_img - 1, (size_main_img + 1) * n_img - 1))


    for i,w in enumerate(imgs) :
        r_main_img, c_main_img = divmod(i, n_img)


        n_subimg = 9
        size_subimg = 18

        sub_img = np.zeros(((size_subimg + 1) * n_subimg/3 - 1, (size_subimg + 1) * n_subimg/3 - 1))


        for r_c_counter,n_w in enumerate(xrange(0,len(w),81)):

            sub_weight = w[n_w:(n_w+81)]

            ## reconstruction weights level 1
            rec = np.dot(sub_weight, hd_weights_level1.T)

            ## define the row and cols where to put the sub_imgs
            r, c = divmod(r_c_counter, 3)

            weight_img = rec.reshape((size_subimg, size_subimg))

            ##sharpening edges
            # blurred_f = gaussian_filter(weight_img, sigma=1)
            #
            # filter_blurred_f = gaussian_filter(blurred_f, sigma=0.5)
            # alpha = 30
            # weight_img = blurred_f + alpha * (blurred_f - filter_blurred_f)

            ##add the subimg to the windows
            sub_img[r * (size_subimg + 1):(r + 1) * (size_subimg + 1) - 1,
            c * (size_subimg + 1):(c + 1) * (size_subimg + 1) - 1] = weight_img



        sub_img -= sub_img.min()
        sub_img /= sub_img.max()


        ##add the subimgs to the main window
        main_img[r_main_img * (size_main_img + 1):(r_main_img + 1) * (size_main_img + 1) - 1,
        c_main_img * (size_main_img + 1):(c_main_img + 1) * (size_main_img + 1) - 1] = sub_img



    plt.imshow(main_img.squeeze(), cmap=plt.cm.gray)
    plt.show()


def AE_showWeights_level2_temporalExperiment(AE_weights_level_2, AE_weights_level_1):
    hd_weights_level1 = AE_weights_level_1[0][0]
    bias_1_level1 = AE_weights_level_1[1]
    bias_2_level1 = AE_weights_level_1[2]

    hd_weights_level2 = AE_weights_level_2[0][0]
    bias_1_level2 = AE_weights_level_2[1]
    bias_2_level2 = AE_weights_level_2[2]

    ##prepare weights from layer 2
    imgs = np.eye(hd_weights_level2.shape[0])
    imgs = np.dot(hd_weights_level2.T, imgs)

    print imgs.shape

    size_main_img = 62
    n_img = int(np.sqrt(imgs.shape[0]))
    main_img = np.zeros(((size_main_img + 1) * n_img - 1, (size_main_img + 1) * n_img - 1))


    for i,w in enumerate(imgs) :
        r_main_img, c_main_img = divmod(i, n_img)


        n_subimg = 9
        size_subimg = 20

        sub_img = np.zeros(((size_subimg + 1) * n_subimg/3 - 1, (size_subimg + 1) * n_subimg/3 - 1))


        for r_c_counter,n_w in enumerate(xrange(0,len(w),144)):

            sub_weight = w[n_w:(n_w+144)]

            ## reconstruction weights level 1
            rec = np.dot(sub_weight, hd_weights_level1.T)

            ## define the row and cols where to put the sub_imgs
            r, c = divmod(r_c_counter, 3)

            weight_img = rec.reshape((size_subimg, size_subimg))

            ##sharpening edges
            # blurred_f = gaussian_filter(weight_img, sigma=1)
            #
            # filter_blurred_f = gaussian_filter(blurred_f, sigma=0.5)
            # alpha = 30
            # weight_img = blurred_f + alpha * (blurred_f - filter_blurred_f)

            ##add the subimg to the windows
            sub_img[r * (size_subimg + 1):(r + 1) * (size_subimg + 1) - 1,
            c * (size_subimg + 1):(c + 1) * (size_subimg + 1) - 1] = weight_img



        sub_img -= sub_img.min()
        sub_img /= sub_img.max()


        ##add the subimgs to the main window
        main_img[r_main_img * (size_main_img + 1):(r_main_img + 1) * (size_main_img + 1) - 1,
        c_main_img * (size_main_img + 1):(c_main_img + 1) * (size_main_img + 1) - 1] = sub_img



    plt.imshow(main_img.squeeze(), cmap=plt.cm.gray)
    plt.show()


def reconstruction_AE_weights_level2(raw_features,original_points_grid_level_2,AE_weights_level_1,AE_weights_level_2):
    hd_weights_level1 = AE_weights_level_1[0][0]
    bias_1_level1 = AE_weights_level_1[1]
    bias_2_level1 = AE_weights_level_1[2]

    hd_weights_level2 = AE_weights_level_2[0][0]
    bias_1_level2 = AE_weights_level_2[1]
    bias_2_level2 = AE_weights_level_2[2]


    ##reconstruct layer 2

    hd2_space = np.dot(raw_features, hd_weights_level2)
    activations_l2 = sigmoid_function(hd2_space + bias_1_level2)
    #activations_l2 = relu_function(hd2_space+bias_1_level2)

    rec_space_l2 = np.dot(activations_l2, hd_weights_level2.T)
    rec_level2 = sigmoid_function(rec_space_l2 + bias_2_level2)


    size_w1 = hd_weights_level1.shape[1]
    size_image = 18 * 18


    for i, sample_level2 in enumerate(rec_level2):

        fig = plt.figure()


        n_subimg = 9
        size_subimg = 18

        ##reconstructed sample
        a = fig.add_subplot(1, 2, 2)
        a.set_title('reconstruction')
        sub_rec_img = np.zeros(((size_subimg + 1) * (n_subimg / 3) - 1, (size_subimg + 1) * n_subimg / 3 - 1))

        sub_samples = sample_level2.reshape((9,size_w1))


        for r_c_counter,sample in enumerate(sub_samples):

            rec_space_l1 = np.dot(sample, hd_weights_level1.T)
            rec_level1 = sigmoid_function(rec_space_l1 + bias_2_level1).reshape((size_subimg, size_subimg))

            #plt.imshow(rec_level1, cmap=plt.cm.gray)
            #plt.show()

            ## print mean pixel value
            # a = np.where(rec_level1 > 0.1)
            # mean_rec = 0
            # for i_c in range(0, len(a[0])):
            #     mean_rec = mean_rec + rec_level1[a[0][i_c], a[1][i_c]]
            # print mean_rec / len(a[0])
            ###

            ## define the row and cols where to put the sub_imgs
            r, c = divmod(r_c_counter, 3)

            ##add the subimg to the windows
            sub_rec_img[r * (size_subimg + 1):(r + 1) * (size_subimg + 1) - 1,
            c * (size_subimg + 1):(c + 1) * (size_subimg + 1) - 1] = rec_level1


        imgplot =plt.imshow(sub_rec_img.squeeze(), cmap=plt.cm.gray)
        #cv2.imshow('ciao',sub_rec_img)
        #cv2.waitKey()


        ##original sample
        a = fig.add_subplot(1, 2, 1)
        a.set_title('original')
        orig_img = np.zeros(((size_subimg + 1) * n_subimg / 3 - 1, (size_subimg + 1) * n_subimg / 3 - 1))

        for r_c_counter,n_orig in enumerate(xrange(0,len(original_points_grid_level_2[i]),size_image)):

            sub_orig_img = original_points_grid_level_2[i][n_orig:(n_orig+size_image)].reshape((size_subimg, size_subimg))

            ## define the row and cols where to put the sub_imgs
            r, c = divmod(r_c_counter, 3)

            ##add the subimg to the windows
            orig_img[r * (size_subimg + 1):(r + 1) * (size_subimg + 1) - 1,
            c * (size_subimg + 1):(c + 1) * (size_subimg + 1) - 1] = sub_orig_img


        #filename = 'C:/Users/dario.dotti/Documents/data_for_vocabulary/camera017/traj_pixel_activation/bayesian_net/gt_images' + '/' + str(i) + '.jpg'
        #plt.imsave(filename, orig_img.squeeze(), cmap=plt.cm.gray)

        imgplot =plt.imshow(orig_img.squeeze(),cmap=plt.cm.gray )
        plt.show()


def reconstruction_AE_weights_level2_temporal(raw_features,original_points_grid_level_2,AE_weights_level_1,AE_weights_level_2):
    hd_weights_level1 = AE_weights_level_1[0][0]
    bias_1_level1 = AE_weights_level_1[1]
    bias_2_level1 = AE_weights_level_1[2]

    hd_weights_level2 = AE_weights_level_2[0][0]
    bias_1_level2 = AE_weights_level_2[1]
    bias_2_level2 = AE_weights_level_2[2]

    ##reconstruct layer 2

    hd2_space = np.dot(raw_features, hd_weights_level2)
    activations_l2 = sigmoid_function(hd2_space + bias_1_level2)
    # activations_l2 = relu_function(hd2_space+bias_1_level2)

    rec_space_l2 = np.dot(activations_l2, hd_weights_level2.T)
    rec_level2 = sigmoid_function(rec_space_l2 + bias_2_level2)

    size_w1 = hd_weights_level1.shape[1]
    size_image = 20 * 20

    rec_mse_local = []
    map(lambda (i, row): rec_mse_local.append(np.sum(np.power(row - raw_features[i], 2))), enumerate(rec_level2))
    print 'all data mse: ', np.sum(rec_mse_local) / len(rec_level2)

    for i, sample_level2 in enumerate(rec_level2):

        fig = plt.figure()

        n_subimg = 9
        size_subimg = 20

        ##reconstructed sample
        a = fig.add_subplot(1, 2, 2)
        a.set_title('reconstruction')
        sub_rec_img = np.zeros(((size_subimg + 1) * (n_subimg / 3) - 1, (size_subimg + 1) * n_subimg / 3 - 1))

        sub_samples = sample_level2.reshape((3, size_w1))



        for r_c_counter, sample in enumerate(sub_samples):
            rec_space_l1 = np.dot(sample, hd_weights_level1.T)
            rec_level1 = sigmoid_function(rec_space_l1 + bias_2_level1).reshape((size_subimg, size_subimg))

            # plt.imshow(rec_level1, cmap=plt.cm.gray)
            # plt.show()

            ## print mean pixel value
            # a = np.where(rec_level1 > 0.1)
            # mean_rec = 0
            # for i_c in range(0, len(a[0])):
            #     mean_rec = mean_rec + rec_level1[a[0][i_c], a[1][i_c]]
            # print mean_rec / len(a[0])
            ###

            ## define the row and cols where to put the sub_imgs
            r, c = divmod(r_c_counter, 3)

            ##add the subimg to the windows
            sub_rec_img[r * (size_subimg + 1):(r + 1) * (size_subimg + 1) - 1,
            c * (size_subimg + 1):(c + 1) * (size_subimg + 1) - 1] = rec_level1

        imgplot = plt.imshow(sub_rec_img.squeeze(), cmap=plt.cm.gray)
        # cv2.imshow('ciao',sub_rec_img)
        # cv2.waitKey()


        ##original sample
        a = fig.add_subplot(1, 2, 1)
        a.set_title('original')
        orig_img = np.zeros(((size_subimg + 1) * n_subimg / 3 - 1, (size_subimg + 1) * n_subimg / 3 - 1))

        for r_c_counter, n_orig in enumerate(xrange(0, len(original_points_grid_level_2[i]), size_image)):
            sub_orig_img = original_points_grid_level_2[i][n_orig:(n_orig + size_image)].reshape(
                (size_subimg, size_subimg))

            ## define the row and cols where to put the sub_imgs
            r, c = divmod(r_c_counter, 3)

            ##add the subimg to the windows
            orig_img[r * (size_subimg + 1):(r + 1) * (size_subimg + 1) - 1,
            c * (size_subimg + 1):(c + 1) * (size_subimg + 1) - 1] = sub_orig_img

        # filename = 'C:/Users/dario.dotti/Documents/data_for_vocabulary/camera017/traj_pixel_activation/bayesian_net/gt_images' + '/' + str(i) + '.jpg'
        # plt.imsave(filename, orig_img.squeeze(), cmap=plt.cm.gray)

        imgplot = plt.imshow(orig_img.squeeze(), cmap=plt.cm.gray)
        plt.show()


def plot_images_l1(imgs, loc, title=None, channels=1):
    '''Plot an array of images.
    We assume that we are given a matrix of data whose shape is (n*n, s*s*c) --
    that is, there are n^2 images along the first axis of the array, and each
    image is c squares measuring s pixels on a side. Each row of the input will
    be plotted as a sub-region within a single image array containing an n x n
    grid of images.
    '''
    n = int(np.sqrt(len(imgs)))
    assert n * n == len(imgs), 'images array must contain a square number of rows!'
    s = int(np.sqrt(len(imgs[0]) / channels))
    assert s * s == len(imgs[0]) / channels, 'images must be square!'

    img = np.zeros(((s+1) * n - 1, (s+1) * n - 1, channels), dtype=imgs[0].dtype)
    for i, pix in enumerate(imgs):
        r, c = divmod(i, n)

        weight_img = pix.reshape((s, s, 1))

        ##sharpening edges
        blurred_f = gaussian_filter(weight_img, sigma=1)

        filter_blurred_f = gaussian_filter(blurred_f, sigma=0.5)
        alpha = 30
        weight_img = blurred_f + alpha * (blurred_f - filter_blurred_f)

        img[r * (s+1):(r+1) * (s+1) - 1,
            c * (s+1):(c+1) * (s+1) - 1] = weight_img
    img -= img.min()
    img /= img.max()

    ax = plt.gcf().add_subplot(loc)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.imshow(img.squeeze(), cmap=plt.cm.gray)
    if title:
        ax.set_title(title)


def plot_layers_l1(weights, tied_weights, channels=1):
    '''Create a plot of weights, visualized as "bottom-level" pixel arrays.'''
    #if hasattr(weights[0], 'get_value'):
        #weights = [w.get_value() for w in weights]
    k = min(len(weights), 9)
    imgs = np.eye(weights[0].shape[0])
    for i, weight in enumerate(weights[:-1]):
        imgs = np.dot(weight.T, imgs)
        plot_images_l1(imgs,
                    100 + 10 * k + i + 1,
                    channels=channels,
                    title='Layer {}'.format(i+1))
    weight = weights[-1]
    n = weight.shape[1] / channels
    if int(np.sqrt(n)) ** 2 != n:
        return
    if tied_weights:
        imgs = np.dot(weight.T, imgs)
        plot_images_l1(imgs,
                    100 + 10 * k + k,
                    channels=channels,
                    title='Layer {}'.format(k))
    else:
        plot_images_l1(weight,
                    100 + 10 * k + k,
                    channels=channels,
                    title='Decoding weights')


def AE_reconstruction_level1(raw_features,AE_weights_level_1):

    hd_weights = AE_weights_level_1[0][0]
    bias_1 = AE_weights_level_1[1]
    bias_2 = AE_weights_level_1[2]


    ##compute AE reconstruction
    #hd1_space = np.dot(raw_features, hd_weights)
    activations = sigmoid_function((np.dot(raw_features, hd_weights))+ bias_1)
    #activations = relu_function(hd1_space+ bias_1)


    #rec_space = np.dot(activations, hd_weights.T)
    rec = sigmoid_function((np.dot(activations, hd_weights.T)) + bias_2)
    #rec = relu_function(rec_space + bias_2)


    img_size = 120
    ##calculate rec error over the data
    rec_mse_local = map(lambda (i, row): np.sum(np.power(np.subtract(row, raw_features[i]),2)),enumerate(rec))
    print 'all data mse: ', np.sum(rec_mse_local) / len(rec)


    #print count
    ##visualize original vs reconstruction image

    for i, row in enumerate(rec):
        #if np.sum(np.power(row-raw_features[i],2)) < 3: continue

        fig = plt.figure()

        #temp_orig = raw_features[i].reshape((img_size,img_size))
        ##print mean pixel value
        # a = np.where(temp_orig > 0.001)
        # mean_orig = 0
        # for i_c in range(0,len(a[0])):
        #     mean_orig = mean_orig + temp_orig[a[0][i_c], a[1][i_c]]
        #print mean_orig/len(a[0])
        ####

        img_orig = raw_features[i].reshape((img_size, img_size, 1))
        a = fig.add_subplot(1, 2, 1)
        a.set_title('original')
        imgplot = plt.imshow(img_orig.squeeze(), cmap=plt.cm.gray)

        # temp = row.reshape((img_size, img_size))
        # ## print mean pixel value
        # a = np.where(temp > 0.1)
        # mean_rec = 0
        # for i_c in range(0, len(a[0])):
        #     mean_rec = mean_rec+ temp[a[0][i_c], a[1][i_c]]
        #print mean_rec/len(a[0])
        ###

        img_rec = row.reshape((img_size, img_size, 1))
        a = fig.add_subplot(1, 2, 2)
        a.set_title('reconstruction')
        imgplot = plt.imshow(img_rec.squeeze(), cmap=plt.cm.gray)

        plt.show()


def hid_unit_activation_allLayers(raw_features, AE_weights):

    hd_weights = AE_weights[0][0]
    bias_1_level1 = AE_weights[1]

    #hd2_space = np.dot(raw_features, hd_weights)
    activations_l2 = sigmoid_function((np.dot(raw_features, hd_weights)) + bias_1_level1)

    matrix_activations_all_data = []
    counter = 0
    for i in np.arange(len(activations_l2)):
        if counter % 10000 == 0: print counter, datetime.now().time()
        counter += 1

        hist_activation = np.zeros((1, hd_weights.shape[1]))

        ##speed up version of the for loop using operator.set_item
        map(lambda (i_v, v): operator.setitem(hist_activation, (0, i_v), v), enumerate(activations_l2[i]))

        if len(matrix_activations_all_data) > 0:
            matrix_activations_all_data = np.vstack((matrix_activations_all_data, hist_activation[0]))
        else:
            matrix_activations_all_data = hist_activation

    print matrix_activations_all_data.shape
    return matrix_activations_all_data





def check_rec_in_clusters(matrix_activations_data_l1,pred,AE_weight_layer1, kmean_centers, matrix_act_transf,save_img):


    newpath = 'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/100pca_agglomerative_cluster_layer1_15_clusters_cosine_5sec/'


    for cluster_class in range(15):

        print cluster_class

        ##get activation for the current class
        index_samples = np.where(pred==cluster_class)[0][:5]

        activation_current_class = matrix_activations_data_l1[index_samples]
        activation_current_class_t = matrix_act_transf[index_samples]

        ######take the n closest samples #######
        ##compute the distance between activantion of current cluster and its cluster center
        #d = cdist(activation_current_class_t, kmean_centers[cluster_class].reshape((1,-1)))
        #closest_samples = np.sort(d, axis=0)[:5]
        #index_min = [np.where(d == c_s) for c_s in closest_samples]
        #activation_closest_samples = [activation_current_class[i_m[0][0]] for i_m in index_min]
        #######

        mean_rec = np.zeros((120,120))
        for i_sample,sample_current_class in enumerate(activation_current_class):

            if np.sum(sample_current_class)==0: continue

            sample_current_class = sample_current_class.reshape((50,AE_weight_layer1[0][1].shape[1]))

            for i_s_c in xrange(0,len(sample_current_class),5):
                #rec_space = np.dot(sample_current_class[i_s_c], AE_weight_layer1[0][0].T)
                #rec = sigmoid_function(rec_space + AE_weight_layer1[2]).reshape((120,120))

                ##deep AE
                rec_deep = sigmoid_function((np.dot(sample_current_class[i_s_c],  AE_weight_layer1[0][1].T)) + AE_weight_layer1[3])
                rec = sigmoid_function((np.dot(rec_deep, AE_weight_layer1[0][0].T)) + AE_weight_layer1[4]).reshape((120,120))



                #mean_rec += rec

                ##Show
                # imgplot = plt.imshow(rec.squeeze(), cmap=plt.cm.gray)
                # plt.show()

                ##Save
                if save_img:
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    filename = newpath + str(cluster_class) + '_' +str(i_sample)+ '_' + str(i_s_c)+ '.jpg'
                    # mean_imgs_cluster = cv2.resize(mean_imgs_cluster,(54,54),interpolation= cv2.INTER_LINEAR)
                    plt.imsave(filename, rec.squeeze(), cmap=plt.cm.gray)


def save_n_layer2_example_per_clusters(matrix_activations_data_l2,pred,AE_weights_level_1, AE_weights_level_2,kmean_centers, save_img):

    newpath = 'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/10_cluster_layer2_new/'

    hd_weights_level1 = AE_weights_level_1[0][0]
    bias_1_level1 = AE_weights_level_1[1]
    bias_2_level1 = AE_weights_level_1[2]

    hd_weights_level2 = AE_weights_level_2[0][0]
    bias_1_level2 = AE_weights_level_2[1]
    bias_2_level2 = AE_weights_level_2[2]

    for cluster_class in range(10):
        print 'cluster: ', cluster_class

        ##get activation for the current class
        index_samples = np.where(pred==cluster_class)
        activation_current_class = matrix_activations_data_l2[index_samples]

        ######take the n closest samples #######
        ##compute the distance between activantion of current cluster and its cluster center
        d = cdist(activation_current_class, kmean_centers[cluster_class].reshape(1, 169))
        closest_samples = np.sort(d, axis=0)[:5]
        index_min = [np.where(d == c_s) for c_s in closest_samples]
        activation_closest_samples = [activation_current_class[i_m[0][0]] for i_m in index_min]
        #######

        rec_space_l2 = np.dot(activation_closest_samples, hd_weights_level2.T)
        rec_level2 = sigmoid_function(rec_space_l2 + bias_2_level2)

        size_w1 = hd_weights_level1.shape[1]
        size_image = 20 * 20

        for i, sample_level2 in enumerate(rec_level2):

            #fig = plt.figure()

            n_subimg = 9
            size_subimg = 20

            ##reconstructed sample
            #a = fig.add_subplot(1, 2, 2)
            #a.set_title('reconstruction')
            sub_rec_img = np.zeros(((size_subimg + 1) * (n_subimg / 3) - 1, (size_subimg + 1) * n_subimg / 3 - 1))

            try:
                sub_samples = sample_level2.reshape((3, size_w1))
            except:
                continue


            for r_c_counter, sample in enumerate(sub_samples):
                rec_space_l1 = np.dot(sample, hd_weights_level1.T)
                rec_level1 = sigmoid_function(rec_space_l1 + bias_2_level1).reshape((size_subimg, size_subimg))


                ## define the row and cols where to put the sub_imgs
                r, c = divmod(r_c_counter, 3)

                ##add the subimg to the windows
                sub_rec_img[r * (size_subimg + 1):(r + 1) * (size_subimg + 1) - 1,
                c * (size_subimg + 1):(c + 1) * (size_subimg + 1) - 1] = rec_level1

            #imgplot = plt.imshow(sub_rec_img.squeeze(), cmap=plt.cm.gray)
            #plt.show()
            ##Save
            if save_img:
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                filename = newpath + str(cluster_class) + '_' + str(i) + '.jpg'
                # mean_imgs_cluster = cv2.resize(mean_imgs_cluster,(54,54),interpolation= cv2.INTER_LINEAR)
                sub_rec_img = cv2.flip(sub_rec_img,1)
                plt.imsave(filename, sub_rec_img.squeeze(), cmap=plt.cm.gray)


def visualize_activations(matrix_activation):
    ### Training 625 matrix activation on shallow ae on only arms
    # participant_length = [0, 2197, 2082, 1873, 1595, 1779, 1991, 2148, 1702, 2484, 1744, 2902, 1947, 1860, 1743, 1645,
    #                       2398, 2287, 1998, 1573]
    # s = []
    # dim = 30
    # for l in xrange(1, len(participant_length)):
    #     slide = matrix_activation[participant_length[l - 1]:(participant_length[l - 1] + participant_length[l])]
    #
    #     for m in xrange(0, len(slide) - dim, dim):
    #         if len(s) > 0:
    #             s = np.vstack((s, matrix_activation[m:m + dim].reshape((1, -1))))
    #         else:
    #             s = matrix_activation[m:m + dim].reshape((1, -1))

    ### trained deep AE on upperBody
    participant_length = [0, 2876, 2394, 2256, 1998, 1887, 2597, 2703, 2105, 3137, 2190, 4072, 2226, 2282, 2480, 2120,
                          2536, 2507, 2511, 1675]
    s = []
    dim = 50
    for l in xrange(1, len(participant_length)):
        slide = matrix_activation[participant_length[l - 1]:(participant_length[l - 1] + participant_length[l])]

        for m in xrange(0, len(slide) - dim, dim):
            if len(s) > 0:
                s = np.vstack((s, matrix_activation[m:m + dim].reshape((1, -1))))
            else:
                s = matrix_activation[m:m + dim].reshape((1, -1))
    print s.shape
    #s = np.array(random.sample(matrix_activation, 30000))
    # kernel_bandwith = 5.1
    # X = img_processing.my_mean_shift(s, iterations=5, kernel_bandwith=kernel_bandwith)
    # print datetime.now().time()
    # my_kmean = KMeans(n_clusters=3, n_jobs=-1, algorithm='full')
    # X = my_kmean.fit(s)
    # means = np.mean(X,axis=1)

    pca = decomposition.PCA(n_components=100)  # 2-dimensional PCA whiten=True, svd_solver='randomized'

    s_t = pca.fit(s)
    data_organizer.save_matrix_pickle(s_t,
                                       'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/100pca_deep900225AE_5sec_data.txt')
    s_t = pca.transform(s)
    #print s_t.shape
    print np.sum(pca.explained_variance_ratio_)
    # plt.bar(range(100), pca.explained_variance_ratio_)
    # plt.show()

    ## testing clustering
    #m_s = KMeans(n_clusters=10, n_jobs=-1)
    #m_s = MeanShift(n_jobs=-1,bandwidth=0.9)
    #m_s.fit(s_t)
    #s_t = s
    m_s = AgglomerativeClustering(n_clusters=15, affinity='cosine', linkage='average')
    m_s.fit(s_t)

    y_tr = m_s.fit_predict(s_t)
    print Counter(y_tr)
    ##since agglomerative clustering doesnt have predict I use svm with the cluster labels for classification
    clf = svm.LinearSVC().fit(s_t, y_tr)

    data_organizer.save_matrix_pickle(clf,
                                       'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/linearSVM_agglomerative15c_5sec_100pca.txt')
    #print 'file saved'

    colors = np.array(np.random.randint(0, 255, size=(20, 3))) / 255.0
    color_labels = [colors[p] for p in y_tr]

    ## 2D
    plt.scatter(s_t[:, 0], s_t[:, 1],c = color_labels)
    #plt.scatter(m_s[:, 0], m_s[:, 1], marker='^', c='r')
    plt.show()

    ##3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(s_t[:, 0], s_t[:, 1], s_t[:, 2],c = color_labels)  # s_t[:, 1], s_t[:, 2],s_t[:,0]
    #ax.scatter(m_s[:, 0], m_s.means_[:, 1], m_s.means_[:, 2], marker='^', c='r')
    plt.show()

    return s,s_t,m_s,y_tr


def cluster_activation_allLayers(matrix_activation, AE_weights_layer1, AE_weights_level2, n_layer):

    matrix_activation,matrix_act_transf,cluster_model,pred = visualize_activations(matrix_activation)

    # pred = cluster_model.labels_
    # kmean_centers = []

    #hs.determine_number_k_kMeans(np.array(random.sample(matrix_activation,20000)))
    #my_kmean = KMeans(n_clusters=10,n_jobs=-1,algorithm='full')
    #cluster_model = my_kmean.fit(matrix_activation)
    #data_organizer.save_matrix_pickle(cluster_model,'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/10_cluster_model_layer2_new.txt')
    #cluster_model = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/40_cluster_model_layer2_new.txt')

    ## Testing the model
    #pred = cluster_model.predict(matrix_activation)
    #print Counter(pred).most_common()
    kmean_centers = []#np.array(cluster_model.cluster_centers_)

    #d = cdist(kmean_centers.reshape(10, 625), kmean_centers.reshape(10, 625))

    ## Visualize the clusters
    if n_layer ==1:
        check_rec_in_clusters(matrix_activation, pred, AE_weights_layer1,kmean_centers, matrix_act_transf, save_img=1)
    elif n_layer==2:
        save_n_layer2_example_per_clusters(matrix_activation, pred, AE_weights_layer1, AE_weights_level2,
                                           kmean_centers, save_img=1)

    return pred


def create_cluster_labels_participant_task(raw_features, AE_weights_level_2):

    cluster_model = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/10_cluster_model_layer2_new.txt')

    final_labels = []
    for participant in raw_features:
        participant_label = []

        for task in participant:
            task_label = []

            for f_vector in task:
                hd = np.dot(f_vector, AE_weights_level_2[0][0])
                act = sigmoid_function(hd + AE_weights_level_2[1])

                label = cluster_model.predict(act.reshape((1, -1)))
                task_label.append(label[0])

            participant_label.append(task_label)

        final_labels.append(participant_label)

    data_organizer.save_matrix_pickle(final_labels,
                                      'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/10clusters_labels_l2_participants_tasks_new.txt')



def main():

    ###### Layer 1 #########
    #raw_features = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/posture_shoulder_arms_3fps.txt')
    AE_weights_level_1 = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/deep_AE_900_225_weights_008hd1_noNoise.txt')#625_weights_008hd1


    # ############
    #
    #data_new = [f for part in raw_features for task in part for f in task]
    #data_new= []
    #map(lambda part: map(lambda task: data_new.append(task), part),raw_features)



     ####Visually check weights
    #plot_layers_l1(AE_weights_level_1[0],tied_weights=True)
    #plt.show()


    #plt.show()
    # ##Visually check the reconstruction
    #AE_reconstruction_level1(data_new,AE_weights_level_1)
    #
    # ##create activation matrix
    #matrix_activations_data_l1 = hid_unit_activation_allLayers(data_new, AE_weights_level_1)

    #data_organizer.save_matrix_pickle(matrix_activations_data_l1,'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/matrix_act_400_weights.txt')
    #matrix_activations_data_l1 = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/matrix_act_625_weights_6fps.txt')
    matrix_activations_data_l1 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/matrix_deep_act_225_weights.txt')

    # ####Clustering on the activation matrix
    cluster_activation_allLayers(matrix_activations_data_l1, AE_weights_level_1,[],n_layer=1)
    return 0
    # ################


    #### Layer 2 temporal reconstruction####

    raw_features = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/feature_matrix_participant_task_l2_new.txt')
    AE_weights_level_1 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/ae/head_joint_id1/144weights_l1_hd1002.txt')
    original_points_grid_level_2 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/head_joint_id1/orig_points_participant_task_l2_new.txt')
    AE_weights_level_2 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/ae/head_joint_id1/169weights_l2_001_new.txt')

    data_new = [f for part in raw_features for task in part for f in task]

    ## Visualize weights layer 2 temporal##
    # AE_showWeights_level2_temporalExperiment(AE_weights_level_2, AE_weights_level_1)
    # original_points_grid_level_2_new = [f for part in original_points_grid_level_2 for task in part for f in task]
    # reconstruction_AE_weights_level2_temporal(data_new, original_points_grid_level_2_new, AE_weights_level_1, AE_weights_level_2)

    #### Get activations and cluster ####
    #matrix_activations_data_l2 = hid_unit_activation_allLayers(data_new,AE_weights_level_2)
    #cluster_activation_allLayers(matrix_activations_data_l2, AE_weights_level_1, AE_weights_level_2,n_layer=2 )

    #####


    create_cluster_labels_participant_task(raw_features, AE_weights_level_2)





if __name__ == '__main__':
    main()