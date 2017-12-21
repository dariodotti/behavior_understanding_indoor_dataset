from scipy.spatial.distance import cdist,pdist
import numpy as np
import os
import cv2
from sklearn.cluster import KMeans
from scipy.cluster import  hierarchy
from collections import Counter
from sklearn import svm,linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import cophenet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats.stats import pearsonr

import data_organizer
import hierarchical_ae_learning_methods as hs
import AE_rec



def pca_on_data(n_minute_data):
    from sklearn import decomposition


    pca = decomposition.PCA(n_components=3)
    s_t = pca.fit_transform(n_minute_data)
    # print s_t.shape
    print np.sum(pca.explained_variance_ratio_)
    #plt.bar(range(200), pca.explained_variance_ratio_)
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(s_t[:, 0], s_t[:, 1], s_t[:, 2], c='r')  # s_t[:, 1], s_t[:, 2],s_t[:,0]
    # ax.scatter(m_s[:, 0], m_s.means_[:, 1], m_s.means_[:, 2], marker='^', c='r')
    plt.show()


def bow_on_features(feature_p_n_minute,y_tr,l_per_participant):

    start = 0
    cl_id_p_n_minute = []
    for i_p in range(0, len(feature_p_n_minute)):
        cl_id_p_n_minute.append(y_tr[start:(start + len(feature_p_n_minute[i_p]))])
        start += len(feature_p_n_minute[i_p])

    ## similarity between participants using histograms because of length difference ##
    l = Counter(y_tr).keys()
    p_hist = []
    for p in cl_id_p_n_minute:
        hist = np.zeros((1, len(l)), dtype=int)
        for v in p:
            hist[0, v - 1] += 1
        p_hist.append(hist)

    p_hist = np.concatenate(p_hist, axis=0)
    ## Pearson Correlation to double check##
    # pearson_matrix = np.zeros((46, 46))
    # for i_fp in range(0, len(p_hist)):
    #     for j_fp in range(0, len(p_hist)):
    #         pearson_matrix[i_fp, j_fp] = pearsonr(p_hist[i_fp][0], p_hist[j_fp][0])[0]

    ## delete clusters with less than 3 samples
    # p_hist = np.concatenate(p_hist, axis=0)
    # l_per_participant = np.delete(np.array(l_per_participant), [3, 7, 16, 34, 37, 11, 30, 43, 13, 17, 36])
    # p_hist = np.delete(p_hist, [3, 7, 16, 34, 37, 11, 30, 43, 13, 17, 36], axis=0)

    print p_hist.shape


    X_train, X_test, y_train, y_test = train_test_split(p_hist, l_per_participant, test_size=0.3)
    model = svm.LinearSVC(penalty='l1',dual=False, C=0.01, class_weight={1: 10, 3: .5}).fit(X_train, y_train)  # , gamma=10
    #model = svm.NuSVC(nu=0.05, kernel='linear',gamma=0.001, decision_function_shape='ovr',class_weight={1: 10, 3: .5}).fit(X_train, y_train) #, gamma=10
    y_pred = model.predict(X_test)

    #print accuracy_score(y_test, y_pred)
    print classification_report(y_test, y_pred)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def check_decision_functions_and_save_samples(model,X_train):
    ## see decision function ##
    Z = model.decision_function(X_train)
    #Z= model.predict_proba(X_train)
    AE_weights_level_1 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/deep_AE_900_225_weights_008hd1_noNoise.txt')

    # class_best_example = []
    class_names = ['1', '2', '3']
    for c in range(0, 3):
        print class_names[c]
        indx = np.argsort(Z[:, c])[::-1][:3]

        speed_class= []
        or_class= []
        KA_class=[]
        for idx in indx:
            #print Z[idx]
            # class_best_example.append(X_train[idx])
            hot = X_train[idx][:24]
            keys_area = X_train[idx][24:30]
            posture_5_seconds = X_train[idx][72:].reshape((16, 225))

            speed_class.append(np.sum(hot.reshape((8, 3)), axis=0))
            or_class.append(np.sum(hot.reshape((8, 3)), axis=1))
            KA_class.append(keys_area)

            for p in range(0, len(posture_5_seconds)):
                rec_deep = sigmoid_function(
                    (np.dot(posture_5_seconds[p], AE_weights_level_1[0][1].T)) + AE_weights_level_1[3])
                rec = sigmoid_function((np.dot(rec_deep, AE_weights_level_1[0][0].T)) + AE_weights_level_1[4]).reshape(
                    (120, 120))

                #print np.sum(hot.reshape((8, 3)), axis=0)
                #print keys_area

                filename = 'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/svm_decision_func_fusion/' + \
                           class_names[c] + '_' + str(idx) + '_' + str(p) + '.jpg'
                plt.imsave(filename, rec.squeeze(), cmap=plt.cm.gray)

                # cv2.imshow('posture',rec)
                # cv2.waitKey(0)
        print np.mean(np.array(speed_class),axis=0)
        print np.mean(np.array(KA_class),axis=0)
        print np.mean(np.array(or_class),axis=0)


def raw_feature_classification(n_minute,feature_p_n_minute, l_per_participant):

    ## feature and label vectors with all data ##
    start = 0
    labels_data = np.zeros((n_minute.shape[0],1),dtype=int)
    for i_p in xrange(0, len(feature_p_n_minute)):
        for i in xrange(0,len(feature_p_n_minute[i_p])):
            labels_data[start+i,0] = int(l_per_participant[i_p])

        start += len(feature_p_n_minute[i_p])

    ##

    ### AE posture vs RAW angle posture##
    #hot_key_a = n_minute[:, :30]
    # sk_angles = n_minute[:, 30:72]
    # posture_AE = n_minute[:, 72:]
    #
    # n_minute = np.hstack((hot_key_a, posture_AE))
    # print n_minute.shape
    ####

    ## feature and label vector all participants but one ##
    # r_list=[]
    # start = 0
    # for i_p in xrange(0, len(feature_p_n_minute)):
    #     print '## subject: ',i_p
    #     test_p = n_minute[start:(start+len(feature_p_n_minute[i_p]))]
    #     label_p = labels_data[start:(start+len(feature_p_n_minute[i_p]))]
    #
    #     train_ps = np.vstack((n_minute[:start,:], n_minute[start+len(feature_p_n_minute[i_p]):,:]))
    #     label_ps = np.vstack((labels_data[:start], labels_data[start+len(feature_p_n_minute[i_p]):,:]))
    #
    #     model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, weights='distance').fit(train_ps, label_ps.ravel())
    #     # model = svm.NuSVC(nu=0.5,decision_function_shape='ovr',class_weight={1:10,3:.5}).fit(X_train, y_train)#nu=0.05, ,class_weight={1:10,3:.5}
    #
    #     y_pred = model.predict(test_p)
    #     #print classification_report(label_p, y_pred)
    #     #print accuracy_score(label_p,y_pred)
    #     r = precision_recall_fscore_support(label_p,y_pred,average='weighted')
    #     r_list.append(r[2])
    #
    #
    #     start += len(feature_p_n_minute[i_p])
    #
    # print np.mean(np.array(r_list))
    ###


    ### try 2 clusters classification ##
    # idx = np.where(labels_data==1)
    # labels_data = np.delete(labels_data,idx,axis=0)
    # n_minute = np.delete(n_minute,idx,axis=0)

    # for l in range(0,len(labels_data)):
    #     if labels_data[l]==1:
    #         labels_data[l]=3
    ###########



    #print X_train.shape
    for i in range(0,5):
        X_train, X_test, y_train, y_test = train_test_split(n_minute, labels_data.ravel(), test_size=0.1)
        #model = svm.NuSVC(nu=0.5,decision_function_shape='ovr',class_weight={1:10,3:.5}).fit(X_train, y_train)#nu=0.05, ,class_weight={1:10,3:.5}
        #model = RandomForestClassifier(class_weight={1:10,3:.5}).fit(X_train, y_train)
        model = KNeighborsClassifier(n_neighbors=2, n_jobs=-1,weights='distance').fit(X_train, y_train)
        #data_organizer.save_matrix_pickle(model,'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/svm_2sec_2general_cluster.txt')
        y_pred = model.predict(X_test)

        #print y_pred
        #print y_test
        print 'acc training: ', accuracy_score(y_train[:500], model.predict(X_train[:500]))
        #print precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print classification_report(y_test, y_pred)


    #check_decision_functions_and_save_samples(model,X_train)


def check_samples_in_clusters(data,y_tr, kmean_centers, save_img):

    newpath = 'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/posture/posture_50clusters_allTasks/'

    AE_weights_level_1 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/deep_AE_900_225_weights_008hd1_noNoise.txt')

    for cluster_class in range(50):

        print cluster_class

        ##get activation for the current class
        index_samples = np.where(y_tr == cluster_class)[0]

        activation_current_class = data[index_samples]
        # activation_current_class_t = matrix_act_transf[index_samples]

        ######take the n closest samples #######
        ##compute the distance between activantion of current cluster and its cluster center
        d = cdist(activation_current_class, kmean_centers[cluster_class].reshape((1, -1)))
        closest_samples = np.sort(d, axis=0)[:3]
        index_min = [np.where(d == c_s) for c_s in closest_samples]
        activation_closest_samples = [activation_current_class[i_m[0][0]] for i_m in index_min]
        #######

        mean_rec = np.zeros((120, 120))
        for i_sample, sample_current_class in enumerate(activation_closest_samples):

            if np.sum(sample_current_class) == 0: continue

            sample_current_class = sample_current_class.reshape((16, AE_weights_level_1[0][1].shape[1]))

            for i_s_c in xrange(0, len(sample_current_class)):
                # rec_space = np.dot(sample_current_class[i_s_c], AE_weight_layer1[0][0].T)
                # rec = sigmoid_function(rec_space + AE_weight_layer1[2]).reshape((120,120))

                ##deep AE
                rec_deep = sigmoid_function(
                    (np.dot(sample_current_class[i_s_c], AE_weights_level_1[0][1].T)) + AE_weights_level_1[3])
                rec = sigmoid_function((np.dot(rec_deep, AE_weights_level_1[0][0].T)) + AE_weights_level_1[4]).reshape(
                    (120, 120))

                # mean_rec += rec

                ##Show
                # imgplot = plt.imshow(rec.squeeze(), cmap=plt.cm.gray)
                # plt.show()

                ##Save
                if save_img:
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    filename = newpath + str(cluster_class) + '_' + str(i_sample) + '_' + str(i_s_c) + '.jpg'
                    # mean_imgs_cluster = cv2.resize(mean_imgs_cluster,(54,54),interpolation= cv2.INTER_LINEAR)
                    plt.imsave(filename, rec.squeeze(), cmap=plt.cm.gray)



def dist_skeleton_angles_features(sk_f):
    orientation_intervals = [[range(0, 45)], [range(45, 90)], [range(90, 135)], [range(135, 180)], [range(180, 225)],
                             [range(225, 270)], [range(270, 315)], [range(315, 360)]]

    # labels_angles = np.zeros((sk_f.shape[0], 7), dtype=int)
    # for i_sk,sk in enumerate(sk_f):
    #     sk = sk.reshape((7,6))[:,3]
    #
    #     for i_v, v in enumerate(sk):
    #         for i_o, o in enumerate(orientation_intervals):
    #             if int(v) in o[0]:
    #                 labels_angles[i_sk, i_v] = i_o




    ##  discretize in 8 angles all the data
    sk_f_reshape = sk_f.reshape((sk_f.shape[0] * 7, 6))
    data = sk_f_reshape[:,3]
    #hist_fr = np.zeros((1,int(np.max(data))+1),dtype=int)
    hist_fr = np.zeros((1,len(orientation_intervals)),dtype=int)
    labels_angles = np.zeros((1, data.shape[0]), dtype=int)
    for i_v,v in enumerate(data):
        for i_o, o in enumerate(orientation_intervals):
            if int(v) in o[0]:
                hist_fr[0,i_o] +=1
                labels_angles[0, i_v] = i_o
                break

    print np.mean(data)
    # plt.bar(range(0,hist_fr.shape[1]),hist_fr[0])
    # plt.show()
    ##

    ## reshape every row contains label for each joint ##
    labels_angles = labels_angles.reshape((labels_angles.shape[1] / 7, 7))

    return labels_angles





def main_laban_posture_RAW():
    ## load features from determined tasks
    # features_participants_orig_1 = data_organizer.load_matrix_pickle(
    #     'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/experiment_upperBody_pathPlanning/RAWpostureUpperBody_path_features_2sec_skeletonF_task45.txt')#'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/experiment_upperBody_pathPlanning/pca_RAWpostureUpperBody_path_features.txt')
    # features_participants_orig_2 = data_organizer.load_matrix_pickle(
    #     'C:/Users/dario.dotti/Desktop/data_recordings_master/data_personality/RAWpostureUpperBody_path_features_master_2sec_skeletonF_task45.txt') #'C:/Users/dario.dotti/Desktop/data_recordings_master/data_personality/pca_RAWpostureUpperBody_path_features_master.txt')
    # features_participants_orig = features_participants_orig_1 + features_participants_orig_2
    # del features_participants_orig[44]
    ## load features from all tasks
    features_participants_orig = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/experiment_upperBody_pathPlanning/RAWpostureUpperBody_path_features_2sec_skeletonF_task0123.txt')

    ##

    ## separate feature vector again to see what is more informative ##
    features_participants_3 = np.concatenate(features_participants_orig,axis=0)
    #hot_keyA = features_participants_3[:, :30]
    sk_f = features_participants_3[:, 30:72]
    #posture = features_participants_3[:,72:]
    #print posture.shape
    # #
    # start = 0
    # f_participant = []
    # for i_p in range(0, len(features_participants_orig)):
    #     f_participant.append(posture[start:(start + len(features_participants_orig[i_p]))])
    #     start += len(features_participants_orig[i_p])

    ##

    ## visualize distribution of skeleton features ##
    labels_angles = dist_skeleton_angles_features(sk_f)
    #data_organizer.save_matrix_pickle(labels_angles,'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/posture/Temporal_dynamics_exp/labels_angles_task0123.txt')
    ##


    ### clustering on posture data ###
    ## kmeans
    # posture = posture.reshape(posture.shape[0],posture.shape[1])
    # #hs.determine_number_k_kMeans(posture)
    # my_kmeans = KMeans(n_clusters= 50,n_jobs= -1).fit(posture)
    # y_tr = my_kmeans.predict(posture)
    # cc = my_kmeans.cluster_centers_
    # #check_samples_in_clusters(posture,y_tr,cc,save_img=1)
    # data_organizer.save_matrix_pickle(my_kmeans,
    #                                   'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/posture/kmeans_50_posture_allTasks.txt')
    ##hierarchical cl
    #Z = hierarchy.linkage(posture, method='average', metric='euclidean')
    # check if metric preserve original distance
    #c, coph_dists = cophenet(Z, pdist(posture))
    #print c
    #y_tr = hierarchy.fcluster(Z, 10,criterion="distance") #cosine = 0.5
    #
    # ##print y_tr
    #print Counter(y_tr)
    ##

    #### classification on data #######
    #n_minute = np.concatenate(features_participants_orig, axis=0)
    #
    #l_per_participant  = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/clusters_on_pearson_corr_personality_scores.txt')
    #
    # extrovert_label_per_p =[2, 0, 1, 0, 2, 2, 1, 1, 2, 1, 2, 1, 0, 1, 1, 1, 0, 2, 2, 2, 2, 1, 1, 2, 2, 0, 2, 0, 1, 0, 0, 1, 1, 2, 1, 0, 2, 1, 0, 1, 0, 0, 1, 2, 0, 1]
    # consc_label_per_p = [2, 2, 1, 2, 2, 2, 2, 1, 0, 2, 2, 0, 1, 0, 1, 2, 1, 0, 2, 0, 1, 0, 2, 1, 2, 1, 0, 1, 0, 2, 0, 2, 2, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 2]
    # nevrotic_label_per_p = [0, 2, 1, 1, 1, 2, 2, 1, 0, 0, 2, 2, 2, 1, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 1, 0]
    #
    #
    # ## baseline classifications
    #raw_feature_classification(n_minute, features_participants_orig, l_per_participant)

    #bow_on_features(f_participant, y_tr, l_per_participant)

    ##


def main_laban_posture_ID():
    cl_model_posture = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/posture/kmeans_50_posture_allTasks.txt')

    features_participants_orig = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/experiment_upperBody_pathPlanning/RAWpostureUpperBody_path_features_2sec_skeletonF_ALLTASKS.txt')  # 'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/experiment_upperBody_pathPlanning/pca_RAWpostureUpperBody_path_features.txt')
    # features_participants_orig_2 = data_organizer.load_matrix_pickle(
    #     'C:/Users/dario.dotti/Desktop/data_recordings_master/data_personality/RAWpostureUpperBody_path_features_master_2sec_skeletonF_ALLTASKS.txt')  # 'C:/Users/dario.dotti/Desktop/data_recordings_master/data_personality/pca_RAWpostureUpperBody_path_features_master.txt')
    # features_participants_orig = features_participants_orig_1 + features_participants_orig_2

    l_per_participant = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/clusters_on_pearson_corr_personality_scores.txt')



    ## separate feature vector again to see what is more informative ##
    features_participants_3 = np.concatenate(features_participants_orig, axis=0)
    # hot_keyA = features_participants_3[:, :30]
    skeleton_angles = features_participants_3[:, 30:72]
    posture = features_participants_3[:, 72:]
    print posture.shape

    ## feature and label vectors with all data ##
    start = 0
    labels_data = np.zeros((posture.shape[0], 1), dtype=int)
    for i_p in xrange(0, len(features_participants_orig)):
        for i in xrange(0, len(features_participants_orig[i_p])):
            labels_data[start + i, 0] = int(l_per_participant[i_p])

        start += len(features_participants_orig[i_p])

    labels_data = labels_data.ravel()
    y_tr = cl_model_posture.predict(posture)


    ## visualize posture distribution ##
    hist = np.zeros((3, 50))
    for pers_label in xrange(1, 4):
        for i_l in xrange(0, len(y_tr)):
            if labels_data[i_l] == pers_label: hist[(pers_label - 1), y_tr[i_l]] += 1
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.25
    ind = np.arange(0, 50)

    ##normalized heights
    rect1 = plt.bar(ind, hist[0,:]/len(np.where(labels_data==1)[0]), width)
    rect2 = plt.bar(ind+width, hist[1, :]/len(np.where(labels_data==2)[0]), width, color='red')
    rect3 = plt.bar(ind+(width*2), hist[2, :]/len(np.where(labels_data==3)[0]), width, color='green')

    ax.legend((rect1[0], rect2[0], rect3[0]), ('class1', 'class2','class3'), fontsize=11)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(ind)
    plt.show()
    ###

    ## check posture differences using skeleton raw angles ##
    for pers_label in xrange(1,4):
        print pers_label
        angles_per_cl = []
        for i_l in xrange(0,len(y_tr)):
            if labels_data[i_l] == pers_label:
                angles_per_cl.append(skeleton_angles[i_l])

        ## show mean per joints ##
        ## matrix 7 joints x 6 stat of angles
        angles_per_cl = np.array(angles_per_cl).reshape((7*len(angles_per_cl),6))

        for i_joint in xrange(7):
            joint_stat = []
            for i_value in xrange(i_joint,angles_per_cl.shape[0],7):
                joint_stat.append(angles_per_cl[i_value])
            print np.mean(np.array(joint_stat),axis=0)



if __name__ == '__main__':
    #main_laban_posture_RAW()
    main_laban_posture_ID()
