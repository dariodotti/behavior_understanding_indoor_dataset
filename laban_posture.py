from scipy.spatial.distance import cdist,pdist
import numpy as np
import cv2
from sklearn.cluster import KMeans,SpectralClustering,MeanShift,AgglomerativeClustering
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
    #Z = model.decision_function(X_train)
    #data_organizer.save_matrix_pickle(Z,'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/decision_function_svm.txt')

    Z= model.predict_proba(X_train)
    #Z= data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/decision_function_svm.txt')
    AE_weights_level_1 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/deep_AE_900_225_weights_008hd1_noNoise.txt')

    # class_best_example = []
    class_names = ['1', '2', '3']
    for c in range(0, 3):
        print class_names[c]
        indx = np.argsort(Z[:, c])[::-1][:500]

        speed_class= []
        or_class= []
        KA_class=[]
        sk_angles_list= []
        for idx in indx:
            #print Z[idx]
            # class_best_example.append(X_train[idx])
            hot = X_train[idx][:24]
            keys_area = X_train[idx][24:30]

            posture_5_seconds = X_train[idx][72:].reshape((16, 225))

            speed_class.append(np.sum(hot.reshape((8, 3)), axis=0))
            or_class.append(np.sum(hot.reshape((8, 3)), axis=1))
            KA_class.append(keys_area)
            sk_angles_list.append(np.array(X_train[idx][30:72]).reshape((7,6)))

            # for p in range(0, len(posture_5_seconds)):
            #     rec_deep = sigmoid_function(
            #         (np.dot(posture_5_seconds[p], AE_weights_level_1[0][1].T)) + AE_weights_level_1[3])
            #     rec = sigmoid_function((np.dot(rec_deep, AE_weights_level_1[0][0].T)) + AE_weights_level_1[4]).reshape(
            #         (120, 120))
            #
            #     #print np.sum(hot.reshape((8, 3)), axis=0)
            #     #print keys_area
            #
            #     filename = 'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/svm_decision_func_fusion/' + \
            #                class_names[c] + '_' + str(idx) + '_' + str(p) + '.jpg'
            #     plt.imsave(filename, rec.squeeze(), cmap=plt.cm.gray)

                # cv2.imshow('posture',rec)
                # cv2.waitKey(0)
        print np.mean(np.array(speed_class),axis=0)
        print np.mean(np.array(KA_class),axis=0)
        print np.mean(np.array(or_class),axis=0)
        a = np.mean(np.array(sk_angles_list),axis=0)
        for i in a:
            for ii in i:
                print ("%.4f" % ii),




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
    # hot_key_a = n_minute[:, :30]
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
    #for i in range(0,5):
    X_train, X_test, y_train, y_test = train_test_split(n_minute, labels_data.ravel(), test_size=0.1)
    #model = svm.NuSVC(nu=0.5,decision_function_shape='ovr',class_weight={1:10,3:.5}).fit(X_train, y_train)#nu=0.05, ,class_weight={1:10,3:.5}
    #model = RandomForestClassifier(class_weight={1:10,3:.5}).fit(X_train, y_train)
    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1,weights='distance').fit(X_train, y_train)
    #data_organizer.save_matrix_pickle(model,'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/svm_2sec_2general_cluster.txt')
    y_pred = model.predict(X_test)

    #print y_pred
    #print y_test
    print 'acc training: ', accuracy_score(y_train[:500], model.predict(X_train[:500]))
    #print precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print classification_report(y_test, y_pred)


    check_decision_functions_and_save_samples(model,X_train)












def main_laban_posture_RAW():
    features_participants_orig_1 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/experiment_upperBody_pathPlanning/RAWpostureUpperBody_path_features_2sec_skeletonF.txt')#'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/experiment_upperBody_pathPlanning/pca_RAWpostureUpperBody_path_features.txt')
    features_participants_orig_2 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Desktop/data_recordings_master/data_personality/RAWpostureUpperBody_path_features_master_2sec_skeletonF.txt') #'C:/Users/dario.dotti/Desktop/data_recordings_master/data_personality/pca_RAWpostureUpperBody_path_features_master.txt')
    features_participants_orig = features_participants_orig_1 + features_participants_orig_2

    ##

    ## separate feature vector again to see what is more informative ##
    #features_participants_3 = np.concatenate(features_participants_orig,axis=0)
    # hot = features_participants_3[:, :24]
    # key_areas = features_participants_3[:,24:30]
    #posture = features_participants_3[:,30:130]

    # #
    # start = 0
    # f_participant = []
    # for i_p in range(0, len(features_participants_orig)):
    #     f_participant.append(posture[start:(start + len(features_participants_orig[i_p]))])
    #     start += len(features_participants_orig[i_p])

    ##

    ##concatenate features to form n_minute feature vectors
    # t = 3
    # feature_p_n_minute=[]
    # for p in features_participants_orig:
    #     n_minute_p = []
    #     for n_slice in range(0, len(p) - (t-1), t/2):
    #         n_minute_p.append(p[n_slice:(n_slice + t)].reshape((1,-1)))
    #     feature_p_n_minute.append(np.concatenate(n_minute_p,axis=0))

    feature_p_n_minute = features_participants_orig
    n_minute = np.concatenate(feature_p_n_minute,axis=0)

    #print n_minute.shape

    #pca_on_data(n_minute)

    ### clustering on data ###
    #hs.determine_number_k_kMeans(n_minute)

    #Z = hierarchy.linkage(n_minute, method='average', metric='euclidean')
    # check if metric preserve original distance
    #c, coph_dists = cophenet(Z, pdist(n_minute))
    #print c
    #y_tr = hierarchy.fcluster(Z, 5,criterion="distance") #cosine = 0.5
    #
    # ##print y_tr
    #print Counter(y_tr)
    # data_organizer.save_matrix_pickle(y_tr,'C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/label_features_clustering.txt')


    #y_tr = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/label_features_clustering.txt')
    ##

    #### classification on data #######
    l_per_participant  = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_2_data_recording/clusters_on_pearson_corr_personality_scores.txt')

    extrovert_label_per_p =[2, 0, 1, 0, 2, 2, 1, 1, 2, 1, 2, 1, 0, 1, 1, 1, 0, 2, 2, 2, 2, 1, 1, 2, 2, 0, 2, 0, 1, 0, 0, 1, 1, 2, 1, 0, 2, 1, 0, 1, 0, 0, 1, 2, 0, 1]
    consc_label_per_p = [2, 2, 1, 2, 2, 2, 2, 1, 0, 2, 2, 0, 1, 0, 1, 2, 1, 0, 2, 0, 1, 0, 2, 1, 2, 1, 0, 1, 0, 2, 0, 2, 2, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 2]
    nevrotic_label_per_p = [0, 2, 1, 1, 1, 2, 2, 1, 0, 0, 2, 2, 2, 1, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 1, 0]


    ## baseline classifications
    raw_feature_classification(n_minute, feature_p_n_minute, l_per_participant)

    #bow_on_features(f_participant, y_tr, l_per_participant)

    ##



def main_laban_posture_ID():
    features_participants_orig = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/data_for_personality_exp/after_data_cleaning/posture_data/upperBody/experiment_upperBody_pathPlanning/IDpostureUpperBody_path_features.txt')

    #features_participants = np.concatenate(features_participants_orig, axis=0)
    #n_minute = features_participants[:,:30]
    #posture_l = features_participants[:,30:]

    ## compare the  first n minute ##
    n_minute = [np.array(p[:20]).reshape((1,-1)) for p in features_participants_orig]
    n_minute = np.concatenate(n_minute,axis=0)
    n_minute_similarity = np.concatenate([cdist(p[:20].reshape((1,-1)), n_minute, 'cosine') for p in features_participants_orig],axis=0)
    ##


    Z = hierarchy.linkage(n_minute, method='average', metric='cosine')
    #check if metric preserve original distance
    c, coph_dists = cophenet(Z, pdist(n_minute))
    #print c
    y_tr = hierarchy.fcluster(Z, 0.2, criterion="distance")


    #y_tr = m_s.(n_minute)
    #print y_tr
    #print Counter(y_tr)


    # X_train, X_test, y_train, y_test = train_test_split(n_minute,posture_l.reshape((-1,1)),test_size=0.1)
    # ##since agglomerative clustering doesnt have predict I use svm with the cluster labels for classification
    # clf = svm.LinearSVC().fit(X_train, y_train)
    # y_prediction = clf.predict(X_test)
    # print accuracy_score(y_test,y_prediction)








if __name__ == '__main__':
    #main_laban_posture_ID()
    main_laban_posture_RAW()