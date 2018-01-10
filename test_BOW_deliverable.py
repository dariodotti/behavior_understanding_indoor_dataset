import numpy as np
<<<<<<< HEAD
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
=======
from sklearn.metrics import classification_report,precision_recall_fscore_support,accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pomegranate import *

>>>>>>> 9348384985d2847c272133ff77ce6181ca1fa082

import data_organizer
import img_processing
import AE_rec



feature_p_1 = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Desktop/Hier_AE_deliverable/head_joint_id1/feature_matrix_participant_task_l2_new_realCoordinates.txt')
<<<<<<< HEAD
feature_p_2 =data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Desktop/Hier_AE_deliverable/head_joint_id1/feature_matrix_participant_master_task_l2_new_realCoordinates.txt')


feature_p = feature_p_1[:19]+feature_p_2[:27]
real_coord = feature_p_1[19:]+feature_p_2[27:]
list_poly = img_processing.divide_image(np.zeros((414,512),dtype=np.uint8))


space_features = []
for i_p in xrange(0,len(real_coord)):
    for i_task in xrange(0,len(real_coord[i_p])):
        for i_slice in  xrange(0,len(real_coord[i_p][i_task])):
            votes = np.zeros((16, 3))

            for p in xrange(0,len(real_coord[i_p][i_task][i_slice])):
                size_per_frame = int(len(real_coord[i_p][i_task][i_slice][p])/3)
                x  = real_coord[i_p][i_task][i_slice][p][:size_per_frame]
                y =  real_coord[i_p][i_task][i_slice][p][size_per_frame:(size_per_frame*2)]
                z = real_coord[i_p][i_task][i_slice][p][(size_per_frame*2):]


                for i_area, areas in enumerate(list_poly):
                    for i_point in xrange(0,len(x)):
                        if areas.contains_point((int(x[i_point]), int(y[i_point]))):
                            if z[i_point] < (4.232 - (1.433 * 2)):
                                votes[i_area,0] +=1
                            elif z[i_point] > (4.232 - (1.433 * 2)) and z[i_point]  < (4.232 - 1.433):
                                votes[i_area, 1] += 1
                            elif z[i_point] > (4.232 - 1.433):
                                votes[i_area, 2] += 1
            if len(space_features)>0:
                space_features = np.vstack((space_features,normalize(votes.reshape((1,-1)))))
            else:
                space_features = normalize(votes.reshape((1,-1)))


=======
feature_p_2 = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Desktop/Hier_AE_deliverable/head_joint_id1/feature_matrix_participant_master_task_l2_new_realCoordinates.txt')
#cl_model = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Desktop/Hier_AE_deliverable/40_cluster_model_layer2_new.txt')

feature_p = feature_p_1[:46]+feature_p_2[:46]
real_coord = feature_p_1[46:]+feature_p_2[46:]

print len(feature_p),len(real_coord)
print feature_p[45]
print real_coord[0]
>>>>>>> 9348384985d2847c272133ff77ce6181ca1fa082

AE_weights_level_2 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Desktop/Hier_AE_deliverable/ae/head_joint_id1/169weights_l2_001_new.txt')

data_new = [f for part in feature_p for task in part for f in task]

<<<<<<< HEAD
matrix_activations_data_l2 = AE_rec.hid_unit_activation_allLayers(data_new,AE_weights_level_2)

print matrix_activations_data_l2.shape,space_features.shape

#matrix_activations_data_l2 = np.hstack((matrix_activations_data_l2, space_features))

cluster_labels = KMeans(n_clusters=30).fit_predict(matrix_activations_data_l2)
n_labels = len(np.unique(cluster_labels))

start = 0
#cl_id_task = []
hist_matrix = []
for i_p in xrange(0, len(feature_p)):
    for n_task in xrange(len(feature_p[i_p])):
        hist_areas = np.zeros((1,n_labels*48),dtype=int)
        #hist= np.zeros((1,n_labels),dtype=int)

        for i_l,l in enumerate(cluster_labels[start:(start + len(feature_p[i_p][n_task]))]):
            s = space_features[start+i_l]
            idx_max = np.where(s==np.max(s))[0][0]
            hist_areas[0,(idx_max*n_labels)+l] += 1
            #hist[0,l] +=1



        hist_matrix.append(hist_areas)
        #cl_id_task.append(cluster_labels[start:(start + len(feature_p[i_p][n_task]))])
        start += len(feature_p[i_p][n_task])
hist_matrix = np.concatenate(hist_matrix,axis=0)

print hist_matrix.shape

=======

matrix_activations_data_l2 = AE_rec.hid_unit_activation_allLayers(data_new,AE_weights_level_2)

my_kmean = KMeans(n_clusters=20)
cluster_labels = my_kmean.fit_predict(matrix_activations_data_l2)
#cluster_labels = cl_model.predict(matrix_activations_data_l2)


n_labels = len(np.unique(cluster_labels))

start = 0
cl_id_task = []
hist_matrix = []
for i_p in xrange(0, len(feature_p)):
    for n_task in xrange(len(feature_p[i_p])):
        hist= np.zeros((1,n_labels))
        for l in cluster_labels[start:(start + len(feature_p[i_p][n_task]))]:
            hist[0,l] +=1
        hist_matrix.append(hist)
        #cl_id_task.append(cluster_labels[start:(start + len(feature_p[i_p][n_task]))])
        #start += len(feature_p[i_p][n_task])
hist_matrix = np.concatenate(hist_matrix,axis=0)


### BOW ###
>>>>>>> 9348384985d2847c272133ff77ce6181ca1fa082
task_labels=[]
for i_p in xrange(0, len(feature_p)):
    for n_task in xrange(len(feature_p[i_p])):
        task_labels.append(n_task)

for t in xrange(0,len(task_labels)):
    if task_labels[t] == 1:
        task_labels[t] = 0
    if task_labels[t]== 4:
        task_labels[t]= 5

<<<<<<< HEAD

# idx = np.where(np.array(task_labels) == 1)
# task_labels = np.delete(task_labels,idx)
# hist_matrix = np.delete(hist_matrix,idx,axis=0)
#
# idx = np.where(np.array(task_labels) == 0)
# task_labels = np.delete(task_labels,idx)
# hist_matrix = np.delete(hist_matrix,idx,axis=0)

for i in range(0,5):
    X_train, X_test, y_train, y_test = train_test_split(hist_matrix, task_labels, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1,weights='distance').fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print classification_report(y_test, y_pred)
=======
#
# # idx = np.where(np.array(task_labels) == 1)
# # task_labels = np.delete(task_labels,idx)
# # hist_matrix = np.delete(hist_matrix,idx,axis=0)
#
#
# for i in range(0,5):
#     X_train, X_test, y_train, y_test = train_test_split(hist_matrix, task_labels, test_size=0.1)
#     model = KNeighborsClassifier(n_neighbors=2, n_jobs=-1,weights='distance').fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     print classification_report(y_test, y_pred)





## feature and label vector all participants but one ##
r_list=[]
start = 0
for i_p in xrange(0, len(feature_p)):
    print '## subject: ',i_p
    test_p = hist_matrix[start:(start+len(feature_p[i_p]))]
    label_p = task_labels[start:(start+len(feature_p[i_p]))]

    train_ps = np.vstack((hist_matrix[:start,:], hist_matrix[start+len(feature_p[i_p]):,:]))
    label_ps = task_labels[:start]+task_labels[start+len(feature_p[i_p]):]#np.vstack((task_labels[:start], task_labels[start+len(feature_p[i_p]):]))

    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, weights='distance').fit(train_ps, np.array(label_ps).ravel())
    # model = svm.NuSVC(nu=0.5,decision_function_shape='ovr',class_weight={1:10,3:.5}).fit(X_train, y_train)#nu=0.05, ,class_weight={1:10,3:.5}

    y_pred = model.predict(test_p)
    print y_pred,label_p
    print classification_report(label_p, y_pred)
    #print accuracy_score(label_p,y_pred)
    #r = precision_recall_fscore_support(label_p,y_pred,average='weighted')


    start += len(feature_p[i_p])

print np.mean(np.array(r_list))





>>>>>>> 9348384985d2847c272133ff77ce6181ca1fa082





<<<<<<< HEAD
=======



###

# task_1 = []
# for i in xrange(3,len(cl_id_task)-2,6):
#     task_1.append(cl_id_task[i])
#
#
# bayes_matrix = []
# for t in task_1:
#     for i in range(0,len(t)-2,3):
#         bayes_matrix.append([t[i],t[i+1],t[i+2]])
#
# bayes_matrix = np.array(bayes_matrix)
#
#
# model = BayesianNetwork.from_samples(bayes_matrix)#, algorithm='chow-liu'
#
# print model.structure
#
# for s in bayes_matrix:
#     print model.predict_proba({'0':s[0], '1':s[1]})
#     print model.predict([[s[0],s[1],None]])
#     print s
#     a = 1
>>>>>>> 9348384985d2847c272133ff77ce6181ca1fa082
