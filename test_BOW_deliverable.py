import numpy as np
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import data_organizer
import img_processing
import AE_rec



feature_p_1 = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Desktop/Hier_AE_deliverable/head_joint_id1/feature_matrix_participant_task_l2_new_realCoordinates.txt')
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



AE_weights_level_2 = data_organizer.load_matrix_pickle(
        'C:/Users/dario.dotti/Desktop/Hier_AE_deliverable/ae/head_joint_id1/169weights_l2_001_new.txt')

data_new = [f for part in feature_p for task in part for f in task]

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

task_labels=[]
for i_p in xrange(0, len(feature_p)):
    for n_task in xrange(len(feature_p[i_p])):
        task_labels.append(n_task)

for t in xrange(0,len(task_labels)):
    if task_labels[t] == 1:
        task_labels[t] = 0
    if task_labels[t]== 4:
        task_labels[t]= 5


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





