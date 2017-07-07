import os
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans,MeanShift
from collections import Counter
from sklearn.preprocessing import normalize

import data_organizer

my_ms = MeanShift(n_jobs=-1,bandwidth=0.3)

def as_classification_experiment(AS_data):
    #AS_data = np.array(data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/AS_activation_6_labels.txt'))
    n_label = 6

    test_label = [0,1,2,3,4,5]
    tot_accuracy = 0

    # one_leave_out_training = []
    # one_leave_out_test = []
    # one_leave_out_labels = []

    for t in xrange(0,len(AS_data),n_label):
        matrix_test = AS_data[t:(t+n_label)]

        AS_matrix_unordered = np.vstack((AS_data[:t],AS_data[(t+n_label):]))


        matrix_training = np.zeros((AS_matrix_unordered.shape[0],AS_matrix_unordered.shape[1]))



        for task in range(0,n_label):
                for row in range(0,AS_matrix_unordered.shape[0],n_label):
                    c = row+task

                    matrix_training[row] = AS_matrix_unordered[c]

        #labels
        labels =  np.zeros((AS_matrix_unordered.shape[0],1))
        b = 0
        e = len(AS_matrix_unordered)/n_label

        for i in range(0,n_label):

            for row in range(b,e):
                labels[row] = i

            b += (len(AS_matrix_unordered)/n_label)
            e += (len(AS_matrix_unordered)/n_label)

    #     one_leave_out_training.append(matrix_training)
    #     one_leave_out_test.append(matrix_test)
    #     one_leave_out_labels.append(np.ravel(labels))
    # return one_leave_out_training,one_leave_out_test,one_leave_out_labels

        ##classification
        lr = LogisticRegression()

        lr.fit(matrix_training,np.ravel(labels))

        pred =  lr.predict(matrix_test)

        accuracy = 0
        for i in range(0,len(pred)):
            if i == 0 or i == 1:
                if pred[i] == 0 or pred[i]==1:
                    accuracy +=1
                    continue
            if pred[i] == test_label[i]:
                accuracy +=1


        tot_accuracy += float(accuracy)/6

    print tot_accuracy/(len(AS_data)/n_label)


def video_classification_experiments(HOT_matrix):

    #HOT_matrix = np.array(data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/hot_spatial_grid_4x4x3_6_labels.txt'))
    #HOT_matrix = np.array(data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/hot_spatial_grid_4x4_5_labels.txt'))

    n_label = 6
    test_label = [0,1,2,3,4,5]
    tot_accuracy = 0

    # one_leave_out_training = []
    # one_leave_out_test = []
    # one_leave_out_labels = []

    for t in xrange(0,len(HOT_matrix),n_label):
        #print t,t+n_label


        matrix_test = HOT_matrix[t:(t+n_label)]

        HOT_matrix_unordered = np.vstack((HOT_matrix[:t],HOT_matrix[(t+n_label):]))


        matrix_training = np.zeros((HOT_matrix_unordered.shape[0],HOT_matrix_unordered.shape[1]))

        for task in range(0,n_label):
            for row in range(0,HOT_matrix_unordered.shape[0],n_label):
                c = row+task

                matrix_training[row] = HOT_matrix_unordered[c]


        #labels
        labels =  np.zeros((HOT_matrix_unordered.shape[0],1))
        b = 0
        e = len(HOT_matrix_unordered)/n_label

        for i in range(0,n_label):

            for row in range(b,e):
                labels[row] = i

            b += (len(HOT_matrix_unordered)/n_label)
            e += (len(HOT_matrix_unordered)/n_label)


        # one_leave_out_training.append(matrix_training)
        # one_leave_out_test.append(matrix_test)
        # one_leave_out_labels.append(np.ravel(labels))

    # return one_leave_out_training,one_leave_out_test,one_leave_out_labels

        ##classification

        lr = LogisticRegression()

        lr.fit(matrix_training,np.ravel(labels))

        pred =  lr.predict(matrix_test)

        accuracy = 0
        for i in range(0,len(pred)):
            if i == 0 or i == 1:
                if pred[i] == 0 or pred[i]==1:
                    accuracy +=1
                    continue
            if pred[i] == test_label[i]:
                accuracy +=1


        tot_accuracy += float(accuracy)/6

    print tot_accuracy/(len(HOT_matrix)/n_label)

        #print lr.score(matrix_test,test_label)


def video_clustering_fit(HOT_matrix,filename):
    global my_ms
    my_ms.fit(HOT_matrix)

    if len(filename)>2:
        data_organizer.save_matrix_pickle(my_ms,filename)


def video_clustering_pred(data):
    predict = my_ms.fit_predict(data)

    #data_organizer.save_matrix_pickle(predict,'C:/Users/dario.dotti/Documents/cl_prediction_2secWindow_band03.txt')
    return predict

def example_for_every_cluster_center(pred):
    with open('C:/Users/dario.dotti/Documents/content.txt','r') as f:
        images_name = f.read().split('\n')

    class_counter = Counter(pred)
    print class_counter.most_common()

    example_for_cl_centers = []


    for k,v in class_counter.most_common(30):

        #print k,v
        index = np.where(pred == k)[0]

        path = 'C:/Users/dario.dotti/Documents/time_windows_HOT/' +images_name[index[0]].split(' ')[3]
        example_for_cl_centers.append([k,v,path])

    return example_for_cl_centers



def visualize_cluster_pred():
    with open('C:/Users/dario.dotti/Documents/content_6labels.txt','r') as f:
        images_name = f.read().split('\n')

    pred = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/cl_prediction_2secWindow_band03.txt')

    class_counter = Counter(pred)
    print class_counter

    for k in class_counter.keys()[:20]:
        print k
        index = np.where(pred == k)[0]
        for i in index[:20 ]:
            path = 'C:/Users/dario.dotti/Documents/time_windows_HOT/' +images_name[i].split(' ')[3]
            img = cv2.imread(path)

            cv2.imshow('ciao',img)
            cv2.waitKey(0)


def test_hist_task(cluster_model,labels_counter,HOT_matrix):

    ##create bag of words using clustering
    keys_labels = map(lambda x: x[0], labels_counter)

    print keys_labels

    subject_data = []
    for subject in HOT_matrix:

        task_data = []

        for task in subject:

            hist = np.zeros((1,40))
            pred = cluster_model.predict(task)

            for p in pred:
                if p in keys_labels:
                    index = np.where(p == keys_labels)[0][0]

                    hist[0][index] +=1
                #else:
                    #print 'outsider'

            hist = normalize(np.array(hist),norm='l1')
            task_data.append(hist)


        subject_data.append(task_data)


    n_label = 6

    tasks_dict = {}

    keyDict = {'0','1','2','3','4','5'}#['lookKey','lookBall','conf','ripet','write','tea']

    for i in keyDict:
        tasks_dict[i] = []

    print tasks_dict

    #save data per task in dict
    for s in subject_data:
        for n_task,t in enumerate(s):
            tasks_dict[str(n_task)].append(t)

    #print np.array((tasks_dict['3'])).shape


    # test_set = tasks_dict['3'][2]
    # del tasks_dict['3'][2]

    labels = []
    for k in tasks_dict.keys():

        for x in range(0,len(tasks_dict[k])):
            labels.append(int(k))


    matrix_training = []
    for k,v in tasks_dict.items():
        print k
        for vv in v:
            if len(matrix_training)>0:
                matrix_training = np.vstack((matrix_training,vv))
            else:
                matrix_training = vv

    print np.array(matrix_training).shape

    return matrix_training,labels,tasks_dict


    # lr = LogisticRegression()
    #
    # lr.fit(matrix_training,np.ravel(labels))
    #
    # pred =  lr.predict(test_set)
    #
    # print pred


def experiment_video():

    #video_classification_experiments()
    #as_classification_experiment()


    #HOT_matrix = np.array(data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/hot_spatial_grid_4x4x3_6_labels_2secWindow.txt'))
    HOT_matrix_6_tasks = np.array(data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/bow_experiment_data/hot_spatial_grid_4x4x3_6_tasks_2secWindow.txt')).tolist()
    HOT_matrix_5_tasks = np.array(data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/bow_experiment_data/hot_spatial_grid_4x4x3_5_tasks_2secWindow.txt')).tolist()


    length_task3 = [45,51,33,51,62]

    ##modify 5 tasks to make it 6 tasks and merge the two matrices
    for n_subj,subject in enumerate(HOT_matrix_5_tasks):

        new_subject = []

        for n_task,task in enumerate(subject):

            if n_task == 3:

                new_subject.append(task[:length_task3[n_subj]])

                new_subject.append(task[length_task3[n_subj]:])
            else:
                new_subject.append(task)
        HOT_matrix_6_tasks.append(new_subject)


    ##transform matrix for clustering
    HOT_matrix_for_cluster = []

    for s in xrange(0,len(HOT_matrix_6_tasks)):
        for t in xrange(0,len(HOT_matrix_6_tasks[s])):

            for time_slice in  xrange(0,len(HOT_matrix_6_tasks[s][t])):

                if len(HOT_matrix_for_cluster)>0:
                    HOT_matrix_for_cluster = np.vstack((HOT_matrix_for_cluster,HOT_matrix_6_tasks[s][t][time_slice]))
                else:
                    HOT_matrix_for_cluster= HOT_matrix_6_tasks[s][t][time_slice]

    print np.array(HOT_matrix_for_cluster).shape


    # #Clustering
    ##video_clustering_fit(concatenated_matrix,'C:/Users/dario.dotti/Documents/cl_model_2secWindow_band03.txt')


    cluster_model = data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/bow_experiment_data/cl_model_2secWindow_band03.txt')
    labels = cluster_model.predict(HOT_matrix_for_cluster)

    labels_counter = Counter(labels).most_common(40)


    matrix_training,labels,tasks_dict = test_hist_task(cluster_model,labels_counter,HOT_matrix_6_tasks)


    return matrix_training,labels,tasks_dict


def experiment_as():

    as_matrix_6_tasks = np.array(data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/bow_experiment_data/AS_activation_6_labels.txt')).tolist()
    as_matrix_5_tasks_transformed = np.array(data_organizer.load_matrix_pickle('C:/Users/dario.dotti/Documents/bow_experiment_data/AS_activation_5_labels_transformed.txt')).tolist()


    as_matrix = np.vstack((as_matrix_6_tasks,as_matrix_5_tasks_transformed))
    #as_matrix = np.array(as_matrix_6_tasks)

    #as_classification_experiment(as_matrix)


    print as_matrix.shape
    return as_matrix





def main_experiment():

    #visualize_cluster_pred()

    BOW_HOT,labels,tasks_dict = experiment_video()

    as_matrix = experiment_as()

    # data_organizer.save_matrix_pickle(BOW_HOT,'C:/Users/dario.dotti/Documents/BOW_16subject_2sec')
    # data_organizer.save_matrix_pickle(labels,'C:/Users/dario.dotti/Documents/BOW_labels_16subject_2sec')

    concatenated_matrix = np.hstack((BOW_HOT,as_matrix))



    ripetitive_index = range(32,48)
    confusion_index = range(48,64)
    tea_index = range(64,80)

    ##classification

    for i_t in ripetitive_index:

        test_set = BOW_HOT[i_t]
        label_test = labels[i_t]

        training = np.vstack((BOW_HOT[:i_t],BOW_HOT[(i_t+1):]))
        label_tr = labels[:i_t] + labels[(i_t+1):]

        lr = LogisticRegression()

        lr.fit(training,np.ravel(label_tr))

        pred =  lr.predict(np.array(test_set).reshape((1,-1)))

        print pred







if __name__ == '__main__':
    main_experiment()

