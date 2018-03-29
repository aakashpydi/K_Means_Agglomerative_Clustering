import sys
import random
import math
import copy
import time
import pandas as pd
import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering
# from sklearn import preprocessing

def computeEuclideanDistance(point_1, point_2):
        euc_dist = 0.0
        for i in range(0, len(point_1)):
            euc_dist += (float(point_1[i]) - float(point_2[i]))*(float(point_1[i]) - float(point_2[i]))
        return float(math.sqrt(euc_dist))

def add_points(point_1, point_2):
    for i in range(0, len(point_1)):
        point_1[i] = float(point_1[i]) + float(point_2[i])
    return point_1

def kmeans_local(file_path, k_value):
    THRESHOLD = 0.1
    data = pd.read_csv(file_path, sep=',', quotechar='"', header=0)
    data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
    X = data.as_matrix()
    row_count = data.shape[0]
    col_count = data.shape[1]

    # we need to initialize K number of cluster seeds
    cluster_indices = list()
    while len(cluster_indices) < k_value:
        seed_index = random.randint(0, row_count - 1)
        if seed_index not in cluster_indices:
            cluster_indices.append(seed_index)

    centroid_points = []
    for index in cluster_indices:
        centroid_points.append(copy.deepcopy(X[index]))

    ############# Used to Verify Output
    # print "Printing SKLEARN output"
    # kmeans = KMeans(n_clusters=k_value, init=np.array(centroid_points)).fit(X)
    # print kmeans.cluster_centers_
    # print kmeans.inertia_
    # print "SKlearn output done"
    #############

    score_function_value = 0.0
    new_score_function_value = float(sys.maxint)

    while (abs(new_score_function_value - score_function_value) > THRESHOLD):
            score_function_value = new_score_function_value

            #iterate through each point in X, and find distance to each of the cluster centroids,
            #assign the min cluster id to the current point
            cluster_labels = list()
            for i in range(0, row_count):
                #we need to find which cluster X[i] is closest to
                min_dist = float(sys.maxint)
                min_index = -1
                index = 0
                for c_point in centroid_points:
                    if (computeEuclideanDistance(c_point, X[i]) < min_dist):
                        min_dist = computeEuclideanDistance(c_point, X[i])
                        min_index = index
                    index = index + 1
                cluster_labels.append(min_index)    #label at each index gives label for entries in X

            #need to find new centroids and then repeat.
            #first we initialize centroid values to zero
            for c_point in centroid_points:
                for i in range (0, len(c_point)):
                    c_point[i] = 0.0

            cluster_sizes = [0] * k_value
            for i in range (0, row_count):
                centroid_points[cluster_labels[i]] = add_points(centroid_points[cluster_labels[i]] , X[i])
                cluster_sizes[cluster_labels[i]] = cluster_sizes[cluster_labels[i]] + 1

            for i in range(0, len(centroid_points)):
                for j in range (0, len(centroid_points[i])):
                    centroid_points[i][j] = float(centroid_points[i][j]) / float(cluster_sizes[i])

            #now you need to find the within cluster sum of squared errors
            new_score_function_value = 0.0
            for i in range (0, row_count):
                # for each X[i], we find sum of squared errors and add to score_function_value
                error = computeEuclideanDistance(X[i], centroid_points[cluster_labels[i]])
                new_score_function_value = new_score_function_value + (error * error)

    print("WC-SSE=" + str(new_score_function_value))
    for i in range(0, len(centroid_points)):
        print("Centroid" + str(i+1) +"=" + str(centroid_points[i]))

def agglomerative_local(file_path, k_value):
    # agglo_start_time = time.time()
    data = pd.read_csv(file_path, sep=',', quotechar='"', header=0)
    data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
    X = data.as_matrix()
    row_count = data.shape[0]
    col_count = data.shape[1]

    # print "STARTING SKLEARN OUTPUT"
    # agg_c = AgglomerativeClustering(n_clusters=k_value, linkage='average').fit(X)
    # #print agg_c.labels_
    # centroid_points = [[0.0 for x in range(col_count)] for y in range(k_value)]
    # label_counts = [0 for x in range (k_value)]
    # centroid_index = 0
    # for i in range(0, row_count):
    #     centroid_points[agg_c.labels_[i]] = add_points(centroid_points[agg_c.labels_[i]], X[i])
    #     label_counts[agg_c.labels_[i]] = label_counts[agg_c.labels_[i]] + 1
    # label_index = 0
    # for c_point in centroid_points:
    #     for i in range(0, len(c_point)):
    #         c_point[i] = c_point[i] / label_counts[label_index]
    #     label_index = label_index + 1
    #     print "Centroid" + str(label_index) + str(c_point)
    # print "Done with SKlearn output\n"


    point_distances = [[0.0 for x in range(row_count)] for y in range(row_count)]
    #print distances
    #first we can compute euclideans distances between all pairs of points
    # beginningTime = time.time()
    for i in range(0, row_count):
        j = i
        while j < row_count:
            point_distances[i][j] = computeEuclideanDistance(X[i], X[j])
            point_distances[j][i] = point_distances[i][j]
            j = j+1
    #print "Time to compute point distances: " + str(time.time() - beginningTime)

    cluster_list = list()
    # beginningTime = time.time()
    for i in range(0, row_count):
        newInnerList = list()
        cluster_list.append([newInnerList, 1])  #the 1 represents the number of elements in cluster
        cluster_list[i][0].append([X[i], i])
    #print "Time to initialize cluster list: " + str(time.time() - beginningTime)
    #print cluster_list
    #now we need distances between clusters
    cluster_distances =  [[0.0 for x in range(len(cluster_list))] for y in range(len(cluster_list))]

    distances_minheap = []
    ignore_cluster_list = []
    totalClusterCount = row_count
    toMerge = None

    # beginningTime = time.time()
    for i in range(0, len(cluster_list)):
        j = i + 1
        while j < len(cluster_list):
            for k in range(0, len(cluster_list[i][0])):
                for l in range(0, len(cluster_list[j][0])):
                    cluster_distances[i][j] = cluster_distances[i][j] + point_distances[cluster_list[i][0][k][1]][cluster_list[j][0][l][1]]
            cluster_distances[i][j] = cluster_distances[i][j] * (1.0/(cluster_list[i][1] * cluster_list[j][1]))
            cluster_distances[j][i] = cluster_distances[i][j]
            heappush(distances_minheap, (cluster_distances[i][j], i , j, cluster_list[i][1], cluster_list[j][1]))
            j = j + 1
    #print "Time to initialize cluster distances: " + str(time.time() - beginningTime) +"\n\n"
    #for cluster in cluster_distances:
    #    print cluster

    while totalClusterCount > k_value:
        loop_start_time = time.time()
        beginningTime = time.time()
        toMerge = heappop(distances_minheap)
        while toMerge[1] in ignore_cluster_list or toMerge[2] in ignore_cluster_list or cluster_list[toMerge[1]][1] != toMerge[3] or cluster_list[toMerge[2]][1] != toMerge[4]:
            toMerge = heappop(distances_minheap)
        #print "Time to find toMerge: " + str(time.time() - beginningTime)

        # beginningTime = time.time()
        cluster_list[toMerge[1]][1] = cluster_list[toMerge[1]][1] + cluster_list[toMerge[2]][1] #update size of new cluster
        for point in cluster_list[toMerge[2]][0]:
            #print point
            cluster_list[toMerge[1]][0].append(point)                                           #add points to cluster
        ignore_cluster_list.append(toMerge[2])
        totalClusterCount = totalClusterCount - 1
        #print "Time to merge clusters: " + str(time.time() - beginningTime)

        # beginningTime = time.time()
        for i in range(0, len(cluster_list)):
            if i == toMerge[1] or i in ignore_cluster_list:
                continue
            cluster_distances[toMerge[1]][i] = 0
            for j in range(0, len(cluster_list[toMerge[1]][0])):
                for k in range(0, len(cluster_list[i][0])):
                    cluster_distances[toMerge[1]][i] = cluster_distances[toMerge[1]][i] + point_distances[cluster_list[toMerge[1]][0][j][1]][cluster_list[i][0][k][1]]
            cluster_distances[toMerge[1]][i] = cluster_distances[toMerge[1]][i] * (1.0/(float(cluster_list[toMerge[1]][1]) * float(cluster_list[i][1])))
            cluster_distances[i][toMerge[1]] = cluster_distances[toMerge[1]][i]

            heap_tuple =  (cluster_distances[toMerge[1]][i], toMerge[1] , i, cluster_list[toMerge[1]][1], cluster_list[i][1])
            heappush(distances_minheap, heap_tuple)
        #print "Time to update distances: " + str(time.time() - beginningTime)


    centroid_points = [[0.0 for x in range(col_count)] for y in range(k_value)]
    centroid_index = 0
    for i in range(0, row_count):
        if i in ignore_cluster_list:
            continue
        else:
            for j in range(0, len(cluster_list[i][0])):
                centroid_points[centroid_index] = add_points(centroid_points[centroid_index], cluster_list[i][0][j][0])

            for j in range(0, len(centroid_points[centroid_index])):
                centroid_points[centroid_index][j] = centroid_points[centroid_index][j] / float(cluster_list[i][1])
            centroid_index = centroid_index + 1

    wse_score = 0.0
    centroid_index = 0
    for i in range(0, row_count):
        if i in ignore_cluster_list:
            continue
        else:
            for j in range(0, len(cluster_list[i][0])):
                error = computeEuclideanDistance(cluster_list[i][0][j][0], centroid_points[centroid_index])
                wse_score = wse_score + (error * error)
            centroid_index = centroid_index + 1
    print "WC-SSE=" + str(wse_score)
    for i, centroid in enumerate(centroid_points, 1):
        print "Centroid" + str(i) + "=" + str(centroid)


# We expect the command line input to be in the following order
file_path = sys.argv[1]     # absolute path to CSV file
k_value = int(sys.argv[2])       # k value to use for clustering
model = sys.argv[3]         # can either be km for KMEANS or ac for agglomerative clustering

if model == "km":
    kmeans_local(file_path, k_value)
elif model == "ac":
    agglomerative_local(file_path, k_value)
