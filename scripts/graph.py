#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import constants as cts
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import connected_components
from collections import Counter


def get_connected_components(mat, min_similarity=0.95):
    mat = (mat > min_similarity).astype(np.int)
    return connected_components(mat, directed=False, return_labels=True)


def get_filenames_in_clusters(component_labels, complete_data, data_indices, min_similar_items=10):
    filenames = []
    for cluster_label, count in Counter(component_labels).items():
        if count > min_similar_items:
            file_indices = np.where(component_labels == cluster_label)[0]
            current_cluster_files = []
            for index in file_indices:
                current_cluster_files.append(complete_data['clean_filename'].iloc[data_indices[index]])
            filenames.append(current_cluster_files)
    return filenames


def run_clustering(mat, complete_data, data_indices, min_similarity=0.95, min_similar_items=10):
    num_components, labels = get_connected_components(mat, min_similarity)
    return get_filenames_in_clusters(labels, complete_data, data_indices, min_similar_items)


class Graph:

    # init function to declare class variables 
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]

        # Recursive DFS

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited 
        visited[v] = True

        # Store the vertex to list 
        temp.append(v)

        # Repeat for all vertices adjacent 
        # to this vertex v 
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list 
                temp = self.DFSUtil(temp, i, visited)
        return temp

        # Iterative DFS

    def dfs_iterative(self, start, visited):
        stack, path = [start], []

        while stack:
            vertex = stack.pop()
            # Mark the current vertex as visited 
            visited[vertex] = True

            if vertex in path:
                continue
            path.append(vertex)
            for neighbor in self.adj[vertex]:
                if visited[neighbor] == False:
                    stack.append(neighbor)

        return path

    # method to add an undirected edge 
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

        # Method to retrieve connected components

    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                #                temp = []  # part of recursive dfs call
                #                cc.append(self.DFSUtil(temp, v, visited)) #recursive dfs
                cc.append(self.dfs_iterative(v, visited))  # iterative dfs
        return cc

    def get_filtered_clusters(self, cc, min_similar_items=10):
        filtered_cc = []
        for cluster_label in range(len(cc)):
            for vertex in cc[cluster_label]:
                if (len(self.adj[vertex]) >= min_similar_items):
                    filtered_cc.append(cc[cluster_label])
                    break
        return filtered_cc

    def get_filenames_in_clusters(self, cc, complete_data, data_indices):
        filenames = []
        for cluster_label in range(len(cc)):
            current_cluster_files = []
            for vertex in cc[cluster_label]:
                current_cluster_files.append(complete_data['clean_filename'].iloc[data_indices[vertex]])
            filenames.append(current_cluster_files)
        return filenames

    @classmethod
    def convert_matrix_to_graph(cls, mat, min_similarity):
        num_vertex = mat.shape[0]
        g = Graph(num_vertex)
        for row in range(num_vertex):
            for col in range(row + 1, num_vertex):
                if mat[row, col] >= min_similarity:
                    g.addEdge(row, col)
                    # print(row,col)
        return g

    def convert_df_to_graph(df, min_similarity):
        g = Graph(df.shape[0])
        df.apply(lambda x: pd.Series(x[x >= min_similarity].index).apply(lambda y: g.addEdge(x.name, y)), axis=1)
        return g

    def calculate_silhouette_score(self, data, labels):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        score = calinski_harabasz_score(scaled_data, labels)

        return score
