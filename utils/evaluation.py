import numpy as np
import sklearn.metrics as metrics
import sys
from munkres import Munkres


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average)
    precision = np.round(precision, decimals)

    # recall
    recall = metrics.recall_score(y_true, y_pred, average=average)
    recall = np.round(recall, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average)
    f_score = np.round(f_score, decimals)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """Computes the predicted labels, where label assignments now
        correspond to the actual labels in y_true (as estimated by Munkres)

        Args:
            cluster_assignments: array of labels, outputted by kmeans
            y_true:              true labels
            n_clusters:          number of clusters in the dataset

        Returns:
            a tuple containing the accuracy and confusion matrix, in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)

    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)

    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari}, **classification_metrics), confusion_matrix


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Using either a newly instantiated ClusterClass or a provided cluster_obj, generates
        cluster assignments based on input data.

        Args:
            x: the points with which to perform clustering
            cluster_obj: a pre-fitted instance of a clustering class,如初始化好的kmeans类
            ClusterClass: a reference to the sklearn clustering class, necessary
              if instantiating a new clustering class
            n_clusters: number of clusters in the dataset, necessary
                        if instantiating new clustering class
            init_args: any initialization arguments passed to ClusterClass

        Returns:
            a tuple containing the label assignments and the clustering object
    """
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def evaluation(y_pred, y_true, accumulated_metrics=None):
    n_clusters = np.size(np.unique(y_true))
    if np.min(y_true) == 1:
        y_true = y_true - 1
    scores, _ = clustering_metric(y_true, y_pred, n_clusters)
    if accumulated_metrics is not None:
        accumulated_metrics['acc'].append(scores['accuracy'])
        accumulated_metrics['nmi'].append(scores['NMI'])
        accumulated_metrics['ARI'].append(scores['ARI'])
        accumulated_metrics['f-mea'].append(scores['f_measure'])
    return scores
