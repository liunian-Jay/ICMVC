import torch
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import scipy.sparse as sp


def normalization_adj(adjacency):
    """calculate L=D^-0.5 * (A+I) * D^-0.5,
    Args:
        adjacency: sp.csr_matrix.
    Returns:
        The normalized adjacency matrix, the type is torch.sparse.FloatTensor
    """
    adjacency += sp.eye(adjacency.shape[0])  # add self-join
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()

    # transform to torch.sparse.FloatTensor
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_similarity_matrix(features, method='heat'):
    """Get the similarity matrix"""
    dist = None
    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        # features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        # features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    return dist


def get_graph(features, topk=10, method='heat'):
    """Generate graph adjacency matrix using different similarity methods"""
    dist = get_similarity_matrix(features, method=method)
    # print(dist)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edges_unordered = []
    for i, ks_i in enumerate(inds):
        for k_i in ks_i:
            if k_i != i:
                edges_unordered.append([i, k_i])
    return edges_unordered


def get_adjacency(features, n, topk=10, self_join=True, method='heat'):
    """Get the standardized adjacency matrix, sparse and dense"""
    # features = features.cpu().numpy()# to cpu
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = get_graph(features, topk, method)
    edges_unordered = np.array(edges_unordered, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    raw_adj = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if self_join:
        adj = adj + sp.eye(adj.shape[0])  # add self-join
    # raw_adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, raw_adj


def transfer_S(dist1, dist2, mask):
    """For the similarity matrix S, missing data uses relational transfer"""
    miss1 = (mask[:, 0] == False)
    miss2 = (mask[:, 1] == False)
    dist1[:, miss1] = 0
    dist1[miss1, :] = dist2[miss1, :]
    dist2[:, miss2] = 0
    dist2[miss2, :] = dist1[miss2, :]
    return dist1, dist2


def get_edges(dist, topk=10):
    """Through the similarity matrix, the graph structure is established"""
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edges_unordered = []
    for i, ks_i in enumerate(inds):
        for k_i in ks_i:
            if k_i != i:
                edges_unordered.append([i, k_i])
    return edges_unordered


def graph2adj(edges_unordered, n, self_join=True):
    """Convert the established graph structure into the adjacency matrix required by GCN"""
    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.array(edges_unordered, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    raw_adj = sparse_mx_to_torch_sparse_tensor(adj + sp.eye(adj.shape[0]))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if self_join:  # add self-join
        adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, raw_adj


def get_miss_adjacency(features1, features2, mask, n, topk=10):
    """
    Get the adjacency matrix of all data (including missing data), 
    note that the self-connection matrix is added here, because 
    the data is processed
    """
    features1 = features1.cpu().numpy()
    features2 = features2.cpu().numpy()
    mask = mask.cpu().numpy()

    dist1, dist2 = get_similarity_matrix(features1, 'heat'), get_similarity_matrix(features2, 'heat')
    dist1, dist2 = transfer_S(dist1, dist2, mask)
    edges1_unordered, edges2_unordered = get_edges(dist1, topk), get_edges(dist2, topk)
    adj1, raw_adj1 = graph2adj(edges1_unordered, n)
    adj2, raw_adj2 = graph2adj(edges2_unordered, n)
    return adj1, raw_adj1, adj2, raw_adj2
