import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder(nn.Module):
    """output latent representation"""

    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(Encoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim:
                if self._batchnorm and i < self._dim - 1:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        latent = self._encoder(x)
        return latent


class Decoder(nn.Module):
    """Decode hidden variable z into x-hat"""

    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(Decoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm and i < self._dim - 1:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, latent):
        x_hat = self._decoder(latent)
        return x_hat


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space.The latent of the middle layer uses softmax activation。"""

    def __init__(self, encoder_dim, activation='relu', batchnorm=True):

        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.
            x: [num, feat_dim] float tensor.
            latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.
            latent: [num, latent_dim] float tensor, representation Z.
            x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.
              x: [num, feat_dim] float tensor.
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat  # , latent


class ClusterLayer(nn.Module):
    def __init__(self, n_z, n_clusters, alpha=1.0):
        super(ClusterLayer, self).__init__()
        self.alpha = alpha
        self.soft_cluster = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.soft_cluster.data)

    def forward(self, z):
        """Soft assignment between embedding points and cluster centroids"""
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.soft_cluster, 2), dim=2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)  
        return q


class GNNLayer(Module):
    def __init__(self,
                 in_features_dim, out_features_dim,
                 activation='relu', use_bias=True):
        super(GNNLayer, self).__init__()
        self.in_features = in_features_dim
        self.out_features = out_features_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(in_features_dim, out_features_dim))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features_dim))
        self.init_parameters()

        self._bn1d = nn.BatchNorm1d(out_features_dim)
        if activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self._activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        elif activation == 'relu':
            self._activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation type %s' % self._activation)

    def init_parameters(self):
        """Initialize weights"""
        torch.nn.init.xavier_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, features, adj, active=True, batchnorm=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)  
        if self.use_bias:
            output += self.bias
        if batchnorm:
            output = self._bn1d(output)
        if active:
            output = self._activation(output)
        return output


class GraphEncoder(nn.Module):
    def __init__(self, encoder_dim, activation='relu', batchnorm=True):
        super(GraphEncoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(GNNLayer(encoder_dim[i], encoder_dim[i + 1], activation=self._activation))
        self._encoder = nn.Sequential(*encoder_layers)

    def forward(self, x, adj, skip_connect=True):

        z = self._encoder[0](x, adj)
        for layer in self._encoder[1:-1]:
            if skip_connect:
                z = layer(z, adj) + z
            else:
                z = layer(z, adj)
        z = self._encoder[-1](z, adj, False, False)
        return z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, activation=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.activation = activation

    def forward(self, z):
        adj = torch.mm(z, z.t())
        adj = self.activation(adj)
        return adj


class InnerProductDecoderW(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, z_dim, activation=torch.sigmoid):
        super(InnerProductDecoderW, self).__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(z_dim, z_dim))
        torch.nn.init.xavier_normal_(self.W)

    def forward(self, z):
        adj = z @ self.W @ torch.t(z)
        adj = self.activation(adj)
        return adj


class DDC(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super(DDC, self).__init__()
        self._latent_dim = latent_dim
        self._n_clusters = n_clusters

        self.clustering = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, z):
        y = self.clustering(z)
        return y


class AttentionLayer(nn.Module):
    def __init__(self, latent_dim):
        super(AttentionLayer, self).__init__()
        self._latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.ReLU(),
            nn.Linear(self._latent_dim * 2, self._latent_dim * 2),
            nn.BatchNorm1d(self._latent_dim * 2),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(self._latent_dim * 2, 2, bias=True)

    def forward(self, h1, h2, tau=10.0):
        h = torch.cat((h1, h2), dim=1)
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) / tau
        e = F.softmax(act, dim=1)
        # weights = torch.mean(e, dim=0)
        # h = weights[0] * h1 + weights[1] * h2
        h = e[:, 0].unsqueeze(1) * h1 + e[:, 1].unsqueeze(1) * h2
        return h


class InstanceProject(nn.Module):
    def __init__(self, latent_dim):
        super(InstanceProject, self).__init__()
        self._latent_dim = latent_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.instance_projector(x)


class ClusterProject(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super(ClusterProject, self).__init__()
        self._latent_dim = latent_dim
        self._n_clusters = n_clusters
        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
            # nn.Linear(self._latent_dim, self._latent_dim),
            # nn.BatchNorm1d(self._latent_dim),
            # nn.ReLU(),
        )
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            # nn.BatchNorm1d(self._n_clusters), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.cluster_projector(x)
        y = self.cluster(z)
        return y, z
