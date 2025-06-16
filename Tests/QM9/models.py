import torch
import torch.nn as nn

from layers import GraphConvolutionLayer, GraphAggregationLayer
from utils import gumbel_softmax

class MolGANGenerator(nn.Module):
    def __init__(self, z_dim, num_nodes, num_node_types, num_edge_types,
                 hidden_dim=128, tau=1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.tau = tau

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # outputs for nodes + edges
        self.logits_node = nn.Linear(hidden_dim, num_nodes * num_node_types)
        self.logits_edge = nn.Linear(hidden_dim, num_nodes * num_nodes * num_edge_types)

    def forward(self, z, *, hard: bool = True):
        batch = z.size(0)
        h = self.net(z)
        ln = self.logits_node(h).view(batch, self.num_nodes, self.num_node_types)
        le = self.logits_edge(h).view(batch, self.num_nodes, self.num_nodes, self.num_edge_types)
        # make adjacency symmetric
        le = (le + le.permute(0,2,1,3)) / 2
        node_soft = gumbel_softmax(ln, tau=self.tau, hard=hard, dim=-1)
        edge_soft = gumbel_softmax(le, tau=self.tau, hard=hard, dim=-1)
        # enforce symmetry after sampling and zero self‑loops
        edge_soft = (edge_soft + edge_soft.permute(0, 2, 1, 3)) / 2
        idx = torch.arange(self.num_nodes, device=edge_soft.device)
        edge_soft[:, idx, idx, :] = 0.0          # clear diagonal
        edge_soft[:, idx, idx, 0] = 1.0          # channel 0 = “no bond”
        return edge_soft, node_soft

class MolGANDiscriminator(nn.Module):
    def __init__(self, num_node_feats, num_edge_types, hidden_dim=128):
        super().__init__()
        in_dim = num_node_feats
        self.conv1 = GraphConvolutionLayer(in_dim, hidden_dim, num_edge_types, activation=nn.ReLU())
        # second convolution takes previous hidden plus original node features
        self.conv2 = GraphConvolutionLayer(hidden_dim + num_node_feats, hidden_dim, num_edge_types, activation=nn.ReLU())
        self.readout = GraphAggregationLayer(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, adj, feats):
        # adj: [B, N, N, R], feats: [B, N, T]
        h = self.conv1(adj, None, feats)
        h = self.conv2(adj, h, feats)
        g = self.readout(h)
        return self.out(g)

class MolGANReward(nn.Module):
    def __init__(self, num_node_feats, num_edge_types, hidden_dim=128):
        super().__init__()
        # identical architecture to D, but outputs *continuous* reward
        self.conv1 = GraphConvolutionLayer(num_node_feats, hidden_dim, num_edge_types, activation=nn.ReLU())
        # second convolution takes previous hidden plus original node features
        self.conv2 = GraphConvolutionLayer(hidden_dim + num_node_feats, hidden_dim, num_edge_types, activation=nn.ReLU())
        self.readout = GraphAggregationLayer(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.out = nn.Linear(hidden_dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, adj, feats):
        h = self.conv1(adj, None, feats)
        h = self.conv2(adj, h, feats)
        g = self.readout(h)
        return self.act(self.out(g))