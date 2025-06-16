import torch
import torch.nn as nn

class GraphConvolutionLayer(nn.Module):
    """
    Relational Graph Convolutional layer (R‑GCN). Excludes the ‘no‑bond’ relation (channel 0) from message passing.
    """
    def __init__(self, in_dim, out_dim, num_edge_types, dropout=0.0, activation=nn.ReLU()):
        super().__init__()
        # num_edge_types includes the “no‑bond” relation in index 0; we skip it for message passing
        self.total_relations = num_edge_types
        self.num_relations = num_edge_types - 1
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(self.num_relations)])
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, hidden, node_feats):
        # adj: [B, N, N, R], node_feats: [B, N, T], hidden: [B, N, H] or None
        # build input
        x = torch.cat([hidden, node_feats], dim=-1) if hidden is not None else node_feats
        # drop “no‑bond” channel (index 0) for message passing
        adj_rel = adj[..., 1:]
        if adj_rel.shape[-1] != self.num_relations:
            raise ValueError(f"Expected {self.num_relations} edge types (excluding no‑bond), got {adj_rel.shape[-1]}")
        supports = torch.stack([lin(x) for lin in self.linears], dim=-1)  # [B, N, out_dim, R-1]
        out = torch.einsum('bijr,bjor->bio', adj_rel, supports)
        # self-loop
        out = out + self.self_lin(x)
        if self.activation is not None:
            out = self.activation(out)
        return self.dropout(out)

class GraphAggregationLayer(nn.Module):
    """
    Gated readout: i = sigmoid(W_i x), j = W_j x, sum(i * j).
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, activation=nn.ReLU()):
        super().__init__()
        self.i_dense = nn.Linear(in_dim, out_dim)
        self.j_dense = nn.Linear(in_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]
        i = torch.sigmoid(self.i_dense(x))
        j = self.j_dense(x)
        if self.activation is not None:
            j = self.activation(j)
        out = torch.sum(i * j, dim=1)  # [B, out_dim]
        if self.activation is not None:
            out = self.activation(out)
        return self.dropout(out)