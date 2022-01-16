import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.models import BaseModel
from cogdl.layers import GCNLayer, SAGELayer, GATLayer, GCNIILayer


class GCN(BaseModel):
    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout,
        activation="relu",
        norm="batchnorm",
    ):
        super(GCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    shapes[i],
                    shapes[i + 1],
                    dropout=dropout if i != num_layers - 1 else 0,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
        return h


class SAGE(BaseModel):
    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        aggr="mean",
        dropout=0.5,
        norm="batchnorm",
        activation="relu",
        normalize=False,
    ):
        super(SAGE, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                SAGELayer(
                    shapes[i],
                    shapes[i + 1],
                    aggr=aggr,
                    normalize=normalize if i != num_layers - 1 else False,
                    dropout=dropout if i != num_layers - 1 else False,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        for layer in self.layers:
            x = layer(graph, x)
        return x


class GAT(BaseModel):
    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout=0.5,
        input_drop=0.25,
        attn_drop=0.1,
        alpha=0.2,
        nhead=4,
        residual=True,
        last_nhead=1,
        norm="batchnorm",
        activation="relu",
    ):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.input_drop = input_drop
        self.attentions = nn.ModuleList()
        self.attentions.append(
            GATLayer(in_feats, hidden_size, nhead=nhead, attn_drop=attn_drop, alpha=alpha, residual=residual, norm=norm, activation=activation)
        )
        for i in range(num_layers - 2):
            self.attentions.append(
                GATLayer(
                    hidden_size * nhead,
                    hidden_size,
                    nhead=nhead,
                    attn_drop=attn_drop,
                    alpha=alpha,
                    residual=residual,
                    norm=norm,
                    activation=activation,
                )
            )
        self.attentions.append(
            GATLayer(
                hidden_size * nhead,
                out_feats,
                attn_drop=attn_drop,
                alpha=alpha,
                nhead=last_nhead,
                residual=False,
            )
        )
        self.num_layers = num_layers
        self.last_nhead = last_nhead
        self.residual = residual

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        x = F.dropout(x, p=self.input_drop, training=self.training)
        for i, layer in enumerate(self.attentions):
            x = layer(graph, x)
            if i != self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GCNII(BaseModel):

    def __init__(
        self,
        in_feats,
        hidden_size,
        out_feats,
        num_layers,
        dropout=0.5,
        alpha=0.1,
        lmbda=0.5,
        residual=False,
    ):
        super(GCNII, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_feats, hidden_size))
        self.fc_layers.append(nn.Linear(hidden_size, out_feats))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)
        self.alpha = alpha
        self.lmbda = lmbda

        self.layers = nn.ModuleList(
            GCNIILayer(hidden_size, self.alpha, math.log(self.lmbda / (i + 1) + 1), residual) for i in range(num_layers)
        )

        self.fc_parameters = list(self.fc_layers.parameters())
        self.conv_parameters = list(self.layers.parameters())

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        init_h = self.dropout(x)
        init_h = self.activation(self.fc_layers[0](init_h))

        h = init_h

        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h, init_h)
            h = self.activation(h)
        h = self.dropout(h)
        out = self.fc_layers[1](h)
        return out

    def get_optimizer(self, args):
        return torch.optim.Adam(
            [
                {"params": self.fc_parameters, "weight_decay": args.wd1},
                {"params": self.conv_parameters, "weight_decay": args.wd2},
            ],
            lr=args.lr,
        )
