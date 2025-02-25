import math

import torch
from torch import nn
from torch.nn import functional as F

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    MolecularGraphConfigs,
    RegularizationConfigs,
)
from ..custom_types import GraphAttentionData
from ..utils.stochastic_depth import StochasticDepth, SkipStochasticDepth
from ..utils.nn_utils import (
    NormalizationType,
    get_normalization_layer,
    get_linear,
    get_feedforward,
)
from .base_block import BaseGraphNeuralNetworkLayer


class EfficientGraphAttentionBlock(nn.Module):
    """
    Efficient Graph Attention Block module.
    Ref: swin transformer
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        # Graph attention
        self.graph_attention = EfficientGraphAttention(
            global_cfg=global_cfg,
            molecular_graph_cfg=molecular_graph_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Feed forward network
        self.feedforward = FeedForwardNetwork(
            global_cfg=global_cfg,
            gnn_cfg=gnn_cfg,
            reg_cfg=reg_cfg,
        )

        # Normalization
        normalization = NormalizationType(reg_cfg.normalization)
        self.norm_attn = get_normalization_layer(normalization)(global_cfg.hidden_size)
        self.norm_ffn = get_normalization_layer(normalization)(global_cfg.hidden_size)

        # Stochastic depth
        self.stochastic_depth_attn = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )
        self.stochastic_depth_ffn = (
            StochasticDepth(reg_cfg.stochastic_depth_prob)
            if reg_cfg.stochastic_depth_prob > 0.0
            else SkipStochasticDepth()
        )

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        # ref: swin transformer https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L452
        # x = x + self.stochastic_depth(self.graph_attention(self.norm_attn(x)))
        # x = x + self.stochastic_depth(self.feedforward(self.norm_ffn(x)))

        # attention
        node_hidden, edge_hidden = self.norm_attn(node_features, edge_features)
        node_hidden, edge_hidden = self.graph_attention(data, node_hidden, edge_hidden)
        node_hidden, edge_hidden = self.stochastic_depth_attn(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )

        # feedforward
        node_hidden, edge_hidden = self.norm_ffn(node_features, edge_features)
        node_hidden, edge_hidden = self.feedforward(node_hidden, edge_hidden)
        node_hidden, edge_hidden = self.stochastic_depth_ffn(
            node_hidden, edge_hidden, data.node_batch
        )
        node_features, edge_features = (
            node_hidden + node_features,
            edge_hidden + edge_features,
        )
        return node_features, edge_features


class EfficientGraphAttention(BaseGraphNeuralNetworkLayer):
    """
    Efficient Graph Attention module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__(global_cfg, molecular_graph_cfg, gnn_cfg, reg_cfg)

        # Edge linear layer
        self.edge_attr_linear = self.get_edge_linear(gnn_cfg, global_cfg, reg_cfg)

        # Node hidden layer
        self.node_hidden_linear = self.get_node_linear(global_cfg, reg_cfg)

        # Edge hidden layer
        self.edge_hidden_linear = get_linear(
            in_features=global_cfg.hidden_size,
            out_features=global_cfg.hidden_size,
            activation=global_cfg.activation,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

        # message linear
        self.use_message_gate = gnn_cfg.use_message_gate
        if self.use_message_gate:
            self.gate_linear = get_linear(
                in_features=global_cfg.hidden_size * 2,
                out_features=global_cfg.hidden_size,
                activation=None,
                bias=True,
            )
            self.candidate_linear = get_linear(
                in_features=global_cfg.hidden_size * 2,
                out_features=global_cfg.hidden_size,
                activation=None,
                bias=True,
            )
        else:
            self.message_linear = get_linear(
                in_features=global_cfg.hidden_size * 3,
                out_features=global_cfg.hidden_size,
                activation=global_cfg.activation,
                bias=True,
            )

        # Multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=global_cfg.hidden_size,
            num_heads=gnn_cfg.atten_num_heads,
            dropout=reg_cfg.atten_dropout,
            bias=True,
            batch_first=True,
        )

        # scalar for attention bias
        self.use_angle_embedding = gnn_cfg.use_angle_embedding
        if self.use_angle_embedding:
            self.attn_scalar = nn.Parameter(
                torch.ones(gnn_cfg.atten_num_heads), requires_grad=True
            )
        else:
            self.attn_scalar = torch.tensor(1.0)

        # Graph attention for aggregation
        # ref: "How Attentive are Graph Attention Networks?" <https://arxiv.org/abs/2105.14491>
        self.use_graph_attention = gnn_cfg.use_graph_attention
        if self.use_graph_attention:
            self.attn_weight = nn.Parameter(
                torch.empty(
                    1,
                    1,
                    gnn_cfg.atten_num_heads,
                    global_cfg.hidden_size // gnn_cfg.atten_num_heads,
                ),
                requires_grad=True,
            )
            # glorot initialization
            stdv = math.sqrt(
                6.0 / (self.attn_weight.shape[-2] + self.attn_weight.shape[-1])
            )
            self.attn_weight.data.uniform_(-stdv, stdv)

    def forward(
        self,
        data: GraphAttentionData,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        # Get edge attributes
        edge_attr = self.get_edge_features(data)
        edge_attr = self.edge_attr_linear(edge_attr)

        # Get node features
        node_features = self.get_node_features(node_features, data.neighbor_list)
        node_hidden = self.node_hidden_linear(node_features)

        # Get edge faetures
        edge_hidden = self.edge_hidden_linear(edge_features)

        # Concatenate edge and node features (num_nodes, num_neighbors, hidden_size)
        if self.use_message_gate:
            message = torch.cat([edge_attr, node_hidden], dim=-1)
            update_gate = torch.sigmoid(self.gate_linear(message))
            candidate = torch.tanh(self.candidate_linear(message))
            message = update_gate * candidate + (1 - update_gate) * edge_hidden
        else:
            message = self.message_linear(
                torch.cat([edge_attr, edge_hidden, node_hidden], dim=-1)
            )

        # Multi-head self-attention
        if self.use_angle_embedding:
            angle_embedding = data.angle_embedding.reshape(
                -1,
                self.attn_scalar.shape[0],
                data.angle_embedding.shape[-2],
                data.angle_embedding.shape[-1],
            ) * self.attn_scalar.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            angle_embedding = angle_embedding.reshape(
                -1, data.angle_embedding.shape[-2], data.angle_embedding.shape[-1]
            )
            attn_mask = data.attn_mask + angle_embedding
        else:
            attn_mask = data.attn_mask
        edge_output = self.multi_head_attention(
            query=message,
            key=message,
            value=message,
            # key_padding_mask=~data.neighbor_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]

        # Aggregation
        if self.use_graph_attention:
            num_nodes, num_neighbors, _ = edge_output.shape
            _, _, num_heads, head_dim = self.attn_weight.shape

            edge_output = edge_output.view(
                num_nodes, num_neighbors, num_heads, head_dim
            )
            # alpha (num_nodes, num_neighbors, num_heads)
            alpha = (edge_output * self.attn_weight).sum(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = alpha.masked_fill(
                data.neighbor_mask.unsqueeze(-1) == 0, float("-inf")
            )
            alpha = F.softmax(alpha, dim=1)
            node_output = (alpha.unsqueeze(-1) * edge_output).sum(1)
            node_output = node_output.view(num_nodes, -1)
            edge_output = edge_output.view(num_nodes, num_neighbors, -1)
        else:
            node_output = self.aggregate(edge_output, data.neighbor_mask)

        return node_output, edge_output


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network module.
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()
        self.mlp_node = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )
        self.mlp_edge = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=gnn_cfg.ffn_hidden_layer_multiplier,
            bias=True,
            dropout=reg_cfg.mlp_dropout,
        )

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor):
        return self.mlp_node(node_features), self.mlp_edge(edge_features)
