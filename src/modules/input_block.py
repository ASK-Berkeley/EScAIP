import torch.nn as nn

from ..configs import (
    GlobalConfigs,
    GraphNeuralNetworksConfigs,
    MolecularGraphConfigs,
    RegularizationConfigs,
)
from ..custom_types import GraphAttentionData
from .base_block import BaseGraphNeuralNetworkLayer
from ..utils.nn_utils import get_feedforward, get_normalization_layer


class InputBlock(nn.Module):
    """
    Wrapper of InputLayer for adding normalization
    """

    def __init__(
        self,
        global_cfg: GlobalConfigs,
        molecular_graph_cfg: MolecularGraphConfigs,
        gnn_cfg: GraphNeuralNetworksConfigs,
        reg_cfg: RegularizationConfigs,
    ):
        super().__init__()

        self.input_layer = InputLayer(global_cfg, molecular_graph_cfg, gnn_cfg, reg_cfg)

        self.norm_node = get_normalization_layer(reg_cfg.normalization)(
            global_cfg.hidden_size
        )
        self.norm_edge = get_normalization_layer(reg_cfg.normalization, "edge")(
            global_cfg.hidden_size
        )

    def forward(self, inputs: GraphAttentionData):
        node_features, edge_features = self.input_layer(inputs)
        return self.norm_node(node_features), self.norm_edge(
            edge_features, inputs.neighbor_mask
        )


class InputLayer(BaseGraphNeuralNetworkLayer):
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
        self.edge_attr_norm = get_normalization_layer(reg_cfg.normalization, "edge")(
            global_cfg.hidden_size
        )

        # ffn for edge features
        self.edge_ffn = get_feedforward(
            hidden_dim=global_cfg.hidden_size,
            activation=global_cfg.activation,
            hidden_layer_multiplier=1,
            dropout=reg_cfg.edge_ffn_dropout,
            bias=True,
        )

    def forward(self, inputs: GraphAttentionData):
        # Get edge features
        edge_features = self.get_edge_features(inputs)

        # Edge processing
        edge_hidden = self.edge_attr_linear(edge_features)
        edge_hidden = self.edge_attr_norm(edge_hidden, inputs.neighbor_mask)
        edge_output = edge_hidden + self.edge_ffn(edge_hidden)

        # Aggregation
        node_output = self.aggregate(edge_output, inputs.neighbor_mask)

        # Update inputs
        return node_output, edge_output
