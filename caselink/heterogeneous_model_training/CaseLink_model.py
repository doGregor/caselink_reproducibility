import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F
import torch
import math
    

class HeteroCaseLink(nn.Module):
    def __init__(self, hidden_channels, dropout, metadata, num_layers=2, num_attention_heads=1):
        super(HeteroCaseLink, self).__init__()
        self.dropout = dropout
        
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_attention_heads, group='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        residual_x_dict = x_dict
        
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        
        for idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if idx + 1 < len(self.convs):
                x_dict = {key: x.relu() for key, x in x_dict.items()}

        """
        if self.dropout > 0:
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        """
        
        x_dict = {key: x + residual_x_dict[key] for key, x in x_dict.items()}

        return x_dict


def early_stopping(highest_f1score, epoch_f1score, epoch_num, continues_epoch):
    if epoch_f1score <= highest_f1score:
        if continues_epoch > 1000:
            return [highest_f1score, True]
        else:
            continues_epoch += 1
            return [highest_f1score, False, continues_epoch]
    else:
        continues_epoch = 0
        return [epoch_f1score, False, continues_epoch]
    

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

