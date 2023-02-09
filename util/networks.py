import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv, Sequential



class SingleScaleGCN(nn.Module):
    def __init__(self, device, activation="lrelu", skip=False):
        super(SingleScaleGCN, self).__init__()
        self.device = device
        self.skip = skip

        h = [4, 16, 32, 64, 128, 256, 256, 512, 256, 256, 128, 64, 32, 16, 3]
        
        activ_dict = {"relu": nn.ReLU(), "lrelu": nn.LeakyReLU()}
        activation_func = activ_dict[activation]
        
        blocks = []
        
        for i in range(12):
            block = Sequential("x, edge_index", [
                (GCNConv(h[i], h[i+1]), "x, edge_index -> x"),
                nn.BatchNorm1d(h[i+1]),
                activation_func,
            ])
            blocks.append(block)
        
        block = Sequential("x, edge_index", [
            (GCNConv(h[12], h[13]), "x, edge_index -> x"),
            nn.BatchNorm1d(h[13]),
            activation_func,
            (nn.Linear(h[13], h[14]), "x -> x"),
        ])
        blocks.append(block)
        
        self.blocks = nn.ModuleList(blocks)

        skip_blocks = []
        for i in range(6):
            skip_blocks.append(nn.Linear(h[i+1]*2, h[i+1]))
        self.skip_blocks = nn.ModuleList(skip_blocks)

    def forward(self, data, dm=None):

        z1, x_pos, edge_index = data.z1.to(self.device), data.x_pos.to(self.device), data.edge_index.to(self.device)
        
        z_min, z_max = torch.min(z1, dim=0, keepdim=True)[0], torch.max(z1, dim=0, keepdim=True)[0]
        z_sc = torch.max(z_max - z_min)
        zc = (z_min + z_max) * 0.5
        z1 = (z1 - zc) / z_sc
        
        if type(dm) == np.ndarray:
            dm = torch.from_numpy(dm)
        elif type(dm) != torch.Tensor:
            dm = torch.ones([z1.shape[0], 1])

        dm = dm.to(self.device)
        z1 = dm * z1
        z1 = torch.cat([z1, dm], dim=1)

        x = z1
        skip_in = []
        for i, b in enumerate(self.blocks):
            if i <= 5:
                """ encoder """
                y = b(x, edge_index)
                x = y
                skip_in.append(y)
            
            elif i <= 7:
                """ bottle-neck """
                y = b(x, edge_index)
                x = y
            else:
                """ decoder """
                if self.skip:
                    x_src = skip_in[13-i]
                    x_cat = torch.cat([x_src, x], dim=1)
                    x = self.skip_blocks[13-i](x_cat)
                y = b(x, edge_index)
                x = y

        return x_pos + x
