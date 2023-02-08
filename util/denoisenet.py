from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch_geometric.nn import GCNConv, Sequential
from torch_scatter import scatter_max
from sklearn.preprocessing import normalize

from util.pointnet import PointNet
from util.meshnet import MeshConv


class PosNet(nn.Module):
    def __init__(self, device, activation="lrelu"):
        super(PosNet, self).__init__()
        self.device = device

        h = [16, 16, 32, 64, 128, 256, 256, 512, 256, 256, 128, 64, 32, 16, 3]
        
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

    def forward(self, data):

        z1, x_pos, edge_index = data.z1.to(self.device), data.x_pos.to(self.device), data.edge_index.to(self.device)
        
        z_min, z_max = torch.min(z1, dim=0, keepdim=True)[0], torch.max(z1, dim=0, keepdim=True)[0]
        z_sc = torch.max(z_max - z_min)
        zc = (z_min + z_max) * 0.5
        z1 = (z1 - zc) / z_sc

        x = z1
        for b in self.blocks:
            y = b(x, edge_index)
            x = y
        
        return x_pos + x

class NormNet(nn.Module):
    def __init__(self, device, activation="lrelu"):
        super(NormNet, self).__init__()
        self.device = device

        h = [7, 16, 32, 64, 128, 256, 256, 512, 256, 256, 128, 64, 32, 16, 3]
        
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

    def forward(self, data):

        z2, edge_index = data.z2.to(self.device), data.face_index.to(self.device)

        x = z2
        for b in self.blocks:
            y = b(x, edge_index)
            x = y
        
        x = torch.tanh(x)
        x_norm = torch.reciprocal(torch.norm(x, dim=1, keepdim=True).expand(-1, 3) + 1.0e-12)
        x = torch.mul(x, x_norm)
        return x