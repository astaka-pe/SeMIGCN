import torch
import torch.nn as nn
import numpy as np
import os
import copy
from torch_geometric.nn import GCNConv, ChebConv, Sequential


class MeshPool(nn.Module):
    def __init__(self, pool_hash):
        super(MeshPool, self).__init__()
        self.register_buffer("pool_hash", pool_hash)
    
    def forward(self, input):
        v_sum = torch.sum(self.pool_hash.to_dense(), dim=1, keepdim=True)
        out = torch.sparse.mm(self.pool_hash, input) / v_sum
        return out
    

class MeshUnpool(nn.Module):
    def __init__(self, unpool_hash):
        super(MeshUnpool, self).__init__()
        self.register_buffer("unpool_hash", unpool_hash)
    
    def forward(self, input):
        out = torch.sparse.mm(self.unpool_hash, input)
        return out

#############################################################

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_index1, edge_index2, pool_hash, K=3, drop_rate=0.0):
        super(DownConv, self).__init__()
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        conv = "chebconv"
        if conv == "chebconv":
            """ chebconv """
            self.model1 = Sequential("x, edge_index", [
                (ChebConv(in_channels, out_channels, K=K), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (ChebConv(out_channels, out_channels, K=K), "x, edge_index -> x"),
                (MeshPool(pool_hash), "x -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),
            ])
            self.model2 = Sequential("x, edge_index", [
                (ChebConv(out_channels, out_channels, K=K), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (ChebConv(out_channels, out_channels, K=K), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (ChebConv(out_channels, out_channels, K=K), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (nn.Dropout(drop_rate), "x -> x"),
            ])
        else:
            """ gcnconv """
            self.model1 = Sequential("x, edge_index", [
                (GCNConv(in_channels, out_channels), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (GCNConv(out_channels, out_channels), "x, edge_index -> x"),
                (MeshPool(pool_hash), "x -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),
            ])
            self.model2 = Sequential("x, edge_index", [
                (GCNConv(out_channels, out_channels), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (GCNConv(out_channels, out_channels), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (GCNConv(out_channels, out_channels), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (nn.Dropout(drop_rate), "x -> x"),
            ])
    
    def forward(self, input):
        out = self.model1(input, self.edge_index1)
        out = self.model2(out, self.edge_index2)
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_index1, edge_index2, unpool_hash, K=3, drop_rate=0.0):
        super(UpConv, self).__init__()
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        conv = "chebconv"
        if conv == "chebconv":
            self.model1 = Sequential("x, edge_index", [
                (ChebConv(in_channels, out_channels, K=K), "x, edge_index -> x"),
                (MeshUnpool(unpool_hash), "x -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),
            ])
            self.model2 = Sequential("x, edge_index", [
                (ChebConv(out_channels, out_channels, K=K), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (ChebConv(out_channels, out_channels, K=K), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (ChebConv(out_channels, out_channels, K=K), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (ChebConv(out_channels, out_channels, K=K), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (nn.Dropout(drop_rate), "x -> x"),
            ])
        else:
            self.model1 = Sequential("x, edge_index", [
                (GCNConv(in_channels, out_channels), "x, edge_index -> x"),
                (MeshUnpool(unpool_hash), "x -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),
            ])
            self.model2 = Sequential("x, edge_index", [
                (GCNConv(out_channels, out_channels), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (GCNConv(out_channels, out_channels), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (GCNConv(out_channels, out_channels), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (GCNConv(out_channels, out_channels), "x, edge_index -> x"),
                (nn.BatchNorm1d(out_channels), "x -> x"),
                (nn.LeakyReLU(), "x -> x"),

                (nn.Dropout(drop_rate), "x -> x"),
            ])
    
    def forward(self, input):
        out = self.model1(input, self.edge_index1)
        out = self.model2(out, self.edge_index2)
        return out


class MGCN(nn.Module):
    def __init__(self, device, smo_mesh, ini_mesh, v_mask, activation="lrelu", drop_rate=0.6, skip=False, drop=True, h0=4):
        super(MGCN, self).__init__()
        self.device = device
        self.skip = skip
        self.drop = drop
        
        pool_levels = 3
        nv = len(smo_mesh.vs)
        nvs = [int(nv*(0.6**i)) for i in range(1, pool_levels+1)]
        self.nvs = nvs
        meshes = [smo_mesh]
        p_hashes = []
        up_hashes = []
        edge_inds = [ini_mesh.edge_index.to(device)]
        v_masks_list = [v_mask.reshape(-1, 1).float()]
        v_masks = v_mask.reshape(-1, 1).float()
        f_masks = (torch.sparse.mm(ini_mesh.f2v_mat, 1.0 - v_mask.reshape(-1,1).float()) == 0).bool().reshape(-1)
        f_masks_list = [f_masks]
        os.makedirs("{}/pooled".format(os.path.dirname(smo_mesh.path)), exist_ok=True)
        for i in range(pool_levels):
            # s_mesh = meshes[i].edge_based_simplification(target_v=nvs[i])
            s_mesh = meshes[i].simplification(target_v=nvs[i])
            meshes.append(s_mesh)
            s_pmask = self.pool_hash_to_mask(s_mesh.pool_hash)
            s_umask = self.unpool_hash_to_mask(s_mesh.pool_hash)
            p_hashes.append(s_pmask.to(device))
            up_hashes.append(s_umask.to(device))
            edge_inds.append(s_mesh.edge_index.to(device))
            vm_i = v_masks_list[-1]
            vm_i_inv = torch.logical_not(vm_i).float()
            vm_j = torch.logical_not(torch.sparse.mm(s_pmask, vm_i_inv)).float()
            v_masks_list.append(vm_j)
            v_masks = torch.cat([v_masks, vm_j], dim=0)
            fm = (torch.sparse.mm(s_mesh.f2v_mat, 1.0 - vm_j.reshape(-1,1)) == 0).bool().reshape(-1)
            f_masks = torch.cat([f_masks, fm], dim=0)
            f_masks_list.append(fm)
            color = np.ones([len(s_mesh.vs), 3])
            s_mesh.vc = color * vm_j.reshape(-1, 1).detach().numpy().copy()
            s_mesh.save("{}/pooled/{}_vs.obj".format(os.path.dirname(smo_mesh.path), len(s_mesh.vs)), color=True)
        
        self.meshes = meshes
        self.p_hashes = p_hashes
        self.up_hashes = up_hashes
        self.edge_inds = edge_inds
        self.v_masks = v_masks
        self.v_masks_list = v_masks_list
        self.f_masks = f_masks
        self.f_masks_list = f_masks_list

        # self.encoder = nn.Sequential(
        #     DownConv(4, 32, edge_inds[0], edge_inds[1], p_hashes[0], drop_rate=0.0),
        #     DownConv(32, 128, edge_inds[1], edge_inds[2], p_hashes[1], drop_rate=0.2),
        #     DownConv(128, 256, edge_inds[2], edge_inds[3], p_hashes[2], drop_rate=0.2),
        # )
        # self.decoder = nn.Sequential(
        #     UpConv(256, 128, edge_inds[3], edge_inds[2], up_hashes[2], drop_rate=0.2),
        #     UpConv(128, 32, edge_inds[2], edge_inds[1], up_hashes[1], drop_rate=0.0),
        #     UpConv(32, 16, edge_inds[1], edge_inds[0], up_hashes[0], drop_rate=0.0),
        #     nn.Linear(16, 3),
        # )
        k=3
        self.encoder1 = DownConv(4, 32, edge_inds[0], edge_inds[1], p_hashes[0], K=k, drop_rate=0.0)
        self.encoder2 = DownConv(32, 128, edge_inds[1], edge_inds[2], p_hashes[1], K=k, drop_rate=0.2)
        self.encoder3 = DownConv(128, 256, edge_inds[2], edge_inds[3], p_hashes[2], K=k, drop_rate=0.2)
        
        self.decoder3 = UpConv(256, 128, edge_inds[3], edge_inds[2], up_hashes[2], K=k, drop_rate=0.2)
        self.decoder2 = UpConv(128, 32, edge_inds[2], edge_inds[1], up_hashes[1], K=k, drop_rate=0.2)
        self.decoder1 = nn.Sequential(
            UpConv(32, 16, edge_inds[1], edge_inds[0], up_hashes[0], K=k, drop_rate=0.0),
            nn.Linear(16, 3),
        )

        self.mcnn3 = Sequential("x, edge_index", [
            (ChebConv(256, 32, K=k), "x, edge_index -> x"),
            # (GCNConv(256, 32), "x, edge_index -> x"),
            (nn.BatchNorm1d(32), "x -> x"),
            (nn.LeakyReLU(), "x -> x"),
            (nn.Linear(32, 3), "x -> x"),
        ])

        self.mcnn2 = Sequential("x, edge_index", [
            (ChebConv(128, 32, K=k), "x, edge_index -> x"),
            # (GCNConv(128, 32), "x, edge_index -> x"),
            (nn.BatchNorm1d(32), "x -> x"),
            (nn.LeakyReLU(), "x -> x"),
            (nn.Linear(32, 3), "x -> x"),
        ])

        self.mcnn1 = Sequential("x, edge_index", [
            (ChebConv(32, 32, K=k), "x, edge_index -> x"),
            # (GCNConv(32, 32, K=k), "x, edge_index -> x"),
            (nn.BatchNorm1d(32), "x -> x"),
            (nn.LeakyReLU(), "x -> x"),
            (nn.Linear(32, 3), "x -> x"),
        ])

        self.skip2 = nn.Linear(256, 128)
        self.skip1 = nn.Linear(64, 32)

        """"""
        org_pos = torch.from_numpy(ini_mesh.vs).float().to(device)
        org_smpos = torch.from_numpy(smo_mesh.vs).float().to(device)
        self.poss = org_pos
        self.poss_list = [org_pos]
        self.smposs = org_smpos
        self.smposs_list = [org_smpos]
        for l in range(pool_levels):
            simp_mesh = copy.deepcopy(meshes[l+1])
            pool = MeshPool(self.p_hashes[l])
            pos = pool(org_pos)
            simp_mesh.vs = pos.detach().to("cpu").numpy().copy()
            color = np.ones([len(simp_mesh.faces), 3])
            color = color * self.f_masks_list[l+1].reshape(-1, 1).detach().numpy().copy()
            simp_mesh.save_as_ply("{}/pooled/ini_{}_vs.ply".format(os.path.dirname(simp_mesh.path), len(simp_mesh.vs)), color)
            sm_pos = torch.from_numpy(meshes[l+1].vs).float().to(device)
            self.poss = torch.cat([self.poss, pos], dim=0)
            self.poss_list.append(pos)
            self.smposs = torch.cat([self.smposs, sm_pos], dim=0)
            self.smposs_list.append(sm_pos)
            org_pos = pos

    def forward(self, data, dm=None):

        z1, _ = data.z1.to(self.device), data.x_pos.to(self.device)

        z_min, z_max = torch.min(z1, dim=0, keepdim=True)[0], torch.max(z1, dim=0, keepdim=True)[0]
        z_sc = torch.max(z_max - z_min)
        zc = (z_min + z_max) * 0.5
        z1 = (z1 - zc) / z_sc

        if type(dm) != np.ndarray:
            dm = torch.ones([z1.shape[0], 1])
        else:
            dm = torch.from_numpy(dm)
        dm = dm.to(self.device)
        z1[:, 0:3] = dm * z1[:, 0:3]
        z1 = torch.cat([z1, dm], dim=1)

        res1_enc = self.encoder1(z1)
        res2_enc = self.encoder2(res1_enc)
        res3_bot = self.encoder3(res2_enc)
        out3 = self.mcnn3(res3_bot, self.edge_inds[3])
        
        res2_dec = self.decoder3(res3_bot)
        if self.skip:
            res2_cat = torch.cat([res2_dec, res2_enc], dim=1)
            res2_dec = self.skip2(res2_cat)
        out2 = self.mcnn2(res2_dec, self.edge_inds[2])
        
        res1_dec = self.decoder2(res2_dec)
        if self.skip:
            res1_cat = torch.cat([res1_dec, res1_enc], dim=1)
            res1_dec = self.skip1(res1_cat)
        out1 = self.mcnn1(res1_dec, self.edge_inds[1])
        
        out0 = self.decoder1(res1_dec)

        pos0 = self.smposs_list[0] + out0
        pos1 = self.smposs_list[1] + out1
        pos2 = self.smposs_list[2] + out2
        pos3 = self.smposs_list[3] + out3
        return (pos0, pos1, pos2, pos3)
    
    def pool(self, dx, pool_hash):
        pool_hash = pool_hash.to(dx.device)
        v_sum = torch.sum(pool_hash.to_dense(), dim=1, keepdim=True)
        dx = torch.sparse.mm(pool_hash, dx) / v_sum
        return dx
    
    def unpool(self, dx, unpool_hash):
        unpool_hash = unpool_hash.to(dx.device)
        dx = torch.sparse.mm(unpool_hash, dx)
        return dx

    def pool_hash_to_mask(self, pool_hash):
        mask_ind = torch.stack([torch.tensor(pool_hash)[:, 1], torch.tensor(pool_hash)[:, 0]], dim=0)
        mask_val = torch.ones(mask_ind.shape[1]).float()
        mask_mat = torch.sparse.FloatTensor(mask_ind, mask_val)
        return mask_mat

    def unpool_hash_to_mask(self, pool_hash):
        mask_ind = torch.tensor(pool_hash).T
        mask_val = torch.ones(mask_ind.shape[1]).float()
        mask_mat = torch.sparse.FloatTensor(mask_ind, mask_val)
        return mask_mat