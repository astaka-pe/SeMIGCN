import numpy as np
import torch

from util.mesh import Mesh
from typing import Union


def squared_norm(x, dim=None, keepdim=False):
    return torch.sum(x * x, dim=dim, keepdim=keepdim)

def norm(x, eps=1.0e-12, dim=None, keepdim=False):
    return torch.sqrt(squared_norm(x, dim=dim, keepdim=keepdim) + eps)

def mask_pos_rec_loss(pred_pos: torch.Tensor, real_pos: Union[torch.Tensor, np.ndarray], mask: np.ndarray, ltype="rmse") -> torch.Tensor:
    """ reconstructuion error for vertex positions """
    if type(real_pos) == np.ndarray:
        real_pos = torch.from_numpy(real_pos)
    real_pos = real_pos.to(pred_pos.device)
    
    if ltype == "l1mae":
        diff_pos = torch.sum(torch.abs(real_pos[mask] - pred_pos[mask]), dim=1)
        loss = torch.sum(diff_pos) / len(diff_pos)

    elif ltype == "rmse":
        diff_pos = torch.abs(real_pos[mask] - pred_pos[mask])
        diff_pos = diff_pos ** 2
        diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
        diff_pos = torch.sum(diff_pos) / len(diff_pos)
        loss = torch.sqrt(diff_pos + 1.0e-6)
    else:
        print("[ERROR]: ltype error")
        exit()
    
    return loss

def pos_rec_loss(pred_pos: Union[torch.Tensor, np.ndarray], real_pos: np.ndarray, ltype="rmse") -> torch.Tensor:
    """ reconstructuion error for vertex positions """
    if type(pred_pos) == np.ndarray:
        pred_pos = torch.from_numpy(pred_pos) 
    if type(real_pos) == np.ndarray:
        real_pos = torch.from_numpy(real_pos)

    real_pos = real_pos.to(pred_pos.device)

    if ltype == "l1mae":
        diff_pos = torch.sum(torch.abs(real_pos - pred_pos), dim=1)
        loss = torch.sum(diff_pos) / len(diff_pos)

    elif ltype == "rmse":
        diff_pos = torch.abs(real_pos - pred_pos)
        diff_pos = diff_pos ** 2
        diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
        diff_pos = torch.sum(diff_pos) / len(diff_pos)
        loss = torch.sqrt(diff_pos + 1.0e-6)
    else:
        print("[ERROR]: ltype error")
        exit()
    return loss

def mesh_laplacian_loss(pred_pos: torch.Tensor, mesh: Mesh, ltype="rmse") -> torch.Tensor:
    """ simple laplacian for output meshes """
    v2v = mesh.Adj.to(pred_pos.device)
    v_dims = mesh.v_dims.reshape(-1, 1).to(pred_pos.device)
    lap_pos = torch.sparse.mm(v2v, pred_pos) / v_dims
    lap_diff = torch.sum((pred_pos - lap_pos) ** 2, dim=1)
    if ltype == "mae":
        lap_diff = torch.sqrt(lap_diff + 1.0e-12)
        lap_loss = torch.sum(lap_diff) / len(lap_diff)
    elif ltype == "rmse":
        lap_loss = torch.sum(lap_diff) / len(lap_diff)
        lap_loss = torch.sqrt(lap_loss + 1.0e-12)
    else:
        print("[ERROR]: ltype error")
        exit()

    return lap_loss

def mask_norm_rec_loss(pred_norm: Union[torch.Tensor, np.ndarray], real_norm: Union[torch.Tensor, np.ndarray], mask: np.ndarray, ltype="l1mae") -> torch.Tensor:
    """ reconstruction loss for (vertex, face) normal """
    if type(pred_norm) == np.ndarray:
        pred_norm = torch.from_numpy(pred_norm)
    if type(real_norm) == np.ndarray:
        real_norm = torch.from_numpy(real_norm).to(pred_norm.device)
    
    if ltype == "l2mae":
        norm_diff = torch.sum((pred_norm[mask] - real_norm[mask]) ** 2, dim=1)
        loss = torch.sqrt(norm_diff + 1e-12)
        loss = torch.sum(loss) / len(loss)
    elif ltype == "l1mae":
        norm_diff = torch.sum(torch.abs(pred_norm[mask] - real_norm[mask]), dim=1)
        loss = torch.sum(norm_diff) / len(norm_diff)
    elif ltype == "l2rmse":
        norm_diff = torch.sum((pred_norm[mask] - real_norm[mask]) ** 2, dim=1)
        loss = torch.sum(norm_diff) / len(norm_diff)
        loss = torch.sqrt(loss + 1e-12)
    elif ltype == "l1rmse":
        norm_diff = torch.sum(torch.abs(pred_norm[mask] - real_norm[mask]), dim=1)
        loss = torch.sum(norm_diff ** 2) / len(norm_diff)
        loss = torch.sqrt(loss + 1e-12)
    elif ltype == "cos":
        cos_loss = 1.0 - torch.sum(torch.mul(pred_norm[mask], real_norm[mask]), dim=1)
        loss = torch.sum(cos_loss, dim=0) / len(cos_loss)
    else:
        print("[ERROR]: ltype error")
        exit()

    return loss

def norm_rec_loss(pred_norm: Union[torch.Tensor, np.ndarray], real_norm: Union[torch.Tensor, np.ndarray], ltype="l1mae") -> torch.Tensor:
    """ reconstruction loss for (vertex, face) normal """
    if type(pred_norm) == np.ndarray:
        pred_norm = torch.from_numpy(pred_norm)
    if type(real_norm) == np.ndarray:
        real_norm = torch.from_numpy(real_norm).to(pred_norm.device)
    
    if ltype == "l2mae":
        norm_diff = torch.sum((pred_norm - real_norm) ** 2, dim=1)
        loss = torch.sqrt(norm_diff + 1e-12)
        loss = torch.sum(loss) / len(loss)
    elif ltype == "l1mae":
        norm_diff = torch.sum(torch.abs(pred_norm - real_norm), dim=1)
        loss = torch.sum(norm_diff) / len(norm_diff)
    elif ltype == "l2rmse":
        norm_diff = torch.sum((pred_norm - real_norm) ** 2, dim=1)
        loss = torch.sum(norm_diff) / len(norm_diff)
        loss = torch.sqrt(loss + 1e-12)
    elif ltype == "l1rmse":
        norm_diff = torch.sum(torch.abs(pred_norm - real_norm), dim=1)
        loss = torch.sum(norm_diff ** 2) / len(norm_diff)
        loss = torch.sqrt(loss + 1e-12)
    elif ltype == "cos":
        cos_loss = 1.0 - torch.sum(torch.mul(pred_norm, real_norm), dim=1)
        loss = torch.sum(cos_loss, dim=0) / len(cos_loss)
    else:
        print("[ERROR]: ltype error")
        exit()

    return loss

def fn_bnf_loss(pos: torch.Tensor, fn: torch.Tensor, mesh: Mesh, ltype="l1mae", loop=5) -> torch.Tensor:
    """ bilateral loss for face normal """
    if type(pos) == np.ndarray:
        pos = torch.from_numpy(pos).to(fn.device)
    else:
        pos = pos.detach()
    fc = torch.sum(pos[mesh.faces], 1) / 3.0
    fa = torch.cross(pos[mesh.faces[:, 1]] - pos[mesh.faces[:, 0]], pos[mesh.faces[:, 2]] - pos[mesh.faces[:, 0]])
    fa = 0.5 * torch.sqrt(torch.sum(fa**2, axis=1) + 1.0e-12)
    
    #fc = torch.from_numpy(mesh.fc).float().to(fn.device)
    #fa = torch.from_numpy(mesh.fa).float().to(fn.device)
    f2f = torch.from_numpy(mesh.f2f).long().to(fn.device)
    no_neig = 1.0 * (f2f != -1)
    
    neig_fc = fc[f2f]
    neig_fa = fa[f2f] * no_neig
    fc0_tile = fc.reshape(-1, 1, 3)
    fc_dist = squared_norm(neig_fc - fc0_tile, dim=2)
    sigma_c = torch.sum(torch.sqrt(fc_dist + 1.0e-12)) / (fc_dist.shape[0] * fc_dist.shape[1])
    #sigma_c = 1.0

    new_fn = fn
    for i in range(loop):
        neig_fn = new_fn[f2f]
        fn0_tile = new_fn.reshape(-1, 1, 3)
        fn_dist = squared_norm(neig_fn - fn0_tile, dim=2)
        sigma_s = 0.3
        wc = torch.exp(-1.0 * fc_dist / (2 * (sigma_c ** 2)))
        ws = torch.exp(-1.0 * fn_dist / (2 * (sigma_s ** 2)))
        
        W = torch.stack([wc*ws*neig_fa, wc*ws*neig_fa, wc*ws*neig_fa], dim=2)

        new_fn = torch.sum(W * neig_fn, dim=1)
        new_fn = new_fn / (norm(new_fn, dim=1, keepdim=True) + 1.0e-12)

    if ltype == "mae":
        bnf_diff = torch.sum((new_fn - fn) ** 2, dim=1)
        bnf_diff = torch.sqrt(bnf_diff + 1.0e-12)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
    elif ltype == "l1mae":
        bnf_diff = torch.sum(torch.abs(new_fn - fn), dim=1)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
    elif ltype == "rmse":
        bnf_diff = torch.sum((new_fn - fn) ** 2, dim=1)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
        loss = torch.sqrt(loss + 1.0e-12)
    elif ltype == "l1rmse":
        bnf_diff = torch.sum(torch.abs(new_fn - fn), dim=1)
        loss = torch.sum(bnf_diff ** 2) / len(bnf_diff)
        loss = torch.sqrt(loss ** 2 + 1.0e-12)
    else:
        print("[ERROR]: ltype error")
        exit()
    
    return loss, new_fn

def fn_bnf_detach_loss(pos: torch.Tensor, fn: torch.Tensor, mesh: Mesh, ltype="l1mae", loop=5) -> torch.Tensor:
    """ bilateral loss for face normal """
    if type(pos) == np.ndarray:
        pos = torch.from_numpy(pos).to(fn.device)
    else:
        pos = pos.detach()
    fc = torch.sum(pos[mesh.faces], 1) / 3.0
    fa = torch.cross(pos[mesh.faces[:, 1]] - pos[mesh.faces[:, 0]], pos[mesh.faces[:, 2]] - pos[mesh.faces[:, 0]])
    fa = 0.5 * torch.sqrt(torch.sum(fa**2, axis=1) + 1.0e-12)
    
    #fc = torch.from_numpy(mesh.fc).float().to(fn.device)
    #fa = torch.from_numpy(mesh.fa).float().to(fn.device)
    f2f = torch.from_numpy(mesh.f2f).long().to(fn.device)
    no_neig = 1.0 * (f2f != -1)
    
    neig_fc = fc[f2f]
    neig_fa = fa[f2f] * no_neig
    fc0_tile = fc.reshape(-1, 1, 3)
    fc_dist = squared_norm(neig_fc - fc0_tile, dim=2)
    sigma_c = torch.sum(torch.sqrt(fc_dist + 1.0e-12)) / (fc_dist.shape[0] * fc_dist.shape[1])
    #sigma_c = 1.0

    new_fn = fn
    for i in range(loop):
        neig_fn = new_fn[f2f]
        fn0_tile = new_fn.reshape(-1, 1, 3)
        fn_dist = squared_norm(neig_fn - fn0_tile, dim=2)
        sigma_s = 0.3
        wc = torch.exp(-1.0 * fc_dist / (2 * (sigma_c ** 2)))
        ws = torch.exp(-1.0 * fn_dist / (2 * (sigma_s ** 2)))
        
        W = torch.stack([wc*ws*neig_fa, wc*ws*neig_fa, wc*ws*neig_fa], dim=2)

        new_fn = torch.sum(W * neig_fn, dim=1)
        new_fn = new_fn / (norm(new_fn, dim=1, keepdim=True) + 1.0e-12)
        new_fn = new_fn.detach()

    if ltype == "mae":
        bnf_diff = torch.sum((new_fn - fn) ** 2, dim=1)
        bnf_diff = torch.sqrt(bnf_diff + 1.0e-12)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
    elif ltype == "l1mae":
        bnf_diff = torch.sum(torch.abs(new_fn - fn), dim=1)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
    elif ltype == "rmse":
        bnf_diff = torch.sum((new_fn - fn) ** 2, dim=1)
        loss = torch.sum(bnf_diff) / len(bnf_diff)
        loss = torch.sqrt(loss + 1.0e-12)
    elif ltype == "l1rmse":
        bnf_diff = torch.sum(torch.abs(new_fn - fn), dim=1)
        loss = torch.sum(bnf_diff ** 2) / len(bnf_diff)
        loss = torch.sqrt(loss ** 2 + 1.0e-12)
    else:
        print("[ERROR]: ltype error")
        exit()
    
    return loss, new_fn