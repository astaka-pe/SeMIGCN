import numpy as np
import torch
import torch.nn.functional as F
import copy
import pymeshlab as ml
from util.mesh import Mesh
from typing import Union
from scipy.sparse import csr_matrix


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

def annotation_color_loss(pred_color: Union[torch.Tensor, np.ndarray], real_color: np.ndarray, ltype="l1mae") -> torch.Tensor:
    """ reconstructuion error for vertex positions """
    if type(pred_color) == np.ndarray:
        pred_color = torch.from_numpy(pred_color)
    mask = np.nonzero(np.sum(real_color, axis=1))[0]
    real_color = torch.from_numpy(real_color).to(pred_color.device)

    if ltype == "l1mae":
        diff_color = torch.sum(torch.abs(real_color[mask] - pred_color[mask]), dim=1)
        loss = torch.sum(diff_color) / len(diff_color)

    elif ltype == "rmse":
        diff_color = torch.abs(real_color[mask] - pred_color[mask])
        diff_color = diff_color ** 2
        diff_color = torch.sum(diff_color, dim=1)
        diff_color = torch.sum(diff_color) / len(diff_color)
        loss = torch.sqrt(diff_color + 1.0e-6)
    else:
        print("[ERROR]: ltype error")
        exit()
    
    return loss

def masked_color_loss(pred_color: Union[torch.Tensor, np.ndarray], real_color: np.ndarray, mask:np.ndarray, ltype="l1mae") -> torch.Tensor:
    """ reconstructuion error for vertex positions """
    if type(pred_color) == np.ndarray:
        pred_color = torch.from_numpy(pred_color)
    real_color = torch.from_numpy(real_color).to(pred_color.device)

    if ltype == "l1mae":
        diff_color = torch.sum(torch.abs(real_color[mask] - pred_color[mask]), dim=1)
        loss = torch.sum(diff_color) / len(diff_color)

    elif ltype == "rmse":
        diff_color = torch.abs(real_color[mask] - pred_color[mask])
        diff_color = diff_color ** 2
        diff_color = torch.sum(diff_color, dim=1)
        diff_color = torch.sum(diff_color) / len(diff_color)
        loss = torch.sqrt(diff_color + 1.0e-6)
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

#TODO: rename this as "consistency loss"
def pos_norm_loss(pos: Union[torch.Tensor, np.ndarray], norm: Union[torch.Tensor, np.ndarray], mesh: Mesh, ltype="mae") -> torch.Tensor:
    """ loss between vertex position and face normal """
    if type(pos) == np.ndarray:
        pos = torch.from_numpy(pos)
    if type(norm) == np.ndarray:
        norm = torch.from_numpy(norm).to(pos.device)
    fc = torch.sum(pos[mesh.faces], 1) / 3.0
    pc = pos[mesh.faces] - fc.reshape(-1, 1, 3)
    dot_f2v = torch.abs(torch.sum(pc * norm.reshape(-1, 1, 3), dim=2))
    mat_vals = dot_f2v.reshape(-1)

    if ltype == "mae":
        loss = torch.sum(mat_vals) / len(mesh.vs)
    elif ltype == "rmse":
        loss = torch.sum(mat_vals ** 2) / len(mat_vals)
        loss = torch.sqrt(loss + 1.0e-6)
    else:
        print("[ERROR]: ltype error")
        exit()

    return loss

def mask_consistency_loss(pos: Union[torch.Tensor, np.ndarray], norm: Union[torch.Tensor, np.ndarray], mesh: Mesh, mask: np.ndarray, ltype="mae") -> torch.Tensor:
    """ loss between vertex position and face normal """
    if type(pos) == np.ndarray:
        pos = torch.from_numpy(pos)
    if type(norm) == np.ndarray:
        norm = torch.from_numpy(norm).to(pos.device)
    fc = torch.sum(pos[mesh.faces], 1) / 3.0
    pc = pos[mesh.faces] - fc.reshape(-1, 1, 3)
    dot_f2v = torch.abs(torch.sum(pc * norm.reshape(-1, 1, 3), dim=2))
    mat_vals = dot_f2v[mask].reshape(-1)

    if ltype == "mae":
        loss = torch.sum(mat_vals) / len(mesh.vs)
    elif ltype == "rmse":
        loss = torch.sum(mat_vals ** 2) / len(mat_vals)
        loss = torch.sqrt(loss + 1.0e-6)
    else:
        print("[ERROR]: ltype error")
        exit()

    return loss

def weighted_norm_rec_loss(pred_norm: Union[torch.Tensor, np.ndarray], real_norm: Union[torch.Tensor, np.ndarray], mask: np.ndarray) -> torch.Tensor:
    """ cosine distance for (vertex, face) normal """
    if type(pred_norm) == np.ndarray:
        pred_norm = torch.from_numpy(pred_norm)
    if type(real_norm) == np.ndarray:
        real_norm = torch.from_numpy(real_norm).to(pred_norm.device)
    mask = torch.from_numpy(mask).to(pred_norm.device)
    cos_loss = 1.0 - torch.sum(torch.mul(pred_norm, real_norm), dim=1)
    cos_loss = cos_loss * mask
    loss = torch.sum(cos_loss, dim=0) / len(cos_loss)
    return loss

def weighted_pos_norm_loss(pos: Union[torch.Tensor, np.ndarray], norm: Union[torch.Tensor, np.ndarray], weight: np.ndarray, mesh: Mesh) -> torch.Tensor:
    """ loss between vertex position and face normal """
    if type(pos) == np.ndarray:
        pos = torch.from_numpy(pos)
    if type(norm) == np.ndarray:
        norm = torch.from_numpy(norm).to(pos.device)
    fc = torch.sum(pos[mesh.faces], 1) / 3.0
    pc = pos[mesh.faces] - fc.reshape(-1, 1, 3)
    dot_f2v = torch.abs(torch.sum(pc * norm.reshape(-1, 1, 3), dim=2))
    weight = torch.from_numpy(weight).to(pos.device)
    dot_f2v = dot_f2v * weight.reshape(-1, 1)
    mat_verts = torch.from_numpy(mesh.faces.reshape(-1)).to(pos.device)
    mat_rows = torch.zeros(len(mat_verts)).long().to(pos.device)
    mat_inds = torch.stack([mat_rows, mat_verts])
    mat_vals = dot_f2v.reshape(-1)
    
    f2v = torch.sparse.FloatTensor(mat_inds, mat_vals, size=torch.Size([1, len(mesh.vs)]))
    loss = torch.sum(f2v.to_dense()) / len(mesh.vs)
    return loss


def bnf(fn: Union[torch.Tensor, np.ndarray], mesh: Mesh, sigma_s=0.7, sigma_c=0.2, iter=1) -> torch.Tensor:
    """ bilateral normal filtering """
    if type(fn) == torch.Tensor:
        fn = fn.to("cpu").detach().numpy().copy()
    new_fn = fn
    new_mesh = copy.deepcopy(mesh)

    for _ in range(iter):
        vs = new_mesh.vs
        vf = new_mesh.vf
        fc = new_mesh.fc
        fa = new_mesh.fa
        f2f = new_mesh.f2f
        
        neig_fc = fc[f2f]
        neig_fa = fa[f2f]
        fc0_tile = np.tile(fc, (1, 3)).reshape(-1, 3, 3)
        fc_dist = np.linalg.norm(neig_fc - fc0_tile, axis=2)
        #sigma_c = np.sum(fc_dist) / (fc_dist.shape[0] * fc_dist.shape[1])
        
        """ normal updating """
        neig_fn = new_fn[f2f]
        fn0_tile = np.tile(new_fn, (1, 3)).reshape(-1, 3, 3)
        fn_dist = np.linalg.norm(neig_fn - fn0_tile, axis=2)

        wc = np.exp(-1.0 * (fc_dist ** 2) / (2 * (sigma_c ** 2)))
        ws = np.exp(-1.0 * (fn_dist ** 2) / (2 * (sigma_s ** 2)))
        
        W = np.stack([wc*ws*neig_fa, wc*ws*neig_fa, wc*ws*neig_fa], 2)

        new_fn = np.sum(W * neig_fn, 1)
        new_fn = new_fn / (np.linalg.norm(new_fn, axis=1, keepdims=True) + 1.0e-12)

        """ vertex updating """
        # TODO: fix bug for high-speed computation
        if iter == -1:
            v2f_inds = np.array(new_mesh.v2f_list[0])
            v2f_vals = np.array(new_mesh.v2f_list[1])
            v2f_areas = np.array(new_mesh.v2f_list[2]).reshape(-1, 1)
            nk = new_fn[v2f_inds[1]]
            ak = fa[v2f_inds[1]]
            v2f_data = np.sum(nk * v2f_vals, 1) * ak
            v2f_mat = csr_matrix((v2f_data, v2f_inds), shape=(len(vs), len(new_fn)))
            d_vs = v2f_mat * new_fn
            d_vs /= (v2f_areas + 1.0e-12)
            new_mesh.vs += d_vs
            Mesh.compute_face_center(new_mesh)
            Mesh.compute_face_normals(new_mesh)
        
        #TODO: use this temporarily (slow!)
        else:
            for v in range(len(vs)):
                f_list = list(vf[v])
                fc_v = fc[f_list]
                new_fn_v = new_fn[f_list]
                sumArea = np.sum(fa[f_list])
                incr = fa[f_list] * np.sum(new_fn_v * (fc_v - vs[v].reshape(-1, 3)), 1)
                incr = incr.reshape(-1, 1) * new_fn_v
                incr = np.sum(incr, 0)
                incr /= sumArea
                new_mesh.vs[v] += incr
            if iter > 1:
                Mesh.compute_face_center(new_mesh)
                Mesh.compute_face_normals(new_mesh)

    return new_fn, new_mesh

def gen_bce_loss(pred_mask: torch.Tensor, real_mask: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if type(real_mask) == np.ndarray:
        real_mask = torch.from_numpy(real_mask)
    real_mask = real_mask.to(pred_mask.device)
    t_inv = torch.logical_not(real_mask)
    loss = -1.0 * torch.sum(torch.log(pred_mask[t_inv])) / torch.sum(t_inv)
    # loss: [inf -> 0]
    return loss

def dis_bce_loss(pred_mask: torch.Tensor, real_mask: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if type(real_mask) == np.ndarray:
        real_mask = torch.from_numpy(real_mask)
    real_mask = real_mask.to(pred_mask.device)
    t = real_mask.bool()
    t_inv = torch.logical_not(real_mask)
    loss_real = -1.0 * torch.sum(torch.log(pred_mask[t])) / torch.sum(t)
    # loss_real: [inf -> 0]
    loss_fake = -1.0 * torch.sum(torch.log(1 - pred_mask[t_inv])) / torch.sum(t_inv)
    # loss_fale: [0 <- inf]
    loss = loss_real + loss_fake
    return loss

def test_loss(fn, g_fn):
    g_fn = torch.from_numpy(g_fn).to(fn.device)
    dif_fn = g_fn - fn
    dif_fn = dif_fn ** 2
    loss = torch.sum(dif_fn, dim=1)
    loss = torch.sum(loss, dim=0) / fn.shape[0]
    loss = torch.sqrt(loss + 1.0e-12)
    return loss

def mad(norm1: Union[np.ndarray, torch.Tensor], norm2: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """ mean angular distance for (face, vertex) normals """
    if type(norm1) == torch.Tensor:
        norm1 = norm1.to("cpu").detach().numpy().copy()
    if type(norm2) == torch.Tensor:
        norm2 = norm2.to("cpu").detach().numpy().copy()

    inner = np.sum(norm1 * norm2, 1)
    sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))
    mad = np.sum(sad) / len(sad)

    return mad

def angular_difference(norm1, norm2):
    inner = np.sum(norm1 * norm2, 1)
    sad = np.rad2deg(np.arccos(np.clip(inner, -1.0, 1.0)))
    return sad

def distance_from_reference_mesh(ms: ml.MeshSet):
    ms.apply_filter("distance_from_reference_mesh", measuremesh=1, refmesh=0)
    m = ms.current_mesh()
    dist = m.vertex_quality_array()
    dist = np.sum(np.abs(dist)) / len(dist)
    return dist

""" --- We don't use the loss functions below --- """

def mae_loss(pred_pos: Union[torch.Tensor, np.ndarray], real_pos: np.ndarray) -> torch.Tensor:
    """mean-absolute error for vertex positions"""
    if type(pred_pos) == np.ndarray:
        pred_pos = torch.from_numpy(pred_pos)
    diff_pos = torch.abs(real_pos - pred_pos)
    diff_pos = torch.sum(diff_pos.squeeze(), dim=1)
    mae_pos = torch.sum(diff_pos) / len(diff_pos)

    return mae_pos

def mae_loss_edge_lengths(pred_pos, real_pos, edges):
    """mean-absolute error for edge lengths"""
    pred_edge_pos = pred_pos[edges,:].clone().detach()
    real_edge_pos = real_pos[edges,:].clone().detach()

    pred_edge_lens = torch.abs(pred_edge_pos[:,0,:]-pred_edge_pos[:,1,:])
    real_edge_lens = torch.abs(real_edge_pos[:,0,:]-real_edge_pos[:,1,:])

    pred_edge_lens = torch.sum(pred_edge_lens, dim=1)
    real_edge_lens = torch.sum(real_edge_lens, dim=1)
    
    diff_edge_lens = torch.abs(real_edge_lens - pred_edge_lens)
    mae_edge_lens = torch.mean(diff_edge_lens)

    return mae_edge_lens

def var_edge_lengths(pred_pos, edges):
    """variance of edge lengths"""
    pred_edge_pos = pred_pos[edges,:].clone().detach()

    pred_edge_lens = torch.abs(pred_edge_pos[:,0,:]-pred_edge_pos[:,1,:])

    pred_edge_lens = torch.sum(pred_edge_lens, dim=1)
    
    mean_edge_len = torch.mean(pred_edge_lens, dim=0, keepdim=True)
    var_edge_len = torch.pow(pred_edge_lens - mean_edge_len, 2.0)
    var_edge_len = torch.mean(var_edge_len)

    return var_edge_len

def norm_laplacian_loss(pred_pos, ve, edges):
    """simple laplacian for output meshes"""
    pred_pos = pred_pos.T
    sub_mesh_vv = [edges[v_e, :].reshape(-1) for v_e in ve]
    sub_mesh_vv = [set(vv.tolist()).difference(set([i])) for i, vv in enumerate(sub_mesh_vv)]

    num_verts = pred_pos.size(1)
    mat_rows = [np.array([i] * len(vv), dtype=np.int64) for i, vv in enumerate(sub_mesh_vv)]
    mat_rows = np.concatenate(mat_rows)
    mat_cols = [np.array(list(vv), dtype=np.int64) for vv in sub_mesh_vv]
    mat_cols = np.concatenate(mat_cols)

    mat_rows = torch.from_numpy(mat_rows).long().to(pred_pos.device)
    mat_cols = torch.from_numpy(mat_cols).long().to(pred_pos.device)
    mat_vals = torch.ones_like(mat_rows).float()
    neig_mat = torch.sparse.FloatTensor(torch.stack([mat_rows, mat_cols], dim=0),
                                        mat_vals,
                                        size=torch.Size([num_verts, num_verts]))
    pred_pos = pred_pos.T
    sum_neigs = torch.sparse.mm(neig_mat, pred_pos)
    sum_count = torch.sparse.mm(neig_mat, torch.ones((num_verts, 1)).type_as(pred_pos))
    nnz_mask = (sum_count != 0).squeeze()
    #lap_vals = sum_count[nnz_mask, :] * pred_pos[nnz_mask, :] - sum_neigs[nnz_mask, :]
    if len(torch.where(sum_count[:, 0]==0)[0]) == 0:
        appr_norm = sum_neigs[nnz_mask, :] / sum_count[nnz_mask, :]
        appr_norm = F.normalize(appr_norm, p=2.0, dim=1)
        lap_cos = 1.0 - torch.sum(torch.mul(pred_pos, appr_norm), dim=1)
    else:
        print("[ERROR] Isorated vertices exist")
        return False
    lap_loss = torch.sum(lap_cos, dim=0) / len(lap_cos)

    return lap_loss

def fn_lap_loss(fn: torch.Tensor, f2f_mat: torch.sparse.Tensor) -> torch.Tensor:
    dif_fn = torch.sparse.mm(f2f_mat.to(fn.device), fn)
    dif_fn = torch.sqrt(torch.sum(dif_fn ** 2, dim=1))
    fn_lap_loss = torch.sum(dif_fn) / len(dif_fn)
    """
    #f2f = torch.from_numpy(f2f).long()
    n_fn = torch.sum(fn[f2f], dim=1) / 3.0
    n_fn = n_fn / torch.norm(n_fn, dim=1).reshape(-1, 1)
    dif_cos = 1.0 - torch.sum(torch.mul(fn, n_fn), dim=1)
    fn_lap_loss = torch.sum(dif_cos, dim=0) / len(dif_cos)
    """
    return fn_lap_loss

def fn_mean_filter_loss(fn: torch.Tensor, f2f_mat: torch.sparse.Tensor) -> torch.Tensor:
    f2f_mat = f2f_mat.to(fn.device)
    neig_fn = fn
    for i in range(10):
        neig_fn = torch.sparse.mm(f2f_mat, neig_fn)
        neig_fn = neig_fn / torch.norm(neig_fn, dim=1, keepdim=True)
    fn_cos = 1.0 - torch.sum(neig_fn * fn, dim=1)

    fn_mean_filter_loss = torch.sum(fn_cos, dim=0) / len(fn_cos)
    
    return fn_mean_filter_loss

def sphere_lap_loss_with_fa(uv: torch.Tensor, f2f_ext: torch.sparse.Tensor) -> torch.Tensor:
    neig_uv = torch.sparse.mm(f2f_ext.to(uv.device), uv.float())
    dif_uv = uv - neig_uv
    dif_uv = torch.norm(dif_uv, dim=1)
    fn_lap_loss = torch.sum(dif_uv, dim=0) / len(dif_uv)
    
    return fn_lap_loss

def fn_bilap_loss(fn: torch.Tensor, fc: np.ndarray, f2f: np.ndarray) -> torch.Tensor:
    fc = torch.from_numpy(fc).float().to(fn.device)
    f2f = torch.from_numpy(f2f).long().to(fn.device)
    
    neig_fc = fc[f2f]
    fc0_tile = fc.repeat(1, 3).reshape(-1, 3, 3)
    fc_dist = torch.norm(neig_fc - fc0_tile, dim=2)
    
    neig_fn = fn[f2f]
    fn0_tile = fn.repeat(1, 3).reshape(-1, 3, 3)
    fn_dist = 1.0 - torch.sum(neig_fn * fn0_tile, dim=2)
   
    sigma_c, _ = torch.max(fc_dist, dim=1)
    sigma_c = sigma_c.reshape(-1, 1)
    sigma_s = torch.std(fn_dist, dim=1).reshape(-1, 1)

    wc = torch.exp(-1.0 * (fc_dist ** 2) / (2 * (sigma_c ** 2) + 1.0e-12))
    ws = torch.exp(-1.0 * (fn_dist ** 2) / (2 * (sigma_s ** 2) + 1.0e-12))

    loss = torch.sum(wc * ws * fn_dist, dim=1) / (torch.sum(wc * ws, dim=1) + 1.0e-12)
    loss = torch.sum(loss) / len(loss)
    
    return loss

def pos4norm(vs, o_mesh, fn):
    #vs = o_mesh.vs
    vf = o_mesh.vf
    fa = torch.tensor(o_mesh.fa).float()
    faces = torch.tensor(o_mesh.faces).long()
    loss = 0.0

    for i, f in enumerate(vf):
        fa_list = fa[list(f)]
        fn_list = fn[list(f)]
        c = torch.sum(vs[faces[list(f)]], dim=1) / 3.0
        x = torch.reshape(vs[i].repeat(len(f)), (-1, 3))
        dot = torch.sum(fn_list * (c - x), 1)
        dot = torch.reshape(dot.repeat(3), (3, -1)).T
        a = torch.reshape(fa_list.repeat(3), (3, -1)).T
        error = torch.sum(a * fn_list * dot, dim=0) / torch.sum(fa_list)
        loss += torch.norm(error)
    loss /= len(vs)
    """
    for i, f in enumerate(vf):
        fa_list = fa[list(f)]
        fn_list = fn[list(f)]
        c = np.sum(vs[faces[list(f)]], 1) / 3.0
        x = np.tile(vs[i], len(f)).reshape(-1, 3)
        dot = np.sum(fn_list * (c - x), 1)
        dot = np.tile(dot, 3).reshape(3, -1).T
        a = np.tile(fa_list, 3).reshape(3, -1).T
        error = np.sum(a * fn_list * dot, 0) / np.sum(fa_list)
        loss += np.linalg.norm(error)
    loss /= len(vs)
    """

    return loss