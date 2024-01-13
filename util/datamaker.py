import glob
import numpy as np
import json
import os
from tqdm import tqdm
import torch
import copy
from .mesh import Mesh
from torch_geometric.data import Data
from typing import Tuple
from pathlib import Path

class Dataset:
    def __init__(self, data):
        self.keys = data.keys
        self.num_nodes = data.num_nodes
        self.num_edges = data.num_edges
        self.num_node_features = data.num_node_features
        self.contains_isolated_nodes = data.contains_isolated_nodes()
        self.contains_self_loops = data.contains_self_loops()
        self.z1 = data['z1']
        self.x_pos = data['x_pos']
        self.x_norm = data['x_norm']
        self.edge_index = data['edge_index']
        self.face_index = data['face_index']


def create_dataset(file_path: str, dm_size=40, kn=[1], cache=False) -> Tuple[dict, Dataset]:
    """ create mesh """
    mesh_dic = {}
    file_path = str(Path(file_path))
    mesh_name = Path(file_path).name
    ini_file = "{}/{}_initial.obj".format(file_path, mesh_name)
    smo_file = "{}/{}_smooth.obj".format(file_path, mesh_name)
    gt_file = "{}/{}_gt.obj".format(file_path, mesh_name)
    org_file = "{}/{}_original.obj".format(file_path, mesh_name)
    vmask_file = "{}/{}_vmask.json".format(file_path, mesh_name)
    fmask_file = "{}/{}_fmask.json".format(file_path, mesh_name)

    print("[Loading meshes...]")
    try:
        ini_mesh = torch.load("{}/{}_initial.pt".format(file_path, mesh_name))
        out_mesh = torch.load("{}/{}_initial.pt".format(file_path, mesh_name))
    except:
        ini_mesh = Mesh(ini_file, build_mat=False)
        out_mesh = copy.deepcopy(ini_mesh)
        torch.save(ini_mesh, "{}/{}_initial.pt".format(file_path, mesh_name))
    smo_mesh = Mesh(smo_file, manifold=False)
    Mesh.copy_attribute(ini_mesh, smo_mesh)
    
    try:
        with open(vmask_file, "r") as vm:
            v_mask = np.array(json.load(vm))
    except:
        v_mask = None
    
    try:
        with open(fmask_file, "r") as fm:
            f_mask = np.array(json.load(fm))
            color_mask(ini_mesh, f_mask)
    except:
        if type(v_mask)==np.ndarray:
            f_mask = vmask_to_fmask(ini_mesh, v_mask)
            color_mask(ini_mesh, f_mask)
        else:
            f_mask = None
        
    """ create graph """
    z1 = ini_mesh.vs - smo_mesh.vs
    z1 = torch.tensor(z1, dtype=torch.float, requires_grad=True)

    x_pos = torch.tensor(smo_mesh.vs, dtype=torch.float)
    x_norm = torch.tensor(ini_mesh.fn, dtype=torch.float)

    edge_index = torch.tensor(ini_mesh.edges.T, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1)
    face_index = torch.from_numpy(ini_mesh.f_edges)

    mesh_dic["ini_file"] = ini_file
    mesh_dic["smo_file"] = smo_file
    mesh_dic["gt_file"] = gt_file
    mesh_dic["org_file"] = org_file
    mesh_dic["v_mask"] = torch.from_numpy(v_mask).bool()
    mesh_dic["f_mask"] = f_mask
    mesh_dic["mesh_name"] = mesh_name
    mesh_dic["ini_mesh"] = ini_mesh
    mesh_dic["out_mesh"] = out_mesh
    mesh_dic["smo_mesh"] = smo_mesh
    mesh_dic["vmask_dummy"] = None
    mesh_dic["fmask_dummy"] = None

    os.makedirs("{}/dummy_mask/".format(file_path, mesh_name), exist_ok=True)
    if cache:
        mesh_dic["vmask_dummy"] = torch.load("{}/dummy_mask/vmask_dummy.pt".format(file_path))
        mesh_dic["fmask_dummy"] = torch.load("{}/dummy_mask/fmask_dummy.pt".format(file_path))
    else:
        print("[Creating synthetic occlusion...]")
        mesh_dic["vmask_dummy"], mesh_dic["fmask_dummy"] = make_dummy_mask(ini_mesh, dm_size=dm_size, kn=kn, exist_face=f_mask)
        torch.save(mesh_dic["vmask_dummy"], "{}/dummy_mask/vmask_dummy.pt".format(file_path))
        torch.save(mesh_dic["fmask_dummy"], "{}/dummy_mask/fmask_dummy.pt".format(file_path))
    
    """ create dataset """
    data = Data(x=z1, z1=z1, x_pos=x_pos, x_norm=x_norm, edge_index=edge_index, face_index=face_index, v_mask=mesh_dic["v_mask"], f_mask=mesh_dic["f_mask"], vmask_dummy=mesh_dic["v_mask"], fmask_dummy=mesh_dic["f_mask"])
    dataset = Dataset(data)
    return mesh_dic, dataset


def make_dummy_mask(mesh, dm_size=40, kn=[3, 4, 5], exist_face=None):
    valid_idx = []
    for i in kn:
        valid_idx.extend(np.arange(dm_size*i, dm_size*(i+1)).tolist())
    
    AI = mesh.AdjI.float()
    # p_list = torch.tensor([0.3, 0.06, 0.02, 0.02, 0.014, 0.008, 0.008, 0.0002, 0.001])
    #p_list = torch.tensor([0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.0007, 0.007])
    p_list = torch.tensor([0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.014, 0.0014, 0.014])
    #p_list = torch.tensor([0.021, 0.021, 0.021, 0.021, 0.021, 0.021, 0.021, 0.0021, 0.021])
    #p_list = torch.tensor([0.3, 0.06, 0.04, 0.02, 0.006, 0.008, 0.004, 0.0002, 0.001])
    vmask = torch.tensor([])
    bar = tqdm(total=sum(kn))
    for k in kn:
        Mv0 = np.random.binomial(1, p_list[k], size=[len(mesh.vs), dm_size])
        Mv0 = torch.from_numpy(Mv0).float()
        for _ in range(k):
            Mv1 = (torch.sparse.mm(AI, Mv0) > 0).float()
            Mv0 = Mv1
            bar.update(1)

        if len(vmask) == 0:
            vmask = 1.0 - Mv0
        else:
            vmask = torch.cat([vmask, 1.0 - Mv0], dim=1)

    fmask = (torch.sparse.mm(mesh.f2v_mat, 1.0 - vmask) == 0).float()
    """ write the masked meshes """
    for i, k in enumerate(kn):
        color = np.ones([len(mesh.faces), 3])
        color[:, 0] = 0.332  # 0.75
        color[:, 1] = 0.664  # 0.75
        color[:, 2] = 1.0  # 0.75
        black = fmask[:, dm_size*i] == 0
        color[black, 0] = 1.0 # 0
        color[black, 1] = 0.664 # 0
        color[black, 2] = 0.0 # 0
        color[exist_face==0, 0] = 1.0 # 0
        color[exist_face==0, 1] = 0.0 # 0.5
        color[exist_face==0, 2] = 1.0 # 1
        dropv = 100 * torch.sum(vmask[:, dm_size*i] == 0) // len(mesh.vs)
        filename = "{}/dummy_mask/{}-neighbor-{}per.ply".format(os.path.dirname(mesh.path), k, dropv)
        mesh.save_as_ply(filename, color)
    """ write mask examples """
    
    # for i in range(10):
    #     color = np.ones([len(mesh.faces), 3])
    #     color[:, 0] = 0.332
    #     color[:, 1] = 0.664
    #     color[:, 2] = 1.0
    #     black = fmask[:, i] == 0
    #     color[black, 0] = 1.0
    #     color[black, 1] = 0.664
    #     color[black, 2] = 0
    #     color[exist_face==0, 0] = 1.0
    #     color[exist_face==0, 1] = 0.0
    #     color[exist_face==0, 2] = 1.0
    #     filename = "{}/dummy_mask/mask_{}.ply".format(os.path.dirname(mesh.path), i)
    #     mesh.save_as_ply(filename, color)
    
    return vmask, fmask

def vmask_to_fmask(mesh, vmask):
    vmask = torch.from_numpy(vmask).reshape(-1, 1).float()
    fmask = (torch.sparse.mm(mesh.f2v_mat, 1.0 - vmask) == 0).bool().reshape(-1)
    return fmask

def color_mask(mesh, fmask):
    color = np.ones([len(mesh.faces), 3])
    color[:, 0] = 0.332
    color[:, 1] = 0.664
    color[:, 2] = 1.0
    color[fmask==0, 0] = 1.0
    color[fmask==0, 1] = 0.0
    color[fmask==0, 2] = 1.0
    os.makedirs("{}/dummy_mask/".format(os.path.dirname(mesh.path)), exist_ok=True)
    filename = "{}/dummy_mask/initial.ply".format(os.path.dirname(mesh.path))
    mesh.save_as_ply(filename, color)