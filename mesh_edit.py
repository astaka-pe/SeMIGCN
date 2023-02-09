import torch
import numpy as np
import json
import argparse

from util.mesh import Mesh
import util.models as Models
import util.datamaker as Datamaker

def get_parse():
    parser = argparse.ArgumentParser(description="Self-supervised Mesh Completion")
    # parser.add_argument("-org", type=str, required=True)
    # parser.add_argument("-new", type=str, required=True)
    # parser.add_argument("-vm", type=str, required=True)
    # parser.add_argument("-out", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-CAD", action="store_true")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print("{:12s}: {}".format(k, v))
    
    return args

def main():
    args = get_parse()
    mesh_name = args.input.split("/")[-1]
    # new_path = "{}/comparison/semi-single-wo-refine.obj".format(args.input)
    new_path = "{}/comparison/sgcn-wo-refine.obj".format(args.input)
    org_path = "{}/comparison/initial.obj".format(args.input)
    vm_path = "{}/{}_vmask.json".format(args.input, mesh_name)
    out_path1 = "{}/comparison/mgcn-w-refine.obj".format(args.input)
    # out_path2 = "{}/comparison/mgcn-w-vu.obj".format(args.input)
    new_mesh = Mesh(new_path, build_mat=False)
    org_mesh = Mesh(org_path)
    with open(vm_path, "r") as f:
        v_mask = np.array(json.load(f))
    v_mask = torch.from_numpy(v_mask)
    v_mask_ring = torch.sparse.mm(org_mesh.AdjI.float(), 1-v_mask.float().reshape(-1, 1)) == 0
    v_mask_ring = v_mask_ring.reshape(-1)
    new_pos = torch.from_numpy(new_mesh.vs).float()
    org_pos = torch.from_numpy(org_mesh.vs).float()
    if args.CAD:
        w = 0.01
    else:
        w = 1.0
    out_pos = Mesh.mesh_merge(org_mesh.Lap, org_mesh, new_pos, v_mask, w=w, w_b=0)
    new_mesh.vs = out_pos.detach().numpy().copy()
    Mesh.save(new_mesh, out_path1)
    # new_norm = torch.from_numpy(new_mesh.fn).float()
    # vu_pos = Models.vertex_updating(out_pos, new_norm, org_mesh)
    # new_mesh.vs[torch.logical_not(v_mask_ring)] = vu_pos[torch.logical_not(v_mask_ring)].detach().numpy().copy()
    # Mesh.save(new_mesh, out_path2)

if __name__ == "__main__":
    main()