import torch
import numpy as np
import json
import argparse

from util.mesh import Mesh
import util.models as Models
import util.datamaker as Datamaker

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str, required=True)
    parser.add_argument("-dst", type=str, required=True)
    parser.add_argument("-vm", type=str, required=True)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("-mu", type=float, default=1.0)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print("{:12s}: {}".format(k, v))
    
    return args

def main():
    args = get_parse()
    dst_path = args.dst
    src_path = args.src
    vm_path = args.vm
    ref_path = args.ref
    dst_mesh = Mesh(dst_path, build_mat=False)
    src_mesh = Mesh(src_path)
    with open(vm_path, "r") as f:
        v_mask = np.array(json.load(f))
    v_mask = torch.from_numpy(v_mask).reshape(-1)
    dst_pos = torch.from_numpy(dst_mesh.vs).float()
    ref_pos = Mesh.mesh_merge(src_mesh.Lap, src_mesh, dst_pos, v_mask, w=args.mu)
    dst_mesh.vs = ref_pos.detach().numpy().copy()
    Mesh.save(dst_mesh, ref_path)


if __name__ == "__main__":
    main()