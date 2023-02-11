import torch
import numpy as np
import json
import argparse

from util.mesh import Mesh
import util.models as Models
import util.datamaker as Datamaker

def get_parse():
    parser = argparse.ArgumentParser(description="Self-supervised Mesh Completion")
    parser.add_argument("-i", "--input", type=str, required=True)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print("{:12s}: {}".format(k, v))
    
    return args

def main():
    args = get_parse()
    mesh_name = args.input.split("/")[-1]
    vm_path = "{}/{}_vmask.json".format(args.input, mesh_name)
    with open(vm_path, "r") as f:
        v_mask = np.array(json.load(f))
    
    results = {}
    results["num_verts"] = len(v_mask)
    results["num_exist"] = np.sum(v_mask)
    results["num_insert"] = len(v_mask) - np.sum(v_mask)
    results["new_ratio"] = results["num_insert"] / results["num_verts"] * 100
    results["exist_ratio"] = results["num_exist"] / results["num_verts"] * 100

    for k, v in results.items():
        print("{:12s}: {}".format(k, v))

if __name__ == "__main__":
    main()