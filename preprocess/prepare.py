import pymeshlab as ml
import pymeshfix as mf
import numpy as np
import torch
import json
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.mesh import Mesh

SMOOTH_ITER = 30
MAXHOLESIZE = 1000
EPSILON = 0.2
REMESH_TARGET = ml.Percentage(0.6)

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-r", "--remesh", type=float, default=0.6)
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print("{:12s}: {}".format(k, v))
    
    return args

def mesh_fix(ms, dirname, meshname):
    org_vm = ms.current_mesh().vertex_matrix()
    org_fm = ms.current_mesh().face_matrix()
    meshfix = mf.MeshFix(org_vm, org_fm)
    meshfix.repair()
    meshfix.save("{}/tmp/{}_fixed.ply".format(dirname, meshname))

def remesh(ms, dirname, meshname, targetlen):
    ms.load_new_mesh("{}/tmp/{}_fixed.ply".format(dirname, meshname))
    # [original, fixed]

    ms.apply_filter("remeshing_isotropic_explicit_remeshing", targetlen=targetlen)
    # [original, remeshed]

    ms.save_current_mesh("{}/tmp/{}_remeshed.obj".format(dirname, meshname))

def normalize(ms):
    ms.apply_filter("transform_scale_normalize", scalecenter="barycenter", unitflag=True, alllayers=True)
    ms.apply_filter("transform_translate_center_set_origin", traslmethod="Center on Layer BBox", alllayers=True)
    
def edge_based_scaling(mesh):
    edge_vec = mesh.vs[mesh.edges][:, 0, :] - mesh.vs[mesh.edges][:, 1, :]
    ave_len = np.sum(np.linalg.norm(edge_vec, axis=1)) / mesh.edges.shape[0]
    mesh.vs /= ave_len
    return ave_len, mesh

def normalize_scale(ms, dirname, meshname):
    try:
        ms.load_new_mesh("{}/{}_gt.obj".format(dirname, meshname))
        ms.save_current_mesh("{}/tmp/{}_gt.obj".format(dirname, meshname))
        # [original, remeshes, gt<-current]
    except:
        pass
    ms.set_current_mesh(1)
    # [original, remeshed<-current, gt]
    normalize(ms)

    ms.save_current_mesh("{}/tmp/{}_initial.obj".format(dirname, meshname))
    ms.set_current_mesh(0)
    ms.save_current_mesh("{}/tmp/{}_original.obj".format(dirname, meshname))
    try:
        ms.set_current_mesh(2)
        ms.save_current_mesh("{}/{}_gt.obj".format(dirname, meshname))
    except:
        pass
    ms.set_current_mesh(1)

    init_mesh = Mesh("{}/tmp/{}_initial.obj".format(dirname, meshname), build_mat=False)
    org_mesh = Mesh("{}/tmp/{}_original.obj".format(dirname, meshname), manifold=False)
    try:
        gt_mesh = Mesh("{}/{}_gt.obj".format(dirname, meshname), manifold=False)
    except:
        pass
    ave_len, init_mesh = edge_based_scaling(init_mesh)
    org_mesh.vs /= ave_len
    Mesh.save(init_mesh, "{}/{}_initial.obj".format(dirname, meshname))
    init_mesh.path = "{}/{}_initial.obj".format(dirname, meshname)
    torch.save(init_mesh, "{}/{}_initial.pt".format(dirname, meshname))
    Mesh.save(org_mesh, "{}/{}_original.obj".format(dirname, meshname), color=True)
    try:
        gt_mesh.vs /= ave_len
        Mesh.save(gt_mesh, "{}/{}_gt.obj".format(dirname, meshname))
    except:
        pass

def write_mask(ms, epsilon, dirname, meshname):
    ms.clear()
    ms.load_new_mesh("{}/{}_original.obj".format(dirname, meshname))
    ms.load_new_mesh("{}/{}_initial.obj".format(dirname, meshname))
    
    ms.apply_filter("distance_from_reference_mesh", measuremesh=ms.current_mesh_id(), refmesh=0, signeddist=False)
    quality = ms.current_mesh().vertex_quality_array()
    mask_vs = quality < epsilon

    with open("{}/{}_vmask.json".format(dirname, meshname), "w") as vm:
        json.dump(mask_vs.tolist(), vm)
    
    new_vs = ms.current_mesh().vertex_matrix()[np.logical_not(mask_vs)]
    with open("{}/{}_inserted.obj".format(dirname, meshname), "w") as f_obj:
        for v in new_vs:
            print("v {} {} {}".format(v[0], v[1], v[2]), file=f_obj)

def smooth(ms, dirname, meshname):
    # [original, normalized, initial]
    ms.apply_filter("laplacian_smooth", selected=True)
    ms.apply_filter("laplacian_smooth", stepsmoothnum=SMOOTH_ITER, cotangentweight=False, selected=False)
    ms.save_current_mesh("{}/{}_smooth.obj".format(dirname, meshname))

def main():
    args = get_parse()

    filename = args.input
    targetlen = ml.Percentage(args.remesh)
    dirname = os.path.dirname(filename)
    meshname = dirname.split("/")[-1]

    ms = ml.MeshSet()
    ms.load_new_mesh(filename) 
    # [original]

    os.makedirs("{}/tmp".format(dirname), exist_ok=True)
    print("[MeshFix]")
    mesh_fix(ms, dirname, meshname)
    print("[Remeshing]")
    remesh(ms, dirname, meshname, targetlen)
    print("[Normalizing scale]")
    normalize_scale(ms, dirname, meshname)
    write_mask(ms, EPSILON, dirname, meshname)
    print("[Smoothing]")
    smooth(ms, dirname, meshname)
    print("[FINISHED]")
    ms.clear()


if __name__ == "__main__":
    main()