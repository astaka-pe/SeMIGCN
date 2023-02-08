import pymeshlab as ml
import numpy as np
import argparse
import glob
import os
import json
import sys
import trimesh

#EPS = 0.05    # for simulated hole
EPS = 1.0   # for real hole
COLOR_MAX = 0.005

def get_parser():
    parser = argparse.ArgumentParser(description="calculate hausdorff distances")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-real", action="store_true")
    args = parser.parse_args()
    for k, v in vars(args).items():
        print("{:12s}: {}".format(k, v))
    return args

def main():
    """ calculate hausdorff distances """
    args = get_parser()
    if args.real:
        EPS = 0.05
    else:
        EPS = 0.01
    i_dir = "{}/comparison".format(args.input)
    o_dir = "{}/colored".format(i_dir)
    os.makedirs(o_dir, exist_ok=True)
    m_name = i_dir.split("/")[-2]
    gt_path = "{}/gt.obj".format(i_dir, m_name)
    org_path = "{}/original.obj".format(i_dir, m_name)
    all_mesh = glob.glob("{}/*.*".format(i_dir))

    ms = ml.MeshSet()
    ms.load_new_mesh(gt_path)

    face_num = ms.current_mesh().face_number()
    diag = ms.current_mesh().bounding_box().diagonal()
    
    ms.load_new_mesh(org_path)
    res = ms.apply_filter("distance_from_reference_mesh", measuremesh=0, refmesh=1, signeddist=False)
    ms.set_current_mesh(0)
    quality = ms.current_mesh().vertex_quality_array()
    new_vs = quality > EPS
    new_pos = ms.current_mesh().vertex_matrix()[new_vs]
    with open("{}/inserted.obj".format(o_dir), "w") as f_obj:
        for v in new_pos:
            print("v {} {} {}".format(v[0], v[1], v[2]), file=f_obj)

    max_val = diag * COLOR_MAX
    with open("{}/max_val.txt".format(o_dir), mode="w") as f:
        f.write("{:.7f}".format(max_val))

    for m_path in all_mesh:
        if m_path == gt_path or m_path == org_path:
            continue
        try:
            ms.load_new_mesh(m_path)
        except:
            print("[ERROR] {} is unknown format".format(m_path))
            continue
        res1 = ms.apply_filter("distance_from_reference_mesh", measuremesh=0, refmesh=ms.current_mesh_id())
        res1 = ms.apply_filter("distance_from_reference_mesh", measuremesh=ms.current_mesh_id(), refmesh=0)
        quality = ms.mesh(0).vertex_quality_array()
        quality_hole = quality[new_vs]
        hd_all = np.sum(np.abs(quality)) / len(quality) / diag
        hd_hole = np.sum(np.abs(quality_hole)) / len(quality_hole) / diag
        ms.apply_filter("colorize_by_vertex_quality", minval=0, maxval=max_val, zerosym=True)
        out_file = os.path.basename(m_path)
        out_path = "{}/all={:.6f}-hole={:.6f}-{}".format(o_dir, hd_all, hd_hole, out_file)
        ms.save_current_mesh(out_path)
    
if __name__ == "__main__":
    main()