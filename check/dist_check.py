from cv2 import COLOR_COLORCVT_MAX
import pymeshlab as ml
import numpy as np
import argparse
import glob
import os
import json
import sys

EPS = 0.05
COLOR_MAX = 0.005

def simple_mesh_distance(out_path, gt_path):
    """ calculate hausdorff distance """
    ms = ml.MeshSet()
    ms.load_new_mesh(gt_path)
    face_num = ms.current_mesh().face_number()
    diag = ms.current_mesh().bounding_box().diagonal()    
    max_val = diag * COLOR_COLORCVT_MAX

    ms.load_new_mesh(out_path)
    
    res1 = ms.apply_filter("distance_from_reference_mesh", measuremesh=0, refmesh=ms.current_mesh_id())
    res1 = ms.apply_filter("distance_from_reference_mesh", measuremesh=ms.current_mesh_id(), refmesh=0)
    quality = ms.mesh(0).vertex_quality_array()
    hd_all = np.sum(np.abs(quality)) / len(quality) / diag
    ms.apply_filter("colorize_by_vertex_quality", minval=0, maxval=max_val, zerosym=True)
    out_dir = os.path.dirname(out_path)
    out_file = os.path.basename(out_path)
    os.makedirs("{}/colored".format(out_dir), exist_ok=True)
    out_path = "{}/colored/all={:.6f}-{}".format(out_dir, hd_all, out_file)
    ms.save_current_mesh(out_path)
    return hd_all

def mesh_distance(gt_path, org_path, out_path, real=False):
    """ calculate hausdorff distance """
    if real:
        EPS = 0.05
    else:
        EPS = 0.01
    ms = ml.MeshSet()
    ms.load_new_mesh(gt_path)

    face_num = ms.current_mesh().face_number()
    diag = ms.current_mesh().bounding_box().diagonal()
    
    ms.load_new_mesh(org_path)
    res = ms.apply_filter("distance_from_reference_mesh", measuremesh=0, refmesh=1, signeddist=False)
    ms.set_current_mesh(0)
    quality = ms.current_mesh().vertex_quality_array()
    new_vs = quality > EPS

    max_val = diag * 0.005

    ms.load_new_mesh(out_path)
    
    res1 = ms.apply_filter("distance_from_reference_mesh", measuremesh=0, refmesh=ms.current_mesh_id())
    res1 = ms.apply_filter("distance_from_reference_mesh", measuremesh=ms.current_mesh_id(), refmesh=0)
    quality = ms.mesh(0).vertex_quality_array()
    quality_hole = quality[new_vs]
    hd_all = np.sum(np.abs(quality)) / len(quality) / diag
    hd_hole = np.sum(np.abs(quality_hole)) / len(quality_hole) / diag
    ms.apply_filter("colorize_by_vertex_quality", minval=0, maxval=max_val, zerosym=True)
    out_dir = os.path.dirname(out_path)
    out_file = os.path.basename(out_path)
    out_path = "{}/all={:.6f}-hole={:.6f}-{}".format(out_dir, hd_all, hd_hole, out_file)
    ms.save_current_mesh(out_path)