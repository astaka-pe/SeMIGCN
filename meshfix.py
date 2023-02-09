import argparse
import pymeshfix as mf
import pymeshlab as ml
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
args = parser.parse_args()

for k, v in vars(args).items():
    print("{:12s}: {}".format(k, v))

""" --- create dataset --- """
mesh_name = args.input.split("/")[-1]
org_path = "{}/{}_original.obj".format(args.input, mesh_name)
os.makedirs("{}/comparison".format(args.input), exist_ok=True)
out_path = "{}/comparison/meshfix.ply".format(args.input)

ms = ml.MeshSet()
ms.load_new_mesh(org_path)
org_vm = ms.current_mesh().vertex_matrix()
org_fm = ms.current_mesh().face_matrix()
meshfix = mf.MeshFix(org_vm, org_fm)
meshfix.repair()
meshfix.save(out_path)