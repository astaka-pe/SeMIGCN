import argparse
import os
from tinymesh import Mesh, hole_fill_context_coherent

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
args = parser.parse_args()

for k, v in vars(args).items():
    print("{:12s}: {}".format(k, v))

""" --- create dataset --- """
mesh_name = args.input.split("/")[-1]
org_path = "{}/{}_original.obj".format(args.input, mesh_name)
os.makedirs("{}/comparison".format(args.input), exist_ok=True)
out_path = "{}/comparison/context.obj".format(args.input)

mesh = Mesh(org_path)
hole_fill_context_coherent(mesh, maxiters=200)
mesh.save(out_path)