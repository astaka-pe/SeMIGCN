# Self-Supervised Mesh Inpainting using Multi-Scale Graph Convolutional Networks

## Usage

### Preperation

- Sample meshes are distributed in `datasets/`
- Put your own mesh in a new arbitrary folder as:
    - Deficient mesh: `datasets/**/{mesh-name}/{mesh-name}_original.obj`
    - Ground truth: `datasets/**/{mesh-name}/{mesh-name}_gt.obj`
- The deficient and the ground truth meshes need not share a same connectivity but their scales must be shared

### Preprocess

- Specify the path of the deficient mesh
- Create **initial mesh** and **smoothed mesh**

```
python preprocess/meshfix.py -i datasets/real/dragon/dragon_original.obj
```
- Computation time: 30 sec

### Training

