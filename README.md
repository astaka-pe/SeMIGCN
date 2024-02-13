# Learning Self-Prior for Mesh Inpainting using Self-Supervised Graph Convolutional Networks

<div align="center">

<h3>
 <a href="https://ieeexplore.ieee.org/abstract/document/10430436" target="_blank" rel="noopener noreferrer">Paper</a> | 
 <a href="https://arxiv.org/abs/2305.00635" target="_blank" rel="noopener noreferrer">arXiv</a>
 <br><br>
Accepted by IEEE TVCG 2024
</h3>
</div>

<div align="center">
    <img src="docs/semi_anim.gif" alt="gif" width=600><br>
    <h2 align="left">Method Overview</h2>
    <img src="docs/overview.png" alt="overview" width=800><br>
</div>

## Usage

### Environments

```
python==3.7
pytorch==1.7.0
torch-geometric==1.7.1
```

If you prefer to run in a newer environment, please refer to [another branch](https://github.com/astaka-pe/SeMIGCN/tree/torch-1.13-docker).

### Installation (Conda)

```
git clone https://github.com/astaka-pe/SeMIGCN
cd SeMIGCN
conda env create -f environment.yml
conda activate semigcn
```

### Preperation

- Unzip `datasets.zip`
- Sample meshes will be placed in `datasets/`
- Put your own mesh in a new arbitrary folder as:
    - Deficient mesh: `datasets/**/{mesh-name}/{mesh-name}_original.obj`
    - Ground truth: `datasets/**/{mesh-name}/{mesh-name}_gt.obj`
- The deficient and the ground truth meshes need not share a same connectivity but their scales must be shared

### Preprocess

- Specify the path of the deficient mesh
- Create **initial mesh** and **smoothed mesh**

```
python preprocess/prepare.py -i datasets/**/{mesh-name}/{mesh-name}_original.obj
```
- options
    - `-r {float}`: Target length of remeshing. The higher the coarser, the lower the finer. `default=0.6`.

- Computation time: 30 sec

### Training

```
python sgcn.py -i datasets/**/{mesh-name}   # SGCN
python mgcn.py -i datasets/**/{mesh-name}   # MGCN
```

- options
    - `-CAD`: For a CAD model
    - `-real`: For a real scan
    - `-cache`: For using cache files (for faster computation)
    - `-mu` : Weight for refinement

### Evaluation

- Create `datasets/**/{mesh-name}/comparison` and put meshes for evaluation
    - A deficient mesh `datasets/**/{mesh-name}/comparison/original.obj` and a ground truth mesh `datasets/**/{mesh-name}/comparison/gt.obj` are needed for evaluation

```
python check/batch_dist_check.py -i datasets/**/{mesh-name}
```

- options
    - `-real`: For a real scan


### Refinement (Option)

- If you want to perform only refinement, run

```
python refinement.py \\
    -src datasets/**/{mesh-name}/{mesh-name}_initial/obj \\
    -dst datasets/**/{mesh-name}/output/**/100_step/.obj \\     # SGCN
    # -dst datasets/**/{mesh-name}/output/**/100_step_0.obj \\    # MGCN
    -vm datasets/**/{mesh-name}/{mesh-name}_vmask.json \\
    -ref {arbitrary-output-filename}.obj \\
```

- option
  - `-mu`: Weight for refinement
    - Choose a weight so that the remaining vertex positions of the initial mesh and the shape of missing regions of the output mesh are saved

## Run other competitive methods

Please refer to [tinymesh](https://github.com/tatsy/tinymesh).

<!-- ### MeshFix [Attene 2010]

```
python meshfix.py -i datasets/**/{mesh-name}
```

### Context-based Coherent Surface Completion [Harary+ 2014]

```
conda activate tinymesh
python context_fill.py -i datasets/**/{mesh-name}
``` -->

## Citation

```
@article{hattori2024semigcn,
  title={Learning Self-Prior for Mesh Inpainting Using Self-Supervised Graph Convolutional Networks},
  author={Hattori, Shota and Yatagawa, Tatsuya and Ohtake, Yutaka and Suzuki, Hiromasa},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024},
  publisher={IEEE}
}
```