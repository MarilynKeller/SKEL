## SKEL

<img src="assets/Ypose_highres.png" alt="Image Description" style="width: 200px;" />

This repository is part of the code release for the Siggraph Asia 2023 paper "From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans". 

[[paper](https://download.is.tue.mpg.de/skel/main_paper.pdf)] [[project page](https://skel.is.tue.mpg.de/)]

This repo contains the pytorch SKEL loader and the code to align it to SMPL sequences.

SKEL is a parametric body shape and skeleton model. Its pose parameter lets you change the body shape and its pose parameter lets you pose the skeleton in an anatomically plausible way. Given shape and pose parameters, SKEL returns joint locations, a body mesh, and a skeleton mesh. SKEL is differentiable and can be fit to various data like motion capture or SMPL sequences.

For more information, please check our Siggraph 2023 paper: From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans.

## Table of Contents

- [Installation](#installation)
  - [Set up the environment](#set-up-the-environment)
  - [Downloading SKEL](#downloading-skel)
  - [Aitviewer](#aitviewer)
  - [SMPL and Mesh package](#smpl-and-mesh-package)
- [Demos](#demos)
  - [Pose parameters](#pose-parameters)
  - [Shape space](#shape-space)
  - [SKEL sequence](#skel-sequence)
- [Aligning SKEL to SMPL mesh](#aligning-skel-to-a-smpl-mesh)
- [Aligning SKEL to SMPL sequences](#aligning-skel-to-smpl-sequences)
- [Acknoledgments](#acknoledgments)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

### Set up the environment
Clone this repository

```shell
git clone https://github.com/MarilynKeller/SKEL
cd SKEL
```

Create a virtual environment and install the SKEL package
```shell
pip install -U pip   
python3.8 -m venv skel_venv
source skel_venv/bin/activate
pip install git+https://github.com/mattloper/chumpy 
pip install -e .
```

### Downloading SKEL
Create an account on https://skel.is.tue.mpg.de/ (Necessary for the download to work).

Then download the SKEL model from the download page with the "Download Models" button.
Extract the downloaded folder and edit the file `SKEL/skel/config.py` to specify the folder containing the downloaded SKEL model folder: `skel_folder = '/path/to/skel_models_v1.0`


To test the SKEL model, run:
```shell
python quickstart.py 
```
This runs the forward pass of SKEL and saves the output as separated body and skeleton meshes.


### Aitviewer

If you want to run the Demos, you will also need our aitviewer fork for visualization:

```shell
cd ..
git clone https://github.com/MarilynKeller/aitviewer-skel.git
cd aitviewer-skel 
pip install -e .
```

Edit then the file `aitviewer/aitviewer/aitvconfig.yaml` to point to the SKEL folder:

```skel_models: "/path/to/skel_models_v1.0"```

### SMPL and Mesh package
If you want to run an alignment to SMPL, you need to download the SMPL model.
First, create an account on https://smpl.is.tue.mpg.de/.
Then download this file: SMPL_python_v.1.1.0.zip from the download page. And run:

```shell
cd ../SKEL
python scripts/setup_smpl.py /path/to/SMPL_python_v.1.1.0.zip  
```

For visualizing the fitting process you need the MPI mesh package, you can install it with the following line:

```shell
pip install git+https://github.com/MPI-IS/mesh.git  
```

## Demos

### Pose parameters
Visualize the effects of the pose parameters of SKEL:

```shell
python examples/skel_poses.py --gender male
```

![]()
<img src="assets/pose_demo.png" alt="Image Description" style="width: 50%;" />

### Shape space
Vizualize the shape space:

```shell
python examples/skel_betas.py --gender female 
```

### Rigging
Visualize the skinning weights of the skin and bones to the SKEL kinematic tree:


```shell
python examples/skel_rigging.py --gender female  
```
<img src="assets/rigging_demo.png" alt="Vizu of SKEL skinning weights" style="width: 400px;" />


### SKEL sequence
Visualize a SKEL sequence. You can find a sample SKEL motion in `skel_models_v1.0/sample_motion/ ` and the corresponding SMPL motion.

```shell
python examples/skel_sequence.py /path/to/skel_models_v1.x/sample_motion/01_01_poses_skel.pkl -z 
```

To visualize the SMPL sequence alongside : 
```shell
python examples/skel_sequence.py /path/to/skel_models_v1.0/sample_motion/01_01_poses_skel.pkl -z --smpl_seq /path/to/skel_models_v1.0/sample_motion/01_01_poses.npz
```

## Aligning SKEL to a SMPL mesh

SKEL can be aligned to SMPL meshes:

```shell
python examples/align_to_SMPL_frame.py --smpl_data 'examples/samples/img_fit/emily-sea-coiWR0gT8Cw-unsplash_0.npz' 
```

## Aligning SKEL to SMPL sequences

SKEL can be aligned to SMPL sequences, here is an example:

```shell
python examples/align_to_SMPL_seq.py examples/samples/amass_seq/CMU_01_01.npz -D 
```

You can visualize the result with:

```shell
python examples/skel_sequence.py output/CMU_01_01/CMU_01_01_skel.pkl -z 
```

You can download more SMPL sequences of this kind from the [AMASS](https://amass.is.tue.mpg.de/) Download page, and selecting the `SMPL+H G` sequences.


## Acknoledgments
This research was done partly at [The Movement Lab](https://tml.stanford.edu/) in Stanford and [Perceiving Systems](https://ps.is.mpg.de/) at the Max Planck Institute for Intelligent Systems.

We thank Neelay Shah for generating all the necessary SMPL fits, A. A. Osman for his expertise on parametric body models, Shashank Tripathi for his help with MOYO, Giorgio Becherini for his expertise on AMASS and mocap fit evaluation, Peter Kultis, Yao Feng and Yuliang Xu for feedbacks on the paper.
We also thank the TML lab and the NMBL lab at Stanford for the fruitful discussions, especially Jennifer Maier for her expertise on the shoulder.

## Citation
If you use this software, please cite the following work and software:

```latex
@inproceedings{keller2023skel,
  title = {From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans},
  author = {Keller, Marilyn and Werling, Keenon and Shin, Soyong and Delp, Scott and 
            Pujades, Sergi and C. Karen, Liu and Black, Michael J.},
  booktitle = {ACM ToG, Proc.~SIGGRAPH Asia},
  volume = {42},
  number = {6},
  month = dec,
  year = {2023},
}
```

### License

This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE.txt](LICENSE.txt) file.


## Contact 

For any questions about SKEL loading, please contact skel@tuebingen.mpg.de.

For commercial licensing, please contact ps-licensing@tue.mpg.de
