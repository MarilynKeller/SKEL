# SKEL



<img src="assets/Ypose_highres.png" alt="Image Description" style="width: 200px;" />

SKEL is a parametric body shape and skeleton model. Its pose parameter lets you change the body shape and its pose parameter lets you pose the skeleton in an anatomical plausible way. Given shape and pose parameters, SKEL returns joint locations, a body mesh and a skeleton mesh. SKEL is differentiable and can be fit to various data like motion capture or SMPL sequences.

For more information, please check our Siggraph 2023 paper: From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans.

[ [paper](https://download.is.tue.mpg.de/skel/main_paper.pdf) ]  [ [project page](https://skel.is.tue.mpg.de/) ] 


# Content

This repo contains the pytorch SKEL loader and the code to align it to SMPL sequences.


# Installation

## Set up the environment
Clone this repository

```
git clone https://github.com/MarilynKeller/SKEL
cd SKEL
```

Create a virtual environment and install the SKEL package
```
pip install -U pip   
python3.8 -m venv skel_venv
source skel_venv/bin/activate
pip install git+https://github.com/mattloper/chumpy 
pip install -e .
```

## Downloading SKEL
Create an account on https://skel.is.tue.mpg.de/. (Necessary for the download to work)

Then download the SKEL model from the download page with the "Download Models" button.
Extract the downloaded folder and edit the file `SKEL/skel/config.py` to specify the folder containing the downloaded SKEL model folder : `skel_folder = '/path/to/skel_models_v1.0`


To test the SKEL model, run:
``` 
python quickstart.py 
```
This runs the forward pass of SKEL and saves the output as separated body and skeleton meshes.


## Aitviewer

If you want to run the Demos, you will also need our aitviewer fork for visualization:

```
cd ..
git clone https://github.com/MarilynKeller/aitviewer-skel.git
cd aitviewer-skel 
pip install -e .
```

Edit then the file `aitviewer/aitviewer/aitvconfig.yaml` to point to the SKEL folder:

```skel_models: "/path/to/skel_models_v1.0"```

## SMPL and Mesh package
If you want to run an alignment to SMPL, you need to download the SMPL model.
First create an account on https://smpl.is.tue.mpg.de/.
Then download this file: SMPL_python_v.1.1.0.zip from the download page. And run:

```
cd ../SKEL
python scripts/setup_smpl.py /path/to/SMPL_python_v.1.1.0.zip  
```

For visualizing the fitting process you need the MPI mesh package, you can install it with the following line:

```
pip install git+https://github.com/MPI-IS/mesh.git  
```

# Demos

Visualize the effects of the pose parameters of SKEL:

```
python examples/skel_poses.py --gender male
```

![]()
<img src="assets/pose_demo.png" alt="Image Description" style="width: 50%;" />

Vizualize the shape space:

```
python examples/skel_betas.py --gender female 
```

Vizualize a SKEL sequence. You can find a sample SKEL motion in `skel_models_v1.0/sample_motion/ ` and the corresponding SMPL motion.

```
python examples/skel_sequence.py /path/to/skel_models_v1.0/sample_motion/01_01_poses_skel.pkl -z 
```

To visualize the SMPL sequence alongside : 
```
python examples/skel_sequence.py /path/to/skel_models_v1.0/sample_motion/01_01_poses_skel.pkl -z --smpl_seq /path/to/skel_models_v1.0/sample_motion/01_01_poses.npz
```

# Alignment

SKEL can be aligned to SMPL sequences. You can download SMPL sequences from the [AMASS](https://amass.is.tue.mpg.de/) Download page, and selecting the `SMPL+H G` sequences.

Here is the command to run the alignment:
```
python examples/align_to_SMPL.py /path/to/AMASS/CMU/01/01_01_poses.npz -F  
```

# Citation
If you use this software, please cite the following work and software:

```
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


# Contact 

For any question about SKEL loading, please contact skel@tuebingen.mpg.de.

For commercial licensing, please contact ps-licensing@tue.mpg.de
