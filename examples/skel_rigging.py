# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import torch

from aitviewer.viewer import Viewer
from aitviewer.renderables.skel import SKELSequence
from skel.skel_model import SKEL

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Test the parametrisation of the model : smpl shape and pose')
        
    parser.add_argument('--gender', type=str, choices=['female', 'male'], required=True)
    parser.add_argument('-e', '--export_mesh', type=str, help='Export the mesh of the skel model to this folder', default=None)

    args = parser.parse_args()

    gender = args.gender
    fps = 15
 
    fpb = 30 # Frames per beta
    number_beta = 3 # Show the variation of the first 3 betas
    B = fpb*number_beta # Total number of frames
   
    # Instantiate the SKEL model
    skel_model = SKEL(gender)
    
    # Make each beta vary between -2 and 2  
    betas = torch.zeros(B, 10)   
    for i in range(3):
        bata_i_vals = betas[i*fpb:(i+1)*fpb, i] = torch.linspace(-2,2, fpb)
            
    # Set the other parameters to default values
    poses = torch.zeros((B, skel_model.num_q_params), dtype=torch.float32)
    trans = torch.zeros((B, 3), dtype=torch.float32)
    
    skel_seq = SKELSequence.t_pose(skel_layer=skel_model, skel_coloring = 'skinning_weights', skin_coloring='skinning_weights',name="SKEL - Skinning weights")

    v = Viewer()
    v.playback_fps = fps
    v.scene.add(skel_seq)
    
    v.run()
