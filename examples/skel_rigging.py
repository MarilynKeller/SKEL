# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import torch

from aitviewer.viewer import Viewer
from aitviewer.renderables.skel import SKELSequence
from skel.skel_model import SKEL

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Display the skinning weights on the SKEL model skin and bones.')
        
    parser.add_argument('--gender', type=str, choices=['female', 'male'], required=True)

    args = parser.parse_args()

    gender = args.gender

    # Instantiate the SKEL model
    skel_model = SKEL(gender)
    
    skel_seq = SKELSequence.t_pose(skel_layer=skel_model, skel_coloring = 'skinning_weights', skin_coloring='skinning_weights',name="SKEL - Skinning weights",
                                   show_joint_arrows=False)

    v = Viewer()
    v.scene.add(skel_seq)
    
    v.run()
