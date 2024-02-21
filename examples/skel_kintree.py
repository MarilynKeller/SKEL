# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import torch

from aitviewer.viewer import Viewer
from aitviewer.renderables.skel import SKELSequence
from skel.skel_model import SKEL

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Display the kinematic tree of SKEL.')       
    parser.add_argument('--gender', type=str, choices=['female', 'male'], required=True)
    args = parser.parse_args()

    gender = args.gender

    # Instantiate the SKEL model
    skel_model = SKEL(gender)
    
    skel_seq = SKELSequence.t_pose(skel_layer=skel_model, name="SKEL", show_joint_angles=True, show_joint_arrows=False, 
                                   skin_color=[0,0,0,0], skel_color=[0.5,0.5,0.5,0.5])
    
    skel_seq.skin_mesh_seq.enabled=False
    skel_seq.skin_mesh_seq.draw_outline = True
    

    v = Viewer()
    v.scene.add(skel_seq)
    origin = [n for n in v.scene.nodes if n.name == 'Origin'][0]
    floor = [n for n in v.scene.nodes if n.name == 'Floor'][0]
    v.scene.remove(origin, floor)
    
    v.run()
