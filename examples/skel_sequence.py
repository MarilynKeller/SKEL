# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os

import numpy as np
import torch

from aitviewer.viewer import Viewer
from aitviewer.renderables.skel import SKELSequence
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.configuration import CONFIG as C
from skel.skel_model import SKEL


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Visualize a SKEL sequence.')
    
    parser.add_argument('skel_file', type=str, help='Path to the SKEL sequence to visualize.')
    parser.add_argument('--smpl_seq', type=str, help='The corresponding SMPL sequence', default=None)
    parser.add_argument('--fps', type=int, help='Fps of the sequence', default=120)
    parser.add_argument('-z', '--z-up', help='Use Z-up coordinate system. \
        This is usefull for vizualizing sequences of AMASS that are 90 degree rotated', action='store_true')
    parser.add_argument('-g', '--gender', type=str, default=None, help='Forces the gender for visualization. By default, the code tries to load the gender from the skel file')
    parser.add_argument('-e', '--export_mesh', type=str, help='Export the mesh of the skel model to this folder', default=None)
    parser.add_argument('--offset', help='Offset the SMPL model to display it beside SKEL.', action='store_true') 
                        
    args = parser.parse_args()
    
    to_display = []
    
    fps_in = args.fps # Fps of the sequence
    fps_out = 30 # Fps at which the sequence will be played back
    # The skeleton mesh has a lot of vertices, so we don't load all the frames to avoid memory issues
    if args.smpl_seq is not None:
        if args.offset:
            translation = np.array([-1.0, 0.0, 0.0])
        else:
            translation = None
            
        smpl_seq = SMPLSequence.from_amass(
                        npz_data_path=args.smpl_seq,
                        fps_out=fps_out,
                        name="SMPL",
                        show_joint_angles=True,
                        position=translation,
                        z_up=args.z_up
                        )   
        to_display.append(smpl_seq)
        

    skel_seq = SKELSequence.from_file(skel_seq_file = args.skel_file, 
                                     poses_type='skel', 
                                     fps_in=fps_in,
                                     fps_out=fps_out,
                                     is_rigged=True, 
                                     show_joint_angles=True, 
                                     name='SKEL', 
                                     z_up=args.z_up)
    to_display.append(skel_seq)

    v = Viewer()
    v.playback_fps = fps_out
    v.scene.add(*to_display)
    v.scene.camera.position = np.array([-5, 1.7, 0.0])
    v.lock_to_node(skel_seq, (2, 0.7, 2), smooth_sigma=5.0)
    
    v.run_animations = True 
    v.run()
