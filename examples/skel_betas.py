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
    
    skel_seq = SKELSequence(skel_layer=skel_model, betas=betas, poses_body=poses, poses_type='skel', 
                            trans=trans, is_rigged=True, show_joint_angles=True, name='SKEL', z_up=False
                            )
    
    if args.export_mesh:
        from psbody.mesh import Mesh
        import tqdm
        print('exporting meshes')
        os.makedirs(args.export_mesh, exist_ok=True)
        for frame_i in tqdm.tqdm(range(skel_seq.poses_body.shape[0])):
            skel_mesh = Mesh(skel_seq.skel_vertices[frame_i], skel_seq.skel_faces)
            skin_mesh = Mesh(skel_seq.skin_vertices[frame_i], skel_seq.skin_faces)
            skel_mesh.write_ply(os.path.join(args.export_mesh, f'skel_{frame_i}.ply'))
            skin_mesh.write_ply(os.path.join(args.export_mesh, f'skin_{frame_i}.ply'))

    v = Viewer()
    v.playback_fps = fps
    v.scene.add(skel_seq)
    
    v.run_animations = True
    v.run()
