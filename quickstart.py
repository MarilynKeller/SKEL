"""
Run a forward pass of the SKEL model with default parameters (T pose) and export the resulting meshes.
Author: Marilyn Keller
"""

import os
import torch
from skel.skel_model import SKEL
import trimesh

if __name__ == '__main__':
    
    device = 'cpu'
    gender = 'female'
    
    skel = SKEL(gender=gender).to(device)

    # Set parameters to default values (T pose)
    pose = torch.zeros(1, skel.num_q_params).to(device) # (1, 46)
    betas = torch.zeros(1, skel.num_betas).to(device) # (1, 10)
    trans = torch.zeros(1, 3).to(device)

    # SKEL forward pass
    skel_output = skel(pose, betas, trans)
    
    # Export meshes    
    os.makedirs('output', exist_ok=True)
    skin_mesh_path = os.path.join('output', f'skin_mesh_{gender}.obj')
    skeleton_mesh_path = os.path.join('output', f'skeleton_mesh_{gender}.obj')
    
    trimesh.Trimesh(vertices=skel_output.skin_verts.detach().cpu().numpy()[0], 
                    faces=skel.skin_f.cpu()).export(skin_mesh_path)
    print('Skin mesh saved to: {}'.format(skin_mesh_path))
    
    trimesh.Trimesh(vertices=skel_output.skel_verts.detach().cpu().numpy()[0],
                    faces=skel.skel_f.cpu()).export(skeleton_mesh_path)
    print('Skeleton mesh saved to: {}'.format(skeleton_mesh_path))