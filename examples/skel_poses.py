# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import math
import os
import numpy as np
import torch

from aitviewer.viewer import Viewer
from aitviewer.renderables.skel import SKELSequence
from skel.skel_model import SKEL
from skel.kin_skel import pose_limits

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Test the parametrisation of the model : smpl shape and pose')
        
    parser.add_argument('--gender', type=str, choices=['female', 'male'], required=True)
    parser.add_argument('-e', '--export_mesh', type=str, help='Export the mesh of the skel model to this folder', default=None)

    args = parser.parse_args()

    gender = args.gender
    fps = 15
   
    # Instantiate the SKEL model
    skel_model = SKEL(gender)

    # List of the parameters to vary
    parameters_to_vary = [
                            # 'pelvis_rotation',
                            # 'pelvis_list',
                            # 'pelvis_tilt',
                            
                            #Back 
                            'lumbar_bending',
                            'lumbar_extension',
                            'lumbar_twist',
        
                            'thorax_bending',
                            # 'thorax_extension',
                            # 'thorax_twist',
                            
                            # 'head_bending',
                            # 'head_extension',
                            'head_twist',
                            
                            #Shoulder
                            # 'shoulder_l_x',
                            # 'shoulder_l_y',
                            # 'shoulder_l_z',
                            
                            # 'shoulder_r_x',
                            # 'shoulder_r_y',
                            # 'shoulder_r_z',

                            'scapula_abduction_r',
                            'scapula_elevation_r',
                            'scapula_upward_rot_r',
                            
                            # 'scapula_abduction_l',
                            # 'scapula_elevation_l',
                            # 'scapula_upward_rot_l',                           
                            
                            #Arm
                            'elbow_flexion_r',
                            'pro_sup_r',
                            'wrist_flexion_r',
                            'wrist_deviation_r',
                            
                            # 'elbow_flexion_l',
                            # 'pro_sup_l',                              
                            # 'wrist_flexion_l',
                            # 'wrist_deviation_l',
                                    
                            #Leg                         
                            # 'hip_flexion_r',
                            # 'hip_adduction_r',
                            # 'hip_rotation_r',
                            
                            # 'hip_adduction_l',
                            # 'hip_rotation_l',
                            # 'hip_flexion_l',
                            
                            'knee_angle_r',
                            'knee_angle_l',
                            
                            #Feet                           
                            'ankle_angle_r',
                            'subtalar_angle_r',
                            'mtp_angle_r', 
                            # 'ankle_angle_l',
                            # 'subtalar_angle_l', 
                            # 'mtp_angle_l',                                                                   
                                
                
                            ]
            
    # Create a synthetic pose sequence, pose parameter by pose parameter            
    min_val = -math.pi
    max_val = math.pi        

    frames_per_bone = 30
    assert frames_per_bone % 2 == 0
    poses = np.zeros((len(parameters_to_vary) * frames_per_bone, skel_model.num_q_params))
    for p_i, param in enumerate(parameters_to_vary):
        ji = skel_model.params_name_to_index(param)
        f_start = p_i * frames_per_bone
        f_end = f_start + frames_per_bone
        if param in pose_limits.keys():
            min_val_param = pose_limits[param][0]
            max_val_param = pose_limits[param][1]
        else:
            min_val_param = min_val
            max_val_param = max_val
                            
        poses[f_start:f_end, ji] = np.concatenate([
            np.linspace(min_val_param, max_val_param, frames_per_bone//2), 
            np.linspace(max_val_param, min_val_param, frames_per_bone//2)
            ], axis=0)          
            
    # Set the other parameters to default values
    betas = torch.zeros((poses.shape[0], 10), dtype=torch.float32)
    trans = torch.zeros((poses.shape[0], 3), dtype=torch.float32)
    
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
