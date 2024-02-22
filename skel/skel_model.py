
"""
Copyright©2024 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import os
import torch.nn as nn
import torch
import numpy as np
import pickle as pkl
from typing import NewType, Optional

from skel.joints_def import curve_torch_3d, left_scapula, right_scapula
from skel.osim_rot import ConstantCurvatureJoint, CustomJoint, EllipsoidJoint, PinJoint, WalkerKnee
from skel.utils import build_homog_matrix, rotation_matrix_from_vectors, with_zeros, matmul_chain
from dataclasses import dataclass, fields

from skel.kin_skel import scaling_keypoints, pose_param_names, smpl_joint_corresp
import skel.config as cg

Tensor = NewType('Tensor', torch.Tensor)

@dataclass
class ModelOutput:
    vertices: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    full_pose: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    v_shaped: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

@dataclass
class SKELOutput(ModelOutput):
    betas: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None
    skin_verts: Optional[Tensor] = None
    skel_verts: Optional[Tensor] = None
    joints: Optional[Tensor] = None
    joints_ori: Optional[Tensor] = None
    betas: Optional[Tensor] = None
    poses: Optional[Tensor] = None
    trans : Optional[Tensor] = None
    pose_offsets : Optional[Tensor] = None
    
    
class SKEL(nn.Module):

    num_betas = 10
    
    def __init__(self, gender, model_path=None, **kwargs):
        super(SKEL, self).__init__()

        if gender not in ['male', 'female']:
            raise RuntimeError(f'Invalid Gender, got {gender}')

        self.gender = gender
        
        if model_path is None:
            # skel_file = f"/Users/mkeller2/Data/skel_models_v1.0/skel_{gender}.pkl"
            skel_file = os.path.join(cg.skel_folder, f"skel_{gender}.pkl")
        else:
            skel_file = os.path.join(model_path, f"skel_{gender}.pkl")
        assert os.path.exists(skel_file), f"Skel model file {skel_file} does not exist"
        
        skel_data = pkl.load(open(skel_file, 'rb'))

        self.num_betas = 10
        self.num_q_params = 46
        self.bone_names = skel_data['bone_names'] 
        self.num_joints = skel_data['J_regressor_osim'].shape[0]
        self.num_joints_smpl = skel_data['J_regressor'].shape[0]
        
        self.bone_axis = skel_data['bone_axis'] 
        self.joints_name = skel_data['joints_name']
        self.pose_params_name = skel_data['pose_params_name']
        
        # register the template meshes
        self.register_buffer('skin_template_v', torch.FloatTensor(skel_data['skin_template_v']))
        self.register_buffer('skin_f', torch.LongTensor(skel_data['skin_template_f']))
        
        self.register_buffer('skel_template_v', torch.FloatTensor(skel_data['skel_template_v']))
        self.register_buffer('skel_f', torch.LongTensor(skel_data['skel_template_f']))
        
        # Shape corrective blend shapes
        self.register_buffer('shapedirs', torch.FloatTensor(np.array(skel_data['shapedirs'][:,:,:self.num_betas])))
        self.register_buffer('posedirs', torch.FloatTensor(np.array(skel_data['posedirs'])))
        
        # Model sparse joints regressor, regresses joints location from a mesh
        self.register_buffer('J_regressor', torch.FloatTensor(skel_data['J_regressor']))
        
        # Regress the anatomical joint location with a regressor learned from BioAmass
        self.register_buffer('J_regressor_osim', torch.FloatTensor(skel_data['J_regressor_osim']))   
        self.register_buffer('joint_sockets', torch.FloatTensor(skel_data['joint_sockets']))
        
        self.register_buffer('per_joint_rot', torch.FloatTensor(skel_data['per_joint_rot']))
        
        # Skin model skinning weights
        self.register_buffer('weights', torch.FloatTensor(skel_data['weights']))

        # Skeleton model skinning weights
        self.register_buffer('skel_weights', torch.FloatTensor(skel_data['skel_weights']))        
        self.register_buffer('skel_weights_rigid', torch.FloatTensor(skel_data['skel_weights_rigid']))        
        
        # Kinematic tree of the model
        self.register_buffer('kintree_table', torch.from_numpy(skel_data['osim_kintree_table'].astype(np.int64)))
        # self.register_buffer('osim_kintree_table', torch.from_numpy(skel_data['osim_kintree_table'].astype(np.int64)))
        self.register_buffer('parameter_mapping', torch.from_numpy(skel_data['parameter_mapping'].astype(np.int64)))
        
        # transformation from osim can pose to T pose
        self.register_buffer('tpose_transfo', torch.FloatTensor(skel_data['tpose_transfo']))
        
        # transformation from osim can pose to A pose
        self.register_buffer('apose_transfo', torch.FloatTensor(skel_data['apose_transfo']))
        self.register_buffer('apose_rel_transfo', torch.FloatTensor(skel_data['apose_rel_transfo']))
        
        # Indices of bones which orientation should not vary with beta in T pose:
        joint_idx_fixed_beta = [0, 5, 10, 13, 18, 23]
        self.register_buffer('joint_idx_fixed_beta', torch.IntTensor(joint_idx_fixed_beta))                   
        
        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))


        # child array
        # TODO create this array in the SKEL creator
        child_array = []
        Nj = self.num_joints
        for i in range(0, Nj):
            try:
                j_array = torch.where(self.kintree_table[0] == i)[0] # candidate child lines
                if len(j_array) == 0:
                    child_index = 0
                else:
                    
                    j = j_array[0]
                    if j>=len(self.kintree_table[1]):
                        child_index = 0
                    else:
                        child_index = self.kintree_table[1,j].item()
                child_array.append(child_index)
            except:
                import ipdb; ipdb.set_trace()

        # print(f"child_array: ")
        # [print(i,child_array[i]) for i in range(0, Nj)]
        self.register_buffer('child', torch.LongTensor(child_array))
        
        # Instantiate joints
        self.joints_dict = nn.ModuleList([ 
            CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]), # 0 pelvis 
            CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]), # 1 femur_r 
            WalkerKnee(), # 2 tibia_r 
            PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]), # 3 talus_r Field taken from .osim Joint-> frames -> PhysicalOffsetFrame -> orientation 
            PinJoint(parent_frame_ori = [-1.76818999, 0.906223, 1.8196000]), # 4 calcn_r 
            PinJoint(parent_frame_ori = [-3.141589999, 0.6199010, 0]), # 5 toes_r 
            CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, -1, -1]), # 6 femur_l 
            WalkerKnee(), # 7 tibia_l 
            PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]), # 8 talus_l 
            PinJoint(parent_frame_ori = [1.768189999 ,-0.906223, 1.8196000]), # 9 calcn_l 
            PinJoint(parent_frame_ori = [-3.141589999, -0.6199010, 0]), #1 0 toes_l 
            ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), #1 1 lumbar 
            ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), #1 2 thorax 
            ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), #1 3 head 
            EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, -1, -1]), #1 4 scapula_r 
            CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]), #1 5 humerus_r 
            CustomJoint(axis=[[0.0494, 0.0366, 0.99810825]], axis_flip=[[1]]), #1 6 ulna_r 
            CustomJoint(axis=[[-0.01716099, 0.99266564, -0.11966796]], axis_flip=[[1]]), #1 7 radius_r 
            CustomJoint(axis=[[1,0,0], [0,0,-1]], axis_flip=[1, 1]), #1 8 hand_r 
            EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, 1, 1]), #1 9 scapula_l 
            CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]), #2 0 humerus_l 
            CustomJoint(axis=[[-0.0494, -0.0366, 0.99810825]], axis_flip=[[1]]), #2 1 ulna_l 
            CustomJoint(axis=[[0.01716099, -0.99266564, -0.11966796]], axis_flip=[[1]]), #2 2 radius_l 
            CustomJoint(axis=[[-1,0,0], [0,0,-1]], axis_flip=[1, 1]), #2 3 hand_l 
        ])
        
    def pose_params_to_rot(self, osim_poses):
        """ Transform the pose parameters to 3x3 rotation matrices
        Each parameter is mapped to a joint as described in joint_dict.
        The specific joint object is then used to compute the rotation matrix.
        """
    
        B = osim_poses.shape[0]
        Nj = self.num_joints
        
        ident = torch.eye(3, dtype=osim_poses.dtype).to(osim_poses.device)
        Rp = ident.unsqueeze(0).unsqueeze(0).repeat(B, Nj,1,1)
        tp = torch.zeros(B, Nj, 3).to(osim_poses.device)
        start_index = 0
        for i in range(0, Nj):
            joint_object = self.joints_dict[i]
            end_index = start_index + joint_object.nb_dof
            Rp[:, i] = joint_object.q_to_rot(osim_poses[:, start_index:end_index])
            start_index = end_index  
        return Rp, tp
    
        
    def params_name_to_index(self, param_name):
        
        assert param_name in pose_param_names
        param_index = pose_param_names.index(param_name)
        return param_index
        
        
    def forward(self, poses, betas, trans, poses_type='skel', skelmesh=True):      
        """
        B = batch size
        D = 3
        Ns : skin vertices
        Nk : skeleton vertices
        
        """
        Ns = self.skin_template_v.shape[0] # nb skin vertices
        Nk = self.skel_template_v.shape[0] # nb skeleton vertices
        Nj = self.num_joints
        B = poses.shape[0]
        device = poses.device
        
        # Check the shapes of the inputs
        assert len(betas.shape) == 2, f"Betas should be of shape (B, {self.num_betas}), but got {betas.shape}"
        assert poses.shape[0] == betas.shape[0], f"Expected poses and betas to have the same batch size, but got {poses.shape[0]} and {betas.shape[0]}"
        assert poses.shape[0] == trans.shape[0], f"Expected poses and betas to have the same batch size, but got {poses.shape[0]} and {trans.shape[0]}"
        
        # Check the device of the inputs
        assert betas.device == device, f"Betas should be on device {device}, but got {betas.device}"
        assert trans.device == device, f"Trans should be on device {device}, but got {trans.device}"  
        
        skin_v0 = self.skin_template_v[None, :]
        skel_v0 = self.skel_template_v[None, :]
        betas = betas[:, :, None] # TODO Name the expanded beta differently
        
        # TODO
        assert poses_type in ['skel', 'bsm'], f"got {poses_type}"
        if poses_type == 'bsm':
            assert poses.shape[1] == self.num_q_params - 3, f'With poses_type bsm, expected parameters of shape (B, {self.num_q_params - 3}, got {poses.shape}'
            poses_bsm = poses
            poses_skel = torch.zeros(B, self.num_q_params)
            poses_skel[:,:3] = poses_bsm[:, :3]
            trans = poses_bsm[:, 3:6] # In BSM parametrization, the hips translation is given by params 3 to 5
            poses_skel[:, 3:] = poses_bsm
            poses = poses_skel
   
        else:
            assert poses.shape[1] == self.num_q_params, f'With poses_type skel, expected parameters of shape (B, {self.num_q_params}), got {poses.shape}'
            pass
        # Load poses as expected
        # Distinction bsm skel. by default it will be bsm

        # ------- Shape ----------
        # Apply the beta offset to the template
        shapedirs  = self.shapedirs.view(-1, self.num_betas)[None, :].expand(B, -1, -1) # B x D*Ns x num_betas
        v_shaped = skin_v0 + torch.matmul(shapedirs, betas).view(B, Ns, 3)
        
        # ------- Joints ----------
        # Regress the anatomical joint location
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor_osim]) # BxJx3 # osim regressor
        # J = self.apose_transfo[:, :3, -1].view(1, Nj, 3).expand(B, -1, -1)  # Osim default pose joints location
        
        
        # Local translation
        J_ = J.clone() # BxJx3
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        t = J_[:, :, :, None] # BxJx3x1
        
        # ------- Bones transformation matrix----------
        
        # Bone initial transform to go from unposed to SMPL T pose
        Rk01 = self.compute_bone_orientation(J, J_)
         
        # BSM default pose rotations
        Ra = self.apose_rel_transfo[:, :3, :3].view(1, Nj, 3,3).expand(B, Nj, 3, 3) 

        # Local bone rotation given by the pose param
        Rp, tp = self.pose_params_to_rot(poses)  # BxNjx3x3 pose params to rotation
            
        R = matmul_chain([Rk01, Ra.transpose(2,3), Rp, Ra, Rk01.transpose(2,3)])

        ###### Compute translation for non pure rotation joints  
        t_posed = t.clone()
        
        # Scapula
        thorax_width = torch.norm(J[:, 19, :] - J[:, 14, :], dim=1) # Distance between the two scapula joints, size B
        thorax_height = torch.norm(J[:, 12, :] - J[:, 11, :], dim=1) # Distance between the two scapula joints, size B
        
        angle_abduction = poses[:,26]
        angle_elevation = poses[:,27]
        angle_rot = poses[:,28]
        angle_zero = torch.zeros_like(angle_abduction)
        t_posed[:,14] = t_posed[:,14] + \
                        (right_scapula(angle_abduction, angle_elevation, angle_rot, thorax_width, thorax_height).view(-1,3,1)
                          - right_scapula(angle_zero, angle_zero, angle_zero, thorax_width, thorax_height).view(-1,3,1))


        angle_abduction = poses[:,36]
        angle_elevation = poses[:,37]
        angle_rot = poses[:,38]
        angle_zero = torch.zeros_like(angle_abduction)
        t_posed[:,19] = t_posed[:,19] + \
                        (left_scapula(angle_abduction, angle_elevation, angle_rot, thorax_width, thorax_height).view(-1,3,1) 
                         - left_scapula(angle_zero, angle_zero, angle_zero, thorax_width, thorax_height).view(-1,3,1))
               
               
        # Knee_r
        # TODO add the Walker knee offset
        # bone_scale = self.compute_bone_scale(J_,J, skin_v0, v_shaped)
        # f1 = poses[:, 2*3+2].clone()
        # scale_femur = bone_scale[:, 2]
        # factor = 0.076/0.080 * scale_femur # The template femur medial laterak spacing #66
        # f = -f1*180/torch.pi #knee_flexion
        # varus = (0.12367*f)-0.0009*f**2
        # introt = 0.3781*f-0.001781*f**2
        # ydis = (-0.0683*f 
        #         + 8.804e-4 * f**2 
        #         - 3.750e-06*f**3
        #         )/1000*factor # up-down
        # zdis = (-0.1283*f 
        #         + 4.796e-4 * f**2)/1000*factor # 
        # import ipdb; ipdb.set_trace()
        # poses[:, 9] = poses[:, 9] + varus
        # t_posed[:,2] = t_posed[:,2] + torch.stack([torch.zeros_like(ydis), ydis, zdis], dim=1).view(-1,3,1)
        # poses[:, 2*3+2]=0
        
        # t_unposed = torch.zeros_like(t_posed)
        # t_unposed[:,2] = torch.stack([torch.zeros_like(ydis), ydis, zdis], dim=1).view(-1,3,1)
        
                        
        # Spine 
        lumbar_bending = poses[:,17]
        lumbar_extension = poses[:,18]
        angle_zero = torch.zeros_like(lumbar_bending)
        interp_t = torch.ones_like(lumbar_bending)
        l = torch.abs(J[:, 11, 1] - J[:, 0, 1]) # Length of the spine section along y axis
        t_posed[:,11] = t_posed[:,11] + \
                        (curve_torch_3d(lumbar_bending, lumbar_extension, t=interp_t, l=l)
                         - curve_torch_3d(angle_zero, angle_zero, t=interp_t, l=l))
 
        thorax_bending = poses[:,20]
        thorax_extension = poses[:,21]
        angle_zero = torch.zeros_like(thorax_bending)
        interp_t = torch.ones_like(thorax_bending)
        l = torch.abs(J[:, 12, 1] - J[:, 11, 1]) # Length of the spine section

        t_posed[:,12] = t_posed[:,12] + \
                        (curve_torch_3d(thorax_bending, thorax_extension, t=interp_t, l=l)
                         - curve_torch_3d(angle_zero, angle_zero, t=interp_t, l=l))                                               

        head_bending = poses[:, 23]
        head_extension = poses[:,24]
        angle_zero = torch.zeros_like(head_bending)
        interp_t = torch.ones_like(head_bending)
        l = torch.abs(J[:, 13, 1] - J[:, 12, 1]) # Length of the spine section
        t_posed[:,13] = t_posed[:,13] + \
                        (curve_torch_3d(head_bending, head_extension, t=interp_t, l=l)
                         - curve_torch_3d(angle_zero, angle_zero, t=interp_t, l=l)) 
                                                    
  
        # ------- Body surface transformation matrix----------           
                                
        G_ = torch.cat([R, t_posed], dim=-1) # BxJx3x4 local transformation matrix
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(B, Nj, -1, -1) # BxJx1x4
        G_ = torch.cat([G_, pad_row], dim=2) # BxJx4x4 padded to be 4x4 matrix an enable multiplication for the kinematic chain
        
        # Global transform
        G = [G_[:, 0].clone()]
        for i in range(1, Nj):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)
        
        # ------- Pose dependant blend shapes ----------
        # Note : Those should be retrained for SKEL as the SKEL joints location are different from SMPL.
        # But the current version lets use get decent pose dependant deformations for the shoulders, belly and knies
        ident = torch.eye(3, dtype=v_shaped.dtype, device=device)
        
        # We need the per SMPL joint bone transform to compute pose dependant blend shapes.
        # Initialize each joint rotation with identity
        Rsmpl = ident.unsqueeze(0).unsqueeze(0).expand(B, self.num_joints_smpl, -1, -1) # BxNjx3x3 
        
        Rskin = G_[:, :, :3, :3] # BxNjx3x3
        Rsmpl[:, smpl_joint_corresp] = Rskin.clone()[:] # BxNjx3x3 pose params to rotation
        pose_feature = Rsmpl[:, 1:].view(B, -1, 3, 3) - ident
        pose_offsets = torch.matmul(pose_feature.view(B, -1),
                                    self.posedirs.view(Ns*3, -1).T).view(B, -1, 3)
        v_shaped_pd = v_shaped + pose_offsets
          
        
        ##########################################################################################
        #Transform skin mesh 
        ############################################################################################

        # Apply global transformation to the template mesh
        rest = torch.cat([J, torch.zeros(B, Nj, 1).to(device)], dim=2).view(B, Nj, 4, 1) # BxJx4x1
        zeros = torch.zeros(B, Nj, 4, 3).to(device) # BxJx4x3
        rest = torch.cat([zeros, rest], dim=-1) # BxJx4x4
        rest = torch.matmul(G, rest) # This is a 4x4 transformation matrix that only contains translation to the rest pose joint location
        Gskin = G - rest
        
        # Compute per vertex transformation matrix (after weighting)
        T = torch.matmul(self.weights, Gskin.permute(1, 0, 2, 3).contiguous().view(Nj, -1)).view(Ns, B, 4,4).transpose(0, 1)
        rest_shape_h = torch.cat([v_shaped_pd, torch.ones_like(v_shaped_pd)[:, :, [0]]], dim=-1)
        v_posed = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        
        # translation
        v_trans = v_posed + trans[:,None,:]
        
        ##########################################################################################
        #Transform joints 
        ############################################################################################

        # import ipdb; ipdb.set_trace()
        root_transform = with_zeros(torch.cat((R[:,0],J[:,0][:,:,None]),2))
        results =  [root_transform]
        for i in range(0, self.parent.shape[0]):
            transform_i =  with_zeros(torch.cat((R[:, i + 1], t_posed[:,i+1]), 2))
            curr_res = torch.matmul(results[self.parent[i]],transform_i)
            results.append(curr_res)
        results = torch.stack(results, dim=1)
        posed_joints = results[:, :, :3, 3]
        J_transformed = posed_joints + trans[:,None,:]
        
    
        ##########################################################################################
        # Transform skeleton
        ############################################################################################

        if skelmesh:
            G_bones = None
            # Shape the skeleton by scaling its bones
            skel_rest_shape_h = torch.cat([skel_v0, torch.ones_like(skel_v0)[:, :, [0]]], dim=-1).expand(B, Nk, -1) # (1,Nk,3)

            # compute the bones scaling from the kinematic tree and skin mesh
            with torch.no_grad():
                bone_scale = self.compute_bone_scale(J_, v_shaped, skin_v0)
                            
                # Apply bone meshes scaling:
                skel_v_shaped = torch.cat([(torch.matmul(bone_scale[:,:,0], self.skel_weights_rigid.T) * skel_rest_shape_h[:, :, 0])[:, :, None], 
                                        (torch.matmul(bone_scale[:,:,1], self.skel_weights_rigid.T) * skel_rest_shape_h[:, :, 1])[:, :, None],
                                        (torch.matmul(bone_scale[:,:,2], self.skel_weights_rigid.T) * skel_rest_shape_h[:, :, 2])[:, :, None],
                                        (torch.ones(B, Nk, 1).to(device))
                                        ], dim=-1) 
                
            
            # Align the bones with the proper axis
            Gk01 = build_homog_matrix(Rk01, J.unsqueeze(-1)) # BxJx4x4
            T = torch.matmul(self.skel_weights_rigid, Gk01.permute(1, 0, 2, 3).contiguous().view(Nj, -1)).view(Nk, B, 4,4).transpose(0, 1) #[1, 48757, 3, 3]
            skel_v_align = torch.matmul(T, skel_v_shaped[:, :, :, None])[:, :, :, 0]
            
            # This transfo will be applied with weights, effectively unposing the whole skeleton mesh in each joint frame. 
            # Then, per joint weighted transformation can then be applied
            G_tpose_to_unposed = build_homog_matrix(torch.eye(3).view(1,1,3,3).expand(B, Nj, 3, 3).to(device), -J.unsqueeze(-1)) # BxJx4x4
            G_skel = torch.matmul(G, G_tpose_to_unposed)            
            G_bones = torch.matmul(G, Gk01)

            T = torch.matmul(self.skel_weights, G_skel.permute(1, 0, 2, 3).contiguous().view(Nj, -1)).view(Nk, B, 4,4).transpose(0, 1)
            skel_v_posed = torch.matmul(T, skel_v_align[:, :, :, None])[:, :, :3, 0]
            
            skel_trans = skel_v_posed + trans[:,None,:]

        else:
            skel_trans = skel_v0
            Gk01 = build_homog_matrix(Rk01, J.unsqueeze(-1)) # BxJx4x4
            G_bones = torch.matmul(G, Gk01)

        joints = J_transformed
        skin_verts = v_trans
        skel_verts = skel_trans       
        joints_ori = G_bones[:,:,:3,:3]
        
        if skin_verts.max() > 1e3:
            import ipdb; ipdb.set_trace()
        
        output = SKELOutput(skin_verts=skin_verts,
                            skel_verts=skel_verts,
                            joints=joints,
                            joints_ori=joints_ori,
                            betas=betas,
                            poses=poses,
                            trans = trans,
                            pose_offsets = pose_offsets)

        return output

    
    def compute_bone_scale(self, J_, v_shaped, skin_v0):
 
        # index                         [0,  1,     2,     3      4,     5,   , ...] # todo add last one, figure out bone scale indices
        # J_ bone vectors               [j0, j1-j0, j2-j0, j3-j0, j4-j1, j5-j2, ...]
        # norm(J) = length of the bone  [j0, j1-j0, j2-j0, j3-j0, j4-j1, j5-j2, ...]
        # self.joints_sockets           [j0, j1-j0, j2-j0, j3-j0, j4-j1, j5-j2, ...]
        # self.skel_weights             [j0, j1,    j2,    j3,    j4,    j5,    ...]
        B = J_.shape[0]
        Nj = J_.shape[1]
        
        bone_scale = torch.ones(B, Nj).to(J_.device)
        
        # BSM template joints location
        osim_joints_r = self.apose_rel_transfo[:, :3, 3].view(1, Nj, 3).expand(B, Nj, 3).clone()
        
        length_bones_bsm = torch.norm(osim_joints_r, dim=-1).expand(B, -1)
        length_bones_smpl = torch.norm(J_, dim=-1) # (B, Nj)
        bone_scale_parent = length_bones_smpl / length_bones_bsm
        
        non_leaf_node = (self.child != 0)
        bone_scale[:,non_leaf_node] = (bone_scale_parent[:,self.child])[:,non_leaf_node]

        # Ulna should have the same scale as radius
        bone_scale[:,16] = bone_scale[:,17]
        bone_scale[:,16] = bone_scale[:,17]

        bone_scale[:,21] = bone_scale[:,22]
        bone_scale[:,21] = bone_scale[:,22]  
        
        # Thorax
        # Thorax scale is defined by the relative position of the thorax to its child joint, not parent joint as for other bones
        bone_scale[:, 12] = bone_scale[:, 11] 
        
        # Lumbars 
        # Lumbar scale is defined by the y relative position of the lumbar joint
        length_bones_bsm = torch.abs(osim_joints_r[:,11, 1])
        length_bones_smpl = torch.abs(J_[:, 11, 1]) # (B, Nj)
        bone_scale_lumbar = length_bones_smpl / length_bones_bsm
        bone_scale[:, 11] = bone_scale_lumbar
        
        # Expand to 3 dimensions and adjest scaling to avoid skin-skeleton intersection and handle the scaling of leaf body parts (hands, feet)
        bone_scale = bone_scale.reshape(B, Nj, 1).expand(B, Nj, 3).clone()
            
        for (ji, doi, dsi), (v1, v2) in scaling_keypoints.items():
            bone_scale[:, ji, doi] = ((v_shaped[:,v1] - v_shaped[:, v2])/ (skin_v0[:,v1] - skin_v0[:, v2]))[:,dsi] # Top over chin       
            #TODO: Add keypoints for feet scaling in scaling_keypoints
        
        # Adjust thorax front-back scaling
        # TODO fix this part
        v1 = 3027 #thorax back
        v2 = 3495 #thorax front
        
        scale_thorax_up = ((v_shaped[:,v1] - v_shaped[:, v2])/ (skin_v0[:,v1] - skin_v0[:, v2]))[:,2]  # good for large people
        v2 = 3506 #sternum
        scale_thorax_sternum = ((v_shaped[:,v1] - v_shaped[:, v2])/ (skin_v0[:,v1] - skin_v0[:, v2]))[:,2] # Good for skinny people
        bone_scale[:, 12, 0] = torch.min(scale_thorax_up, scale_thorax_sternum) # Avoids super expanded ribcage for large people and sternum outside for skinny people
                        
        #lumbars, adjust width to be same as thorax
        bone_scale[:, 11, 0] = bone_scale[:, 12, 0]
        
        return bone_scale
        
   
    
    def compute_bone_orientation(self, J, J_):        
        """Compute each bone orientation in T pose   """
        
        # method = 'unposed'
        # method = 'learned'
        method = 'learn_adjust'
        
        B = J_.shape[0]
        Nj = J_.shape[1]

        # Create an array of bone vectors the bone meshes should be aligned to.
        bone_vect = torch.zeros_like(J_) # / torch.norm(J_, dim=-1)[:, :, None] # (B, Nj, 3)
        bone_vect[:] = J_[:, self.child] # Most bones are aligned between their parent and child joint
        bone_vect[:,16] = bone_vect[:,16]+bone_vect[:,17] # We want to align the ulna to the segment joint 16 to 18
        bone_vect[:,21] = bone_vect[:,21]+bone_vect[:,22] # Same other ulna
        
        # TODO Check indices here
        # bone_vect[:,13] = bone_vect[:,12].clone() 
        bone_vect[:,12] = bone_vect.clone()[:,11].clone() # We want to align the  thorax on the thorax-lumbar segment
        # bone_vect[:,11] = bone_vect[:,0].clone() 
        
        osim_vect = self.apose_rel_transfo[:, :3, 3].clone().view(1, Nj, 3).expand(B, Nj, 3).clone()
        osim_vect[:] = osim_vect[:,self.child]
        osim_vect[:,16] = osim_vect[:,16]+osim_vect[:,17] # We want to align the ulna to the segment joint 16 to 18
        osim_vect[:,21] = osim_vect[:,21]+osim_vect[:,22] # We want to align the ulna to the segment joint 16 to 18
        
        # TODO: remove when this has been checked 
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(osim_vect[:,0,0], osim_vect[:,0,1], osim_vect[:,0,2], color='r')
        # plt.show()
        
        Gk = torch.eye(3, device=J_.device).repeat(B, Nj, 1, 1)  
        
        if method == 'unposed':
            return Gk

        elif method == 'learn_adjust':
            Gk_learned = self.per_joint_rot.view(1, Nj, 3, 3).expand(B, -1, -1, -1) #load learned rotation
            osim_vect_corr = torch.matmul(Gk_learned, osim_vect.unsqueeze(-1)).squeeze(-1)
                
            Gk[:,:] = rotation_matrix_from_vectors(osim_vect_corr, bone_vect)
            # set nan to zero
            # TODO: Check again why the following line was required
            Gk[torch.isnan(Gk)] = 0
            # Gk[:,[18,23]] = Gk[:,[16,21]] # hand has same orientation as ulna
            # Gk[:,[5,10]] = Gk[:,[4,9]] # toe has same orientation as calcaneus
            # Gk[:,[0,11,12,13,14,19]] = torch.eye(3, device=J_.device).view(1,3,3).expand(B, 6, 3, 3) # pelvis, torso and shoulder blade orientation does not vary with beta, leave it
            Gk[:, self.joint_idx_fixed_beta] =  torch.eye(3, device=J_.device).view(1,3,3).expand(B, len(self.joint_idx_fixed_beta), 3, 3) # pelvis, torso and shoulder blade orientation should not vary with beta, leave it
               
            Gk = torch.matmul(Gk, Gk_learned)

        elif method == 'learned':
            """ Apply learned transformation"""       
            Gk = self.per_joint_rot.view(1, Nj, 3, 3).expand(B, -1, -1, -1)
            
        else:
            raise NotImplementedError
        
        return Gk
    