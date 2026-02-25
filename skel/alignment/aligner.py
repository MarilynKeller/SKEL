
"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Soyong Shin, Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import math
import os
import pickle

import numpy as np
from skel.alignment.losses import compute_anchor_pose, compute_anchor_trans, compute_pose_loss, compute_scapula_loss, compute_spine_loss, compute_time_loss, pretty_loss_print
from skel.alignment.utils import location_to_spheres, to_numpy, to_params, to_torch
import torch
from tqdm import trange
import smplx
import torch.nn.functional as F
from psbody.mesh import Mesh, MeshViewer, MeshViewers
import skel.config as cg
from skel.skel_model import SKEL
import omegaconf 

class SkelFitter(object):
    
    def __init__(self, gender, device, num_betas=10, export_meshes=False, config_path=None) -> None:

        self.smpl = smplx.create(cg.smpl_folder, model_type='smpl', gender=gender, num_betas=num_betas, batch_size=1, export_meshes=False).to(device)
        self.skel = SKEL(gender).to(device)
        self.gender = gender
        self.device = device
        self.num_betas = num_betas
        
        # Instanciate masks used for the vertex to vertex fitting
        fitting_indices = pickle.load(open(cg.fitting_mask_file, 'rb'))
        fitting_mask = torch.zeros(6890, dtype=torch.bool, device=self.device)
        fitting_mask[fitting_indices] = 1
        self.fitting_mask = fitting_mask.reshape(1, -1, 1).to(self.device) # 1xVx1 to be applied to verts that are BxVx3
        
        smpl_torso_joints = [0,3]
        verts_mask = (self.smpl.lbs_weights[:,smpl_torso_joints]>0.5).sum(dim=-1)>0
        self.torso_verts_mask = verts_mask.unsqueeze(0).unsqueeze(-1) # Because verts are of shape BxVx3
        
        self.export_meshes = export_meshes

        if config_path is None:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(package_directory, 'config.yaml')
            
        self.cfg =  omegaconf.OmegaConf.load(config_path)
           
        # Instanciate the mesh viewer to visualize the fitting
        if('DISABLE_VIEWER' in os.environ):
            self.mv = None
            print("\n DISABLE_VIEWER flag is set, running in headless mode")
        else:
            self.mv = MeshViewers((1,2),  keepalive=self.cfg.keepalive_meshviewer)
        
        
    def run_fit(self, 
            trans_in, 
            betas_in, 
            poses_in, 
            batch_size=20, 
            skel_data_init=None, 
            force_recompute=False, 
            debug=False,
            watch_frame=0,
            freevert_mesh=None):
        """Align SKEL to a SMPL sequence."""

        self.nb_frames = poses_in.shape[0]
        self.watch_frame = watch_frame
        if self.watch_frame >= self.nb_frames:
            raise ValueError(f'watch_frame {self.watch_frame} is larger than the number of frames {self.nb_frames}. Please provide a watch frame index smaller than the number of frames ({self.nb_frames})')
            
        self.is_skel_data_init = skel_data_init is not None
        self.force_recompute = force_recompute
        
        print('Fitting {} frames'.format(self.nb_frames))
        print('Watching frame: {}'.format(watch_frame))
         
        # Initialize SKEL torch params
        body_params = self._init_params(betas_in, poses_in, trans_in, skel_data_init)
    
        # We cut the whole sequence in batches for parallel optimization  
        if batch_size > self.nb_frames:
            batch_size = self.nb_frames
            print('Batch size is larger than the number of frames. Setting batch size to {}'.format(batch_size))
            
        n_batch = math.ceil(self.nb_frames/batch_size)
        pbar = trange(n_batch, desc='Running batch optimization')
        
        # Initialize the res dict to store the per frame result skel parameters
        res_dict = self._init_res_dict()
     
        # Iterate over the batches to fit the whole sequence
        for i in pbar:  
                
            # if debug:
            #     # Only run the first batch to test, ignore the rest
            #     if i > 0:
            #         print('Debug mode is on, only running the first batch. Ignoring the rest of the sequence.')
            #         break

            
            # Get batch start and end indices
            i_start =  i * batch_size
            i_end = min((i+1) * batch_size, self.nb_frames)

            # Initialize the frames with the last aligned frame from the previous batch if it exists
            if i_start>0:
                with torch.no_grad():
                        # for the translation and global rotation pose[b, 0:3], we already have it from SMPL
                        body_params['poses_skel'][i_start:i_end, 3:].copy_(body_params['poses_skel'][i_start-1:i_start, 3:])
                        body_params['betas_skel'][i_start:i_end].copy_(body_params['betas_skel'][i_start-1:i_start])
                        # body_params['trans_skel'][i_start:i_end].copy_(body_params['trans_skel'][i_start-1:i_start])

            # Fit the batch               
            betas, poses, trans, verts = self._fit_batch(body_params, i, i_start, i_end)
            
            # Store the results
            res_dict['poses'][i_start:i_end] = poses.detach()
            res_dict['betas'][i_start:i_end] = betas.detach()
            res_dict['trans'][i_start:i_end] = trans.detach()
            if self.export_meshes:
                # Store the meshes vertices
                skel_output = self.skel.forward(poses=poses, betas=betas, trans=trans, poses_type='skel', skelmesh=True)
                res_dict['skel_v'][i_start:i_end] = skel_output.skel_verts.detach()
                res_dict['skin_v'][i_start:i_end] = skel_output.skin_verts.detach()
                res_dict['smpl_v'][i_start:i_end] = verts.detach()
            
        # Convert to numpy    
        for key, val in res_dict.items():
            if isinstance(val, torch.Tensor):
                res_dict[key] = val.detach().cpu().numpy()
                
        return res_dict
        
    def _init_params(self, betas_smpl, poses_smpl, trans_smpl, skel_data_init=None):
        """ Return initial SKEL parameters from SMPL data dictionary and an optional SKEL data dictionary."""
    
        # Prepare smpl params 
        betas_smpl = to_torch(betas_smpl, self.device)
        poses_smpl = to_torch(poses_smpl, self.device)
        trans_smpl = to_torch(trans_smpl, self.device)
        
        if skel_data_init is None or self.force_recompute:
        
            poses_skel = torch.zeros((self.nb_frames, self.skel.num_q_params), device=self.device)
            # poses_skel[:, :3] = poses_smpl[:, :3] # Global orient are similar between SMPL and SKEL, so init with SMPL angles
            # poses_skel[:, 0] = -poses_smpl[:, 0] # axis deffinition is different in SKEL

            # convert SMPL pelvis rotation to SKEL root rotation
            global_orient_rodrigues = poses_smpl[:, :3].cpu().numpy()
            # use scipy to convert to euler angles in the order of xzy
            from scipy.spatial.transform import Rotation as R
            R_smpl = R.from_rotvec(global_orient_rodrigues)
            R_smpl = torch.from_numpy(R_smpl.as_matrix()).to(self.device) # (B,3,3)

            # Change of frame between SMPL and SKEL,This is equivalent to a 90 degree rotation around the x axis
            C = torch.tensor([[0., 0., 1.],
                            [ 0., 1., 0.],
                            [ -1., 0., 0.]], device=self.device, dtype=R_smpl.dtype)

            R_skel = C @ R_smpl @ C.T  # (B,3,3)

            # --- Decompose R_skel into SKEL root params where R = Ry(q2) Rx(q1) Rz(q0) ---
            q1 = torch.asin(torch.clamp(-R_skel[:, 1, 2], -1.0, 1.0))  # x
            cb = torch.cos(q1)
            eps = 1e-8
            cb_safe = torch.where(cb.abs() < eps, cb.sign() * eps, cb)

            q2 = torch.atan2(R_skel[:, 0, 2] / cb_safe, R_skel[:, 2, 2] / cb_safe)  # y
            q0 = torch.atan2(R_skel[:, 1, 0] / cb_safe, R_skel[:, 1, 1] / cb_safe)  # z

            poses_skel[:, 0] = q0
            poses_skel[:, 1] = q1
            poses_skel[:, 2] = q2

            betas_skel = torch.zeros((self.nb_frames, 10), device=self.device)
            betas_skel[:] = betas_smpl[..., :10]
            
            trans_skel = trans_smpl # Translation is similar between SMPL and SKEL, so init with SMPL translation
            
        else:
            # Load the data_init and pad it to the length of the sequence if necessary
            init_len = skel_data_init['poses'].shape[0]

            if init_len <= self.nb_frames:

                # Number of frames to pad
                pad_len = self.nb_frames - init_len

                if pad_len > 0:
                    # Repeat the last frame pad_len times
                    last_pose  = skel_data_init['poses'][-1:]
                    last_beta  = skel_data_init['betas'][-1:]

                    pose_pad = np.repeat(last_pose, pad_len, axis=0)
                    beta_pad = np.repeat(last_beta, pad_len, axis=0)

                    skel_data_init['poses'] = np.concatenate(
                        [skel_data_init['poses'], pose_pad],
                        axis=0
                    )

                    skel_data_init['betas'] = np.concatenate(
                        [skel_data_init['betas'], beta_pad],
                        axis=0
                    )

            # Load from previous alignment
            betas_skel = to_torch(skel_data_init['betas'], self.device)
            poses_skel = to_torch(skel_data_init['poses'], self.device)
            trans_skel = to_torch(skel_data_init['trans'], self.device)
            
        # Make a dictionary out of the necessary body parameters
        body_params = {
            'betas_skel': betas_skel,
            'poses_skel': poses_skel,
            'trans_skel': trans_skel,
            'betas_smpl': betas_smpl,
            'poses_smpl': poses_smpl,
            'trans_smpl': trans_smpl
        }

        return body_params
    
    def _init_res_dict(self):
        # out_keys = ['poses', 'betas', 'trans'] 
        # if self.export_meshes:
        #     out_keys += ['skel_v', 'skin_v', 'smpl_v']
        # res_dict = {key: [] for key in out_keys}

        # Create a dictionary with empty tensors to store the results of the optimization for each frame
        res_dict = {}
        res_dict['poses'] = torch.zeros((self.nb_frames, self.skel.num_q_params), device=self.device)
        res_dict['betas'] = torch.zeros((self.nb_frames, self.num_betas), device=self.device)
        res_dict['trans'] = torch.zeros((self.nb_frames, 3), device=self.device)
        if self.export_meshes:
            res_dict['skel_v'] = torch.zeros((self.nb_frames, self.skel.num_verts, 3), device=self.device)
            res_dict['skin_v'] = torch.zeros((self.nb_frames, self.skel.num_verts, 3), device=self.device)
            res_dict['smpl_v'] = torch.zeros((self.nb_frames, self.smpl.num_verts, 3), device=self.device)
            
        res_dict['gender'] = self.gender
        return res_dict

    
    def _fit_batch(self, body_params_in, i, i_start, i_end):
        """ Create parameters for the batch and run the optimization."""
        
        # Sample a batch 
        assert body_params_in['betas_smpl'].shape[0] == 1, f"beta_smpl should be of shape 1xF where F is the number of frames, got {body_params_in['betas_smpl'].shape}"
        body_params = { key: val[i_start:i_end] for key, val in body_params_in.items() if key!='betas_smpl'} # Ignore the betas_smpl since it is the same for all frames
        body_params['betas_smpl'] = body_params_in['betas_smpl'].clone()

        # SMPL params
        betas_smpl = body_params['betas_smpl']
        poses_smpl = body_params['poses_smpl']
        trans_smpl = body_params['trans_smpl']
        
        # SKEL params
        betas = to_params(body_params['betas_skel'], device=self.device)
        poses = to_params(body_params['poses_skel'], device=self.device)
        trans = to_params(body_params['trans_skel'], device=self.device)

        if 'verts' in body_params:
            verts = body_params['verts']
        else:
            # Run a SMPL forward pass to get the SMPL body vertices
            smpl_output = self.smpl(betas=betas_smpl, body_pose=poses_smpl[:,3:], transl=trans_smpl, global_orient=poses_smpl[:,:3])
            verts = smpl_output.vertices
                   
        # Optimize         
        config = self.cfg.optim_steps
        current_cfg = config[0]
        if not self.is_skel_data_init:
            # Optimize the global rotation and translation for the initial fitting
            print(f'Step 0: {current_cfg.description}')
            self._optim([trans], poses, betas, trans, verts, current_cfg)

        for ci, cfg in enumerate(config[1:]):
        # for ci, cfg in enumerate([config[-1]]): # To debug, only run the last step
            current_cfg.update(cfg)
            print(f'Step {ci+1}: {current_cfg.description}')
            self._optim([poses], poses, betas, trans, verts, current_cfg)
        
        # # Refine by optimizing the whole body
        # cfg.update(self.cfg_optim[])
        # cfg.update({'mode' : 'free', 'tolerance_change': 0.0001, 'l_joint': 0.2e4})
        # self._optim([trans, poses], poses, betas, trans, verts, cfg)
    
        return betas, poses, trans, verts
    
    def _optim(self,
            params, 
            poses,
            betas,
            trans,
            verts,
            cfg,
            ):
        
            # regress anatomical joints from SMPL's vertices 
            anat_joints = torch.einsum('bik,ji->bjk', [verts, self.skel.J_regressor_osim]) 
            dJ=torch.zeros((poses.shape[0], 24, 3), device=betas.device)
            
            # Create the optimizer
            optimizer = torch.optim.LBFGS(params, 
                                          lr=cfg.lr, 
                                          max_iter=cfg.max_iter, 
                                          line_search_fn=cfg.line_search_fn,  
                                          tolerance_change=cfg.tolerance_change)
                
            poses_init = poses.detach().clone()               
            trans_init = trans.detach().clone()

            def closure():
                optimizer.zero_grad()
                
                fi = self.watch_frame #frame of the batch to display
                output = self.skel.forward(poses=poses[fi:fi+1], 
                                            betas=betas[fi:fi+1], 
                                            trans=trans[fi:fi+1], 
                                            poses_type='skel', 
                                            dJ=dJ[fi:fi+1],
                                            skelmesh=True)
            
                self._fstep_plot(output, cfg, verts[fi:fi+1], anat_joints[fi:fi+1], )
                    
                loss_dict = self._fitting_loss(poses,
                                        poses_init,
                                        betas,
                                        trans,
                                        trans_init,
                                        dJ,
                                        anat_joints,
                                        verts,
                                        cfg)
                
                print(pretty_loss_print(loss_dict))
                             
                loss = sum(loss_dict.values())                     
                loss.backward()
            
                return loss

            for step_i in range(cfg.num_steps):
                loss = optimizer.step(closure).item()

    def _get_masks(self, cfg):
        pose_mask = torch.ones((self.skel.num_q_params)).to(self.device).unsqueeze(0)
        verts_mask = torch.ones_like(self.fitting_mask)
        joint_mask = torch.ones((self.skel.num_joints, 3)).to(self.device).unsqueeze(0).bool()
        
        # Mask vertices 
        if cfg.mode=='root_only':
            # Only optimize the global rotation of the body, i.e. the first 3 angles of the pose
            pose_mask[:] = 0 # Only optimize for the global rotation  
            pose_mask[:,:3] = 1
            # Only fit the thorax vertices to recover the proper body orientation and translation
            verts_mask = self.torso_verts_mask  
            
        elif cfg.mode=='fixed_upper_limbs':
            upper_limbs_joints = [0,1,2,3,6,9,12,15,16,17]
            verts_mask = (self.smpl.lbs_weights[:,upper_limbs_joints]>0.5).sum(dim=-1)>0
            verts_mask = verts_mask.unsqueeze(0).unsqueeze(-1)
            
            joint_mask[:, [3,4,5,8,9,10,18,23], :] = 0 # Do not try to match the joints past the elbow and knees
            
            pose_mask[:] = 1           
            pose_mask[:,:3] = 0    # Block the global rotation
            pose_mask[:,19] = 0  # block the lumbar twist
            # pose_mask[:, 36:39] = 0 
            # pose_mask[:, 43:46] = 0
            # pose_mask[:, 62:65] = 0
            # pose_mask[:, 62:65] = 0
            
        elif cfg.mode=='fixed_root': 
            pose_mask[:] = 1           
            pose_mask[:,:3] = 0  # Block the global rotation
            # pose_mask[:,19] = 0  # block the lumbar twist    
            
            # The orientation of the upper limbs is often wrong in SMPL so ignore these vertices for the finale step
            upper_limbs_joints = [1,2,16,17] 
            verts_mask = (self.smpl.lbs_weights[:,upper_limbs_joints]>0.5).sum(dim=-1)>0
            verts_mask = torch.logical_not(verts_mask)
            verts_mask = verts_mask.unsqueeze(0).unsqueeze(-1)
                         
        elif cfg.mode=='free': 
            verts_mask = torch.ones_like(self.fitting_mask )

            joint_mask[:]=0
            joint_mask[:, [19,14], :] = 1 # Only fir the scapula join to avoid collapsing shoulders
            
        else:
            raise ValueError(f'Unknown mode {cfg.mode}')
            
        return pose_mask, verts_mask, joint_mask
            
    def _fitting_loss(self,
                    poses,
                    poses_init,
                    betas,
                    trans,
                    trans_init,
                    dJ,
                    anat_joints,
                    verts,
                    cfg):
        
        loss_dict = {}
        
        
        pose_mask, verts_mask, joint_mask = self._get_masks(cfg) 
        poses = poses * pose_mask + poses_init * (1-pose_mask)

        # Mask joints to not optimize before computing the losses 
        
        output = self.skel.forward(poses=poses, betas=betas, trans=trans, poses_type='skel', dJ=dJ, skelmesh=False)
               
        # Fit the SMPL vertices
        # We know the skinning of the forearm and the neck are not perfect,
        # so we create a mask of the SMPL vertices that are important to fit, like the hands and the head
        loss_dict['verts_loss_loose'] = cfg.l_verts_loose * (verts_mask  * (output.skin_verts - verts)**2).sum() / (((verts_mask).sum()))

        # Fit the regressed joints, this avoids collapsing shoulders
        # loss_dict['joint_loss'] = cfg.l_joint * F.mse_loss(output.joints, anat_joints)
        loss_dict['joint_loss'] = cfg.l_joint * (joint_mask * (output.joints - anat_joints)**2).mean()
    
        # Time consistancy
        if poses.shape[0] > 1:
            # This avoids unstable hips orientationZ
            loss_dict['time_loss'] = cfg.l_time_loss * F.mse_loss(poses[1:], poses[:-1])
        
        loss_dict['pose_loss'] = cfg.l_pose_loss * compute_pose_loss(poses, poses_init)
        
        if cfg.use_basic_loss is False:
            # These losses can be used to regularize the optimization but are not always necessary
            loss_dict['anch_rot'] = cfg.l_anch_pose * compute_anchor_pose(poses, poses_init)
            loss_dict['anch_trans'] = cfg.l_anch_trans * compute_anchor_trans(trans, trans_init)
                
            loss_dict['verts_loss'] = cfg.l_verts * (verts_mask * self.fitting_mask * (output.skin_verts - verts)**2).sum() / (self.fitting_mask*verts_mask).sum()
                   
            # Regularize the pose
            loss_dict['scapula_loss'] = cfg.l_scapula_loss * compute_scapula_loss(poses)
            loss_dict['spine_loss'] = cfg.l_spine_loss * compute_spine_loss(poses)
            
            # Adjust the losses of all the pose regularizations sub losses with the pose_reg_factor value
            for key in ['scapula_loss', 'spine_loss', 'pose_loss']:
                loss_dict[key] = cfg.pose_reg_factor * loss_dict[key]
                
        return loss_dict

    def _fstep_plot(self, output, cfg, verts, anat_joints):
        "Function to plot each step"
        
        if('DISABLE_VIEWER' in os.environ):
            return
        
        pose_mask, verts_mask, joint_mask = self._get_masks(cfg) 
            
        skin_err_value = ((output.skin_verts[0] - verts[0])**2).sum(dim=-1).sqrt()
        skin_err_value = skin_err_value / 0.05
        skin_err_value = to_numpy(skin_err_value)
            
        skin_mesh = Mesh(v=to_numpy(output.skin_verts[0]), f=[], vc='white')
        skel_mesh = Mesh(v=to_numpy(output.skel_verts[0]), f=self.skel.skel_f.cpu().numpy(), vc='white')
        
        # Display vertex distance on SMPL
        smpl_verts = to_numpy(verts[0])
        smpl_mesh = Mesh(v=smpl_verts, f=self.smpl.faces)
        smpl_mesh.set_vertex_colors_from_weights(skin_err_value, scale_to_range_1=False)       
        
        smpl_mesh_masked = Mesh(v=smpl_verts[to_numpy(verts_mask[0,:,0])], f=[], vc='green')
        smpl_mesh_pc = Mesh(v=smpl_verts, f=[], vc='green')
        
        skin_mesh_err = Mesh(v=to_numpy(output.skin_verts[0]), f=self.skel.skin_f.cpu().numpy(), vc='white')
        skin_mesh_err.set_vertex_colors_from_weights(skin_err_value, scale_to_range_1=False) 
        # List the meshes to display
        meshes_left = [skin_mesh_err, smpl_mesh_pc, skel_mesh]
        meshes_right = [smpl_mesh_masked, skin_mesh, skel_mesh]

        if cfg.l_joint > 0:
            # Plot the joints
            meshes_right += location_to_spheres(to_numpy(output.joints[joint_mask[:,:,0]]), color=(1,0,0), radius=0.02)
            meshes_right += location_to_spheres(to_numpy(anat_joints[joint_mask[:,:,0]]), color=(0,1,0), radius=0.02) \
                

        self.mv[0][0].set_dynamic_meshes(meshes_left)
        self.mv[0][1].set_dynamic_meshes(meshes_right)

        # print(poses[frame_to_watch, :3])
        # print(trans[frame_to_watch])
        # print(betas[frame_to_watch, :3])
        # mv.get_keypress()