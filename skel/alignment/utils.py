
import os
import pickle
import torch
import numpy as np
from psbody.mesh.sphere import Sphere

# to_params = lambda x: torch.from_numpy(x).float().to(self.device).requires_grad_(True)
# to_torch = lambda x: torch.from_numpy(x).float().to(self.device)

def to_params(x, device):
    return x.to(device).requires_grad_(True)

def to_torch(x, device):
    return torch.from_numpy(x).float().to(device)

def to_numpy(x):
    return x.detach().cpu().numpy()

def load_smpl_seq(smpl_seq_path, gender=None, straighten_hands=False):

    if not os.path.exists(smpl_seq_path):
        raise Exception('Path does not exist: {}'.format(smpl_seq_path))
    
    if smpl_seq_path.endswith('.pkl'):
        data_dict = pickle.load(open(smpl_seq_path, 'rb'))
    
    elif smpl_seq_path.endswith('.npz'):
        data_dict = np.load(smpl_seq_path, allow_pickle=True)
        
        if data_dict.files == ['pred_smpl_parms', 'verts', 'pred_cam_t']:
            data_dict = data_dict['pred_smpl_parms'].item()# ['global_orient', 'body_pose', 'body_pose_axis_angle', 'global_orient_axis_angle', 'betas']
        else:
            data_dict = {key: data_dict[key] for key in data_dict.keys()} # convert to python dict
    else:
        raise Exception('Unknown file format: {}. Supported formats are .pkl and .npz'.format(smpl_seq_path))
        
    # Instanciate a dictionary with the keys expected by the fitter
    data_fixed = {}  
    
    # Get gender     
    if 'gender' not in data_dict:
        assert gender is not None, f"The provided SMPL data dictionary does not contain gender, you need to pass it in command line"
        data_fixed['gender'] = gender
    elif not isinstance(data_dict['gender'], str):
            # In some npz, the gender type happens to be: array('male', dtype='<U4'). So we convert it to string
            data_fixed['gender'] = str(data_dict['gender'])
    else:
        data_fixed['gender'] = gender
            
    # convert tensors to numpy arrays 
    for key, val in data_dict.items():
        if isinstance(val, torch.Tensor):
            data_dict[key] = val.detach().cpu().numpy()

    # Get the SMPL pose
    if 'poses' in data_dict: 
        poses = data_dict['poses']
    elif 'body_pose_axis_angle' in data_dict and 'global_orient_axis_angle' in data_dict:
        # assert 'global_orient' in data_dict and 'body_pose' in data_dict, f"Could not find poses in {smpl_seq_path}. Available keys: {data_dict.keys()})"
        poses = np.concatenate([data_dict['global_orient_axis_angle'], data_dict['body_pose_axis_angle']], axis=1)
        poses = poses.reshape(-1, 72)
    elif 'body_pose' in data_dict and 'global_orient' in data_dict:
        poses = np.concatenate([data_dict['global_orient_axis_angle'], data_dict['body_pose_axis_angle']], axis=-1)
    else: 
        raise Exception(f"Could not find poses in {smpl_seq_path}. Available keys: {data_dict.keys()})")
        
    if poses.shape[1] == 156:
        # Those are SMPL+H poses, we remove the hand poses to keep only the body poses
        smpl_poses = np.zeros((poses.shape[0], 72))
        smpl_poses[:, :72-2*3] = poses[:, :72-2*3] # We leave params for SMPL joints 22 and 23 to zero as these DOF are not present in SMPLH
        poses = smpl_poses
    
    # Set SMPL joints 22 and 23 to zero as SKEL has rigid hands 
    if straighten_hands:      
        poses[:, 72-2*3:] = 0
        
    data_fixed['poses'] = poses
        
    # Translation
    if 'trans' not in data_dict:
        data_fixed['trans'] = np.zeros((poses.shape[0], 3))
    else:
        data_fixed['trans'] = data_dict['trans']
        
    # Get betas 
    betas = data_dict['betas'][..., :10] # Keep only the 10 first betas
    if len(betas.shape) == 1 and len(poses.shape) == 2:
        betas = betas[None, :] # Add a batch dimension
    data_fixed['betas'] = betas
     
    for key in ['trans', 'poses', 'betas', 'gender']:
        assert key in data_fixed.keys(), f'Could not find {key} in {smpl_seq_path}. Available keys: {data_fixed.keys()})'
        
    out_dict = {}
    out_dict['trans'] = data_fixed['trans']
    out_dict['poses'] = data_fixed['poses']
    out_dict['betas'] = data_fixed['betas']
    out_dict['gender'] = data_fixed['gender']
    
    return out_dict
        
        
def location_to_spheres(loc, color=(1,0,0), radius=0.02):
    """Given an array of 3D points, return a list of spheres located at those positions.

    Args:
        loc (numpy.array): Nx3 array giving 3D positions
        color (tuple, optional): One RGB float color vector to color the spheres. Defaults to (1,0,0).
        radius (float, optional): Radius of the spheres in meters. Defaults to 0.02.

    Returns:
        list: List of spheres Mesh
    """

    cL = [Sphere(np.asarray([loc[i, 0], loc[i, 1], loc[i, 2]]), radius).to_mesh() for i in range(loc.shape[0])]
    for spL in cL:
        spL.set_vertex_colors(np.array(color)) 
    return cL
