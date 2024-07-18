# -*- coding: utf-8 -*-
#
# Copyright (C) 2020 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# STAR: Sparse Trained  Articulated Human Body Regressor <https://arxiv.org/pdf/2008.08535.pdf>
#
#
# Code Developed by:
# Ahmed A. A. Osman, edited by Marilyn Keller

import scipy
import torch
import numpy as np

def build_homog_matrix(R, t=None):
    """ Create a homogeneous matrix from rotation matrix and translation vector
    @ R: rotation matrix of shape (B, Nj, 3, 3)
    @ t: translation vector of shape (B, Nj, 3, 1)
    returns: homogeneous matrix of shape (B, 4, 4)
    By Marilyn Keller
    """
    
    if t is None:
        B = R.shape[0]
        Nj = R.shape[1]
        t = torch.zeros(B, Nj, 3, 1).to(R.device)
    
    if R is None:
        B = t.shape[0]
        Nj = t.shape[1]
        R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, Nj, 1, 1).to(t.device)
    
    B = t.shape[0]
    Nj = t.shape[1]
        
    # import ipdb; ipdb.set_trace()
    assert R.shape == (B, Nj, 3, 3), f"R.shape: {R.shape}"
    assert t.shape == (B, Nj, 3, 1), f"t.shape: {t.shape}"
    
    G = torch.cat([R, t], dim=-1) # BxJx3x4 local transformation matrix
    pad_row = torch.FloatTensor([0, 0, 0, 1]).to(R.device).view(1, 1, 1, 4).expand(B, Nj, -1, -1) # BxJx1x4
    G = torch.cat([G, pad_row], dim=2) # BxJx4x4 padded to be 4x4 matrix an enable multiplication for the kinematic chain

    return G


def matmul_chain(rot_list):
    
    R_tot = rot_list[-1]
    for i in range(len(rot_list)-2,-1,-1):
        R_tot = torch.matmul(rot_list[i], R_tot)
    return R_tot


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    import torch.nn.functional as F
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector (B x Nj x 3)
    :param vec2: A 3d "destination" vector (B x Nj x 3)
    :return mat: A rotation matrix (B x Nj x 3 x 3) which when applied to vec1, aligns it with vec2.
    """
    for v_id, v in enumerate([vec1, vec2]):
        # vectors shape should be B x Nj x 3
        assert len(v.shape) == 3, f"Vectors {v_id} shape should be B x Nj x 3, got {v.shape}"
        assert v.shape[-1] == 3, f"Vectors {v_id} shape should be B x Nj x 3, got {v.shape}" 
    
    B = vec1.shape[0]
    Nj = vec1.shape[1]
    device = vec1.device
    
    a = vec1 / torch.linalg.norm(vec1, dim=-1, keepdim=True)
    b = vec2 / torch.linalg.norm(vec2,  dim=-1, keepdim=True)
    v = torch.cross(a, b, dim=-1)
    # Compute the dot product along the last dimension of a and b
    c = torch.sum(a * b, dim=-1)
    s = torch.linalg.norm(v, dim=-1) + torch.finfo(float).eps
    v0 = torch.zeros_like(v[...,0], device=device).unsqueeze(-1) 
    kmat_l1 = torch.cat([v0, -v[...,2].unsqueeze(-1), v[...,1].unsqueeze(-1)], dim=-1)
    kmat_l2 = torch.cat([v[...,2].unsqueeze(-1), v0, -v[...,0].unsqueeze(-1)], dim=-1)
    kmat_l3 = torch.cat([-v[...,1].unsqueeze(-1), v[...,0].unsqueeze(-1), v0], dim=-1)
    # Stack the matrix lines along a the -2 dimension
    kmat = torch.cat([kmat_l1.unsqueeze(-2), kmat_l2.unsqueeze(-2), kmat_l3.unsqueeze(-2)], dim=-2) # B x Nj x 3 x 3
    # import ipdb; ipdb.set_trace()
    rotation_matrix = torch.eye(3, device=device).view(1,1,3,3).expand(B, Nj, 3, 3) + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2)).view(B, Nj, 1, 1).expand(B, Nj, 3, 3)
    return rotation_matrix


def quat_feat(theta):
    '''
        Computes a normalized quaternion ([0,0,0,0]  when the body is in rest pose)
        given joint angles
    :param theta: A tensor of joints axis angles, batch size x number of joints x 3
    :return:
    '''
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_sin * normalized,v_cos-1], dim=1)
    return quat

def quat2mat(quat):
    '''
        Converts a quaternion to a rotation matrix
    :param quat:
    :return:
    '''
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def rodrigues(theta):
    '''
        Computes the rodrigues representation given joint angles

    :param theta: batch_size x number of joints x 3
    :return: batch_size x number of joints x 3 x 4
    '''
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat2mat(quat)


def with_zeros(input):
    '''
      Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor

    :param input: A tensor of dimensions batch size x 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    '''
    batch_size  = input.shape[0]
    row_append     = torch.FloatTensor(([0.0, 0.0, 0.0, 1.0])).to(input.device)
    row_append.requires_grad = False
    padded_tensor     = torch.cat([input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], 1)
    return padded_tensor


def with_zeros_44(input):
    '''
      Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor

    :param input: A tensor of dimensions batch size x 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    '''
    import ipdb; ipdb.set_trace()
    batch_size  = input.shape[0]
    col_append  = torch.FloatTensor(([[[[0.0, 0.0, 0.0]]]])).to(input.device)
    padded_tensor = torch.cat([input, col_append], dim=-1)
       
    row_append     = torch.FloatTensor(([0.0, 0.0, 0.0, 1.0])).to(input.device)
    row_append.requires_grad = False
    padded_tensor     = torch.cat([input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], 1)
    return padded_tensor
    

    
    return padded_tensor

def vector_to_rot():
    
    def rotation_matrix(A,B):
    # Aligns vector A to vector B

        ax = A[0]
        ay = A[1]
        az = A[2]

        bx = B[0]
        by = B[1]
        bz = B[2]

        au = A/(torch.sqrt(ax*ax + ay*ay + az*az))
        bu = B/(torch.sqrt(bx*bx + by*by + bz*bz))

        R=torch.tensor([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]], [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]], [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]] ])


        return(R)
    
    
    
def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def location_to_spheres(loc, color=(1,0,0), radius=0.02):
    """Given an array of 3D points, return a list of spheres located at those positions.

    Args:
        loc (numpy.array): Nx3 array giving 3D positions
        color (tuple, optional): One RGB float color vector to color the spheres. Defaults to (1,0,0).
        radius (float, optional): Radius of the spheres in meters. Defaults to 0.02.

    Returns:
        list: List of spheres Mesh
    """
    from psbody.mesh.sphere import Sphere
    import numpy as np
    cL = [Sphere(np.asarray([loc[i, 0], loc[i, 1], loc[i, 2]]), radius).to_mesh() for i in range(loc.shape[0])]
    for spL in cL:
        spL.set_vertex_colors(np.array(color)) 
    return cL

def sparce_coo_matrix2tensor(arr_coo, make_dense=False):
    assert isinstance(arr_coo, scipy.sparse._coo.coo_matrix), f"arr_coo should be a coo_matrix, got {type(arr_coo)}. Please download the updated SKEL pkl files from https://skel.is.tue.mpg.de/."

    values = arr_coo.data
    indices = np.vstack((arr_coo.row, arr_coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = arr_coo.shape

    tensor_arr = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    
    if make_dense:
        tensor_arr = tensor_arr.to_dense()
        
    return tensor_arr
        
    