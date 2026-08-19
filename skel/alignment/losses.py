import torch

def compute_scapula_loss(poses):
    
    scapula_indices = [26, 27, 28, 36, 37, 38]
    
    scapula_poses = poses[:, scapula_indices]
    scapula_loss = torch.linalg.norm(scapula_poses, ord=2)
    return scapula_loss

def compute_spine_loss(poses):
    
    spine_indices = range(17, 25)
    
    spine_poses = poses[:, spine_indices]
    spine_loss = torch.linalg.norm(spine_poses, ord=2)
    return spine_loss

def compute_pose_loss(poses, pose_init):
    
    pose_loss = torch.linalg.norm(poses[:, 3:], ord=2) # The global rotation should not be constrained
    return pose_loss

def compute_anchor_pose(poses, pose_init):
    
    pose_loss = torch.nn.functional.mse_loss(poses[:, :3], pose_init[:, :3])
    return pose_loss 

def compute_anchor_trans(trans, trans_init):

    trans_loss = torch.nn.functional.mse_loss(trans, trans_init)
    return trans_loss 

def compute_time_loss(poses):
    
    with torch.no_grad():
        pose_range = torch.max(poses, dim=0).values - torch.min(poses, dim=0).values
    pose_delta = (poses[1:] - poses[:-1])/pose_range
    time_loss = torch.linalg.norm(pose_delta, ord=2)
    return time_loss

def gaussian_kernel_1d(sigma, device, dtype):
    """Normalized 1D Gaussian kernel (radius ~2.5 sigma) for temporal smoothing."""

    radius = max(1, int(round(2.5 * float(sigma))))
    taps = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (taps / float(sigma)) ** 2)
    return kernel / kernel.sum()

def smooth_time(x, kernel):
    """Smooth a (T, ...) tensor along the time axis with a 1D kernel, replicate padding."""

    radius = (kernel.shape[0] - 1) // 2
    if x.shape[0] <= 1 or radius == 0:
        return x
    padded = torch.cat([x[:1].expand(radius, *x.shape[1:]),
                        x,
                        x[-1:].expand(radius, *x.shape[1:])], dim=0)
    out = torch.zeros_like(x)
    for i in range(kernel.shape[0]):
        out = out + kernel[i] * padded[i : i + x.shape[0]]
    return out

def compute_velocity_matching_loss(verts_fitted, target_vel_smoothed, kernel):
    """Match the fitted skin vertices' frame-to-frame velocity to the target vertices' velocity.

    The time loss is a zero-velocity prior on the poses: it damps jitter but also damps
    genuine motion. This term instead matches the *observed* velocity of the target mesh,
    so it is equivalent to a zero-velocity prior on a still subject and unbiased on a
    moving one.

    Both sides are smoothed with the SAME temporal Gaussian kernel. The target velocity
    must be smoothed because per-frame estimation flutter should not be matched; smoothing
    the fitted velocity with the same kernel makes the blur cancel between the two sides,
    so motion faster than the kernel is arbitrated by the data term instead of being
    suppressed. (Smoothing the target only penalizes genuinely fast motion.)

    verts_fitted: (T, V, 3) fitted skin vertices for the batch.
    target_vel_smoothed: (T-1, V, 3) pre-smoothed target vertex velocity, see smooth_time.
    kernel: the 1D Gaussian kernel the target velocity was smoothed with.
    """

    vel_fitted = smooth_time(verts_fitted[1:] - verts_fitted[:-1], kernel)
    return torch.nn.functional.mse_loss(vel_fitted, target_vel_smoothed)

def pretty_loss_print(loss_dict):
    # Pretty print the loss on the form loss val | loss1 val1 | loss2 val2 
    # Start with the total loss
    loss = sum(loss_dict.values())
    pretty_loss = f'{loss:.4f}'
    for key, val in loss_dict.items():
        pretty_loss += f' | {key} {val:.4f}'
    return pretty_loss   
