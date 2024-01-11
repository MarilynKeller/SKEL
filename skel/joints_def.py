import torch
import numpy as np

def right_scapula(angle_abduction, angle_elevation, angle_rot, thorax_width, thorax_height):
    radius_x = thorax_width / 4 * torch.cos(angle_elevation-np.pi/4)
    radius_y = thorax_width / 4
    radius_z = thorax_height / 2
    t = torch.stack([-radius_x * torch.cos(angle_abduction), 
                        -radius_z*torch.sin(angle_elevation-np.pi/4) ,  # Todo revert sin and cos here
                        radius_y * torch.sin(angle_abduction)
                    ], dim=1)
    
    # h = thorax_height
    # w = thorax_width
    # d = thorax_width/2 # approximation
    
    # theta1 = angle_abduction
    # theta2 = angle_elevation
    
    # tx = h*torch.sin(theta2) + h
    # ty = 0*tx
    # tz = 0*tx
    
    # t = torch.stack([tx, 
    #                  ty,  # Todo revert sin and cos here
    #                  tz
    #                 ], dim=1)
    return t


def left_scapula(angle_abduction, angle_elevation, angle_rot, thorax_width, thorax_height):
    angle_abduction = -angle_abduction
    angle_elevation = -angle_elevation
    radius_x = thorax_width / 4 * torch.cos(angle_elevation-np.pi/4)
    radius_y = thorax_width / 4
    radius_z = thorax_height / 2
    t = torch.stack([radius_x * torch.cos(angle_abduction), 
                        -radius_z*torch.sin(angle_elevation-np.pi/4) , 
                        radius_y * torch.sin(angle_abduction)
                    ], dim=1)
    return t


def curve_torch_1d(angle, t, l):
    """Trace a curve with constan curvature of arc length l and curvature k using plt
    :param angle: angle of the curve in radians B
    :param t: parameter of the curve, float in [0,1] of shape B
    :param l: arc length of the curve, float of shape B
    """
    
    assert angle.shape == t.shape == l.shape, f"Shapes of angle, t and l must be the same, got {angle.shape}, {t.shape}, {l.shape}"
    
    # import ipdb; ipdb.set_trace()
    x = torch.zeros_like(angle)
    y = torch.zeros_like(angle)
    mask_small = torch.abs(angle) < 1e-5
    
    # Process small angles separately
    # We use taylor development for small angles to avoid explosion due to number precision
    tm = t[mask_small]
    anglem = angle[mask_small]
    lm = l[mask_small]
    # import ipdb; ipdb.set_trace()

    x[mask_small] = lm * tm*tm* anglem/2
    y[mask_small] = lm * tm * (1 - tm*tm*tm/6 * anglem*anglem) 
    
    # Process non small angles
    mask_big = torch.logical_not(mask_small)
    tm = t[mask_big]
    anglem = angle[mask_big]
    lm = l[mask_big]
    
    # print(x,y)  
    # return x,y
    c_arc = lm
    r = c_arc / (anglem)
    x[mask_big] = r*(1 - torch.cos(tm*anglem))
    y[mask_big] = r*torch.sin(tm*anglem)   
    return x,y

def curve_torch_3d(angle_x, angle_y, t, l):
    x,y = curve_torch_1d(angle_x, t, l)
    tx = torch.cat([-x.unsqueeze(-1), 
                    y.unsqueeze(-1), 
                    torch.zeros_like(x).unsqueeze(-1)],
                   dim=1).unsqueeze(-1) # Extention
    # import ipdb; ipdb.set_trace()
    x,y = curve_torch_1d(angle_y, t, l)
    ty = torch.cat([torch.zeros_like(x).unsqueeze(-1), 
                    y.unsqueeze(-1), 
                    -x.unsqueeze(-1)],
                   dim=1).unsqueeze(-1) # Bending
    return tx+ty
