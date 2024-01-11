import torch
from skel.joints_def import curve_torch_3d
from skel.utils import axis_angle_to_matrix, euler_angles_to_matrix, rodrigues

class OsimJoint(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def q_to_translation(self, q, **kwargs):       
        return torch.zeros(q.shape[0], 3).to(q.device)


class CustomJoint(OsimJoint):
    
    def __init__(self, axis, axis_flip) -> None:
        super().__init__()
        self.register_buffer('axis', torch.FloatTensor(axis))
        self.register_buffer('axis_flip', torch.FloatTensor(axis_flip))
        self.register_buffer('nb_dof', torch.tensor(len(axis)))  
        
    def q_to_rot(self, q, **kwargs):
        
        ident = torch.eye(3, dtype=q.dtype).to(q.device)
        
        Rp = ident.unsqueeze(0).expand(q.shape[0],3,3) # torch.eye(q.shape[0], 3, 3)
        for i in range(self.nb_dof):
            axis = self.axis[i]
            angle_axis = q[:, i:i+1] * self.axis_flip[i] * axis
            Rp_i = axis_angle_to_matrix(angle_axis)  
            Rp = torch.matmul(Rp_i, Rp)
        return Rp 
 

       
class CustomJoint1D(OsimJoint):
    
    def __init__(self, axis, axis_flip) -> None:
        super().__init__()
        self.axis = torch.FloatTensor(axis)
        self.axis = self.axis / torch.linalg.norm(self.axis)
        self.axis_flip = torch.FloatTensor(axis_flip)
        self.nb_dof = 1
        
    def q_to_rot(self, q, **kwargs):
        axis = self.axis
        angle_axis = q[:, 0:1] * self.axis_flip * axis
        Rp_i = axis_angle_to_matrix(angle_axis) 
        return Rp_i    

    
class WalkerKnee(OsimJoint):
    
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer('nb_dof', torch.tensor(1))
        # self.nb_dof = 1
        
    def q_to_rot(self, q, **kwargs):
        # Todo : for now implement a basic knee
        theta_i = torch.zeros(q.shape[0], 3).to(q.device)
        theta_i[:, 2] = -q[:, 0]
        Rp_i = axis_angle_to_matrix(theta_i) 
        return Rp_i
        
class PinJoint(OsimJoint):
    
    def __init__(self, parent_frame_ori) -> None:
        super().__init__()
        self.register_buffer('parent_frame_ori', torch.FloatTensor(parent_frame_ori))
        self.register_buffer('nb_dof', torch.tensor(1))
        
     
    def q_to_rot(self, q, **kwargs):
        
        talus_orient_torch = self.parent_frame_ori 
        Ra_i = euler_angles_to_matrix(talus_orient_torch, 'XYZ')
        
        z_axis = torch.FloatTensor([0,0,1]).to(q.device)
        axis = torch.matmul(Ra_i, z_axis).to(q.device)
        
        axis_angle = q[:, 0:1] * axis
        Rp_i = axis_angle_to_matrix(axis_angle) 
                
        return Rp_i
    
    
class ConstantCurvatureJoint(CustomJoint):
    
    def __init__(self, **kwargs ) -> None:
        super().__init__( **kwargs)
        

        
class EllipsoidJoint(CustomJoint):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
