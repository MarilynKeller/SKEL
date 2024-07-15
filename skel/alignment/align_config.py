""" Optimization config file. In 'optim_steps' we define each optimization steps. 
Each step inherit and overwrite the parameters of the previous step."""

config = {
    'keepalive_meshviewer': False,
    'optim_steps':
        [
            { 
            'description' : 'Adjust the root orientation and translation',      
            'use_basic_loss': True,
            'lr': 1,
            'max_iter': 20,
            'num_steps': 10,
            'line_search_fn': 'strong_wolfe', #'strong_wolfe',
            'tolerance_change': 1e-7,# 1e-4, #0.01
            'mode' : 'root_only', 

            'l_verts_loose': 300,         
            'l_time_loss': 0,#5e2,      
            
            'l_joint': 0.0,
            'l_verts': 0,
            'l_scapula_loss': 0.0,
            'l_spine_loss': 0.0,
            'l_pose_loss': 0.0,
            

            },
            # Adjust the upper limbs
            {
            'description' : 'Adjust the upper limbs pose',
            'lr': 0.1,
            'max_iter': 20,
            'num_steps': 10,
            'tolerance_change': 1e-7, 
            'mode' : 'fixed_upper_limbs', #'fixed_root', 
            
            'l_verts_loose': 600,
            'l_joint': 1e3,   
            'l_time_loss': 0,# 5e2,            
            'l_pose_loss': 1e-4,
            },
            # Adjust the whole body
            {
            'description' : 'Adjust the whole body pose with fixed root',
            'lr': 0.1,
            'max_iter': 20,
            'num_steps': 10,
            'tolerance_change': 1e-7, 
            'mode' : 'fixed_root', #'fixed_root', 
            
            'l_verts_loose': 600,
            'l_joint': 1e3,   
            'l_time_loss': 0,            
            'l_pose_loss': 1e-4,
            },
            #
            {
            'description' : 'Free optimization',
            'lr': 0.1,
            'max_iter': 20,
            'num_steps': 10,
            'tolerance_change': 1e-7, 
            'mode' : 'free', #'fixed_root', 
            
            'l_verts_loose': 600,
            'l_joint': 1e3,   
            'l_time_loss':0,            
            'l_pose_loss': 1e-4,
            },  
    ]
}