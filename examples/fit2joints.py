
"""
This script is a base to optimize the pose and body shape of the SKEL model to match a 3D skeleton.
"""

import argparse
import math
import sys
from skel.alignment.aligner import compute_pose_loss, compute_scapula_loss
from skel.utils import location_to_spheres
import torch
from skel.skel_model import SKEL
from tqdm import trange
from psbody.mesh import Mesh, MeshViewer

def optim(params, 
          poses,
          betas,
          trans,
          target_joints,
          target_joints_mask,
          skel_model,
          device,
          lr=1e-1,
          max_iter=10,
          num_steps=5,
          line_search_fn='strong_wolfe',
          rot_only=False,
          watch_frame=0,
          pose_reg_factor = 1e1,
          ):
    
    
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter, line_search_fn=line_search_fn)
        pbar = trange(num_steps, leave=False)
        mv = MeshViewer(keepalive=True)
        

        def closure():
            optimizer.zero_grad()
            
            # Visualization
            fi = watch_frame #frame of the batch to display
            output = skel_model.forward(poses=poses[fi:fi+1], 
                                        betas=betas[fi:fi+1], 
                                        trans=trans[fi:fi+1], 
                                        poses_type='skel', 
                                        skelmesh=True)
            meshes_to_display = [Mesh(v=output.skin_verts[fi].detach().cpu().numpy(), f=[], vc='white')] \
                    + [Mesh(v=output.skin_verts[fi].detach().cpu().numpy(), f=[], vc='white')] \
                    + [Mesh(v=output.skel_verts[fi].detach().cpu().numpy(), f=[], vc='white')] \
                    + location_to_spheres(output.joints.detach().cpu().numpy()[fi], color=(1,0,0), radius=0.02) \
                    + location_to_spheres(target_joints.detach().cpu().numpy()[fi], color=(0,1,0), radius=0.02) # The joints we want to match are in green
                    # + [Mesh(v=output.skel_verts[fi].detach().cpu().numpy(), f=skel_model.skel_f.cpu().numpy(), vc='white')] \
            mv.set_dynamic_meshes(meshes_to_display)
            import ipdb; ipdb.set_trace()

          
            # Only optimize the global rotation of the SKEL model
            if rot_only:
                mask = torch.zeros_like(poses).to(device)
                mask[:,:3] = 1
                poses_in = poses * mask
            else:
                poses_in = poses
            
            output = skel_model.forward(poses=poses_in, betas=betas, trans=trans, poses_type='skel', skelmesh=False)

            # Loss to fit SKEL to the target joints
            joint_loss = ((output.joints - target_joints)*target_joints_mask).pow(2).mean()
            
            # Regularize the pose so that it stays close to the initial T pose
            pose_loss = 1e-4*compute_pose_loss(poses)    
            
            scapula_loss = 1e-2 * compute_scapula_loss(poses_in)     
        
            loss = joint_loss + pose_loss + scapula_loss
            
            # make a pretty print of the losses
            print(f"Joint loss: {joint_loss.item():.4f}, \
                  Pose loss: {pose_loss.item():.4f}, \
                  Scapula loss: {scapula_loss.item():.4f}")
      
            loss.backward()
        
            return loss

        for _ in pbar:
            loss = optimizer.step(closure).item()
            with torch.no_grad():
                poses[:] = torch.atan2(poses.sin(), poses.cos())
            pbar.set_postfix_str(f"Loss {loss:.4f}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Fit the SKEL model to a 3D skeleton')
    
    parser.add_argument('--print_joints', action='store_true', help='Print the joints location of the SKEL mean body T pose')
    
    args = parser.parse_args()

    """ Fill this array that gives the 3D location of each joint for your skeleton model. 
    For the joints that do not have an equivalent in the SKEL model, you can set the location to [0,0,0]
    """
    skel_joints_locations= [
        [0.0019138669595122337, -0.19136004149913788, 0.08603459596633911], # pelvis #0
        [-0.07333344221115112, -0.25300851464271545, 0.032850027084350586], # femur_r #1
        [-0.08588700741529465, -0.6144093871116638, 0.019581301137804985], # tibia_r #2
        [-0.08708114176988602, -0.9585244655609131, -0.033686064183712006], # talus_r #3
        [-0.10305365175008774, -0.9954496026039124, -0.06979851424694061], # calcn_r #4
        [-0.08786104619503021, -1.0032943487167358, 0.09505409002304077], # toes_r #5
        [0.07497797161340714, -0.2514311671257019, 0.026695936918258667], # femur_l #6
        [0.08192945271730423, -0.6116349697113037, 0.01632699742913246], # tibia_l #7
        [0.08593304455280304, -0.9565253257751465, -0.036195699125528336], # talus_l #8
        [0.09940887987613678, -0.9929266571998596, -0.07360824942588806], # calcn_l #9
        [0.08981682360172272, -1.0037813186645508, 0.09251146018505096], # toes_l #10
        [-0.0040580411441624165, -0.05980953574180603, 0.00521281361579895], # lumbar_body #11
        [0,0,0], # thorax #12
        [1.3667973689734936e-05, 0.32451707124710083, -0.011099399998784065], # head #13
        [-0.13359515368938446, 0.2278982400894165, -0.011191673576831818], # scapula_r #14
        [-0.1392209231853485, 0.1988346427679062, -0.00913430005311966], # humerus_r #15
        [-0.1294557750225067, -0.039256319403648376, 0.01879701390862465], # ulna_r #16
        [-0.14408335089683533, -0.05108558014035225, 0.019224876537919044], # radius_r #17
        [-0.1466834843158722, -0.2522241175174713, 0.068956658244133], # hand_r #18
        [0.13161833584308624, 0.22813020646572113, -0.004042687825858593], # scapula_l #19
        [0.14141127467155457, 0.19870273768901825, -0.005015652161091566], # humerus_l #20
        [0.13410256803035736, -0.035691216588020325, 0.025579599663615227], # ulna_l #21
        [0.15036527812480927, -0.04728853330016136, 0.02740827016532421], # radius_l #22
        [0.15129032731056213, -0.25007086992263794, 0.08041109889745712]] #'hand_l #23
        
    device = 'cpu'

    # Instanciate the SKEL model
    skel_model = SKEL(gender='female').to(device)

    # Initialize the pose, shape and translation parameters to default values (T pose)
    pose = torch.zeros(1, skel_model.num_q_params).to(device) # (1, 46)
    betas = torch.zeros(1, skel_model.num_betas).to(device) # (1, 10)
    trans = torch.zeros(1, 3).to(device)

    if args.print_joints:
        # Print the location of the SKEL mean body T pose joints
        skel_output = skel_model(pose, betas, trans)
        skeleton_joints_locations = skel_output['joints'] # (1, 24, 3)
        print('\nSKEL mean body T pose joints locations:')
        print(skeleton_joints_locations[0].tolist())
        sys.exit()

    # This bit is for generating the upper example joints locations
    if False:
        # Generate joints location for someone short whith the arms along the body
        betas[0,0] = -2
        pose[0, 29] = math.pi/2
        pose[0, 39] = -math.pi/2
        skel_output = skel_model(pose, betas, trans)
        skeleton_joints_locations = skel_output['joints'] # (1, 24, 3)
        print(skeleton_joints_locations[0].tolist())

        # Reset the default T pose, comment out if needed
        betas[0,0] = 0
        pose[0, 29] = 0
        pose[0, 39] = 0

    # Convert the 3D location of the joints to tensors and expand them to have a batch size of 1
    target_joints = torch.tensor(skel_joints_locations).float().to(device).expand(1, -1, -1) # (1, 24, 3)
    target_joints_mask = torch.tensor([0 if loc==[0,0,0] else 1 for loc in skel_joints_locations]).float().to(device).reshape(1, -1, 1) # (1, 24, 1)

    # import ipdb; ipdb.set_trace()

    pose.requires_grad = True
    betas.requires_grad = True
    trans.requires_grad = True

    # Optimize the translation pose and body shape to match the 3D skeleton
    optim([trans,pose,betas], 
        pose, 
        betas, 
        trans, 
        target_joints=target_joints,
        target_joints_mask=target_joints_mask,
        skel_model = skel_model, 
        max_iter=50,
        num_steps=5,
        device = device, 
        rot_only=False)

    optimized_skel_output = skel_model(pose, betas, trans)
    optimized_skin_mesh = Mesh(v=optimized_skel_output.skin_verts.detach().cpu().numpy(), f=skel_model.skin_f.cpu().numpy())
    optimized_skin_mesh.show()

    # Uncomment to save the mesh as a .ply file
    # optimized skin_mesh.write_ply('optimized_skel.ply')