import math

skel_joints_name= [
'pelvis', #0
'femur_r', #1
'tibia_r', #2
'talus_r', #3
'calcn_r', #4
'toes_r', #5
'femur_l', #6
'tibia_l', #7
'talus_l', #8
'calcn_l', #9
'toes_l', #10
'lumbar_body', #11
'thorax', #12
'head', #13
'scapula_r', #14
'humerus_r', #15
'ulna_r', #16
'radius_r', #17
'hand_r', #18
'scapula_l', #19
'humerus_l', #20
'ulna_l', #21
'radius_l', #22
'hand_l'] #23

pose_param_names = [
 'pelvis_tilt', #0 
 'pelvis_list', #1 
 'pelvis_rotation', #2 
 'hip_flexion_r', #3 
 'hip_adduction_r', #4 
 'hip_rotation_r', #5 
 'knee_angle_r', #6 
 'ankle_angle_r', #7 
 'subtalar_angle_r', #8 
 'mtp_angle_r', #9 
 'hip_flexion_l', #10
 'hip_adduction_l', #11
 'hip_rotation_l', #12
 'knee_angle_l', #13
 'ankle_angle_l', #14
 'subtalar_angle_l', #15
 'mtp_angle_l', #16
 'lumbar_bending', #17
 'lumbar_extension', #18
 'lumbar_twist', #19
 'thorax_bending', #20
 'thorax_extension', #21
 'thorax_twist', #22
 'head_bending', #23
 'head_extension', #24
 'head_twist', #25
 'scapula_abduction_r', #26
 'scapula_elevation_r', #27
 'scapula_upward_rot_r', #28
 'shoulder_r_x', #29
 'shoulder_r_y', #30
 'shoulder_r_z', #31
 'elbow_flexion_r', #32
 'pro_sup_r', #33
 'wrist_flexion_r', #34
 'wrist_deviation_r', #35
 'scapula_abduction_l', #36
 'scapula_elevation_l', #37
 'scapula_upward_rot_l', #38
 'shoulder_l_x', #39
 'shoulder_l_y', #40
 'shoulder_l_z', #41
 'elbow_flexion_l', #42
 'pro_sup_l', #43
 'wrist_flexion_l', #44
 'wrist_deviation_l', #45
]

pose_limits = {
'scapula_abduction_r' :  [-0.628, 0.628],
'scapula_elevation_r' :  [-0.4, -0.1],
'scapula_upward_rot_r' : [-0.190, 0.319],

'scapula_abduction_l' :  [-0.628, 0.628],
'scapula_elevation_l' :  [-0.1, -0.4],
'scapula_upward_rot_l' : [-0.210, 0.219],  

'elbow_flexion_r' : [0, (3/4)*math.pi],
'pro_sup_r'       : [-3/4*math.pi/2, 3/4*math.pi/2],
'wrist_flexion_r' : [-math.pi/2, math.pi/2],
'wrist_deviation_r' :[-math.pi/4, math.pi/4],

'elbow_flexion_l' : [0, (3/4)*math.pi],
'pro_sup_l'       : [-math.pi/2, math.pi/2],
'wrist_flexion_l' : [-math.pi/2, math.pi/2],
'wrist_deviation_l' :[-math.pi/4, math.pi/4],

'shoulder_r_y' : [-math.pi/2, math.pi/2], 

'lumbar_bending' : [-2/3*math.pi/4, 2/3*math.pi/4],
'lumbar_extension' : [-math.pi/4, math.pi/4],
'lumbar_twist' :  [-math.pi/4, math.pi/4],   

'thorax_bending' :[-math.pi/4, math.pi/4], 
'thorax_extension' :[-math.pi/4, math.pi/4], 
'thorax_twist' :[-math.pi/4, math.pi/4],

'head_bending' :[-math.pi/4, math.pi/4], 
'head_extension' :[-math.pi/4, math.pi/4], 
'head_twist' :[-math.pi/4, math.pi/4], 

'ankle_angle_r' : [-math.pi/4, math.pi/4],
'subtalar_angle_r' : [-math.pi/4, math.pi/4],
'mtp_angle_r' : [-math.pi/4, math.pi/4],

'ankle_angle_l' : [-math.pi/4, math.pi/4],
'subtalar_angle_l' : [-math.pi/4, math.pi/4],
'mtp_angle_l' : [-math.pi/4, math.pi/4],

'knee_angle_r' : [0, 3/4*math.pi],
'knee_angle_l' : [0, 3/4*math.pi],

}

# For each joint of SKEL, we define one corresponding joint of SMPL
# Note that there is not a 1 to 1 correspondance, especially for the arm supination and ankle.
# This correspondance is used to leverage the pose dependant blend shapes from SMPL
smpl_joint_corresp = [
0,# pelvis
2,# femur_r (The right femur is joint 2 in SMPL)
5,# tibia_r
8,# talus_r
8,# calcn_r
11,# toes_r
1,# femur_l
4,# tibia_l
7,# talus_l
7,# calcn_l
10,# toes_l
3,# lumbar_body
6,# thorax
15,# head
14,# scapula_r
17,# humerus_r
19,# ulna_r
0,# radius_r # We set it to 0 to ignore it, as Rsmpl[0] is not used to compute the pose dependant blend shapes
21,# hand_r
13,# scapula_l
16,# humerus_l
18,# ulna_l
0,# radius_l
20,# hand_l
]



# Bones scaling
""" 
Most bone meshes are scaled given the limb lengths
For some bones, they need to be scaled wrt the skin. 
Given a bone index, for each dimention we give vertex indices to use for scaling
The template bones fit in the template SMPL mesh. So we compare the
vertices distance for the shaped mesh with the vertices distances of the template
The dimensions represent the different axis:
Note that he bones are scaled in the unposed bone space, which is different than SMPL posed space
0 : front back
1 : up down
2 : left right

(joint index, scaling dimention in bone space, scaling dimention in SMPL space)
"""

scaling_keypoints ={
    # Head
    (13,0,2): (
                410, # between eyes
                384 # back of the head
            ),
    (13, 1, 1): (
                414, # Top of the head
                384 # Clavicula hole
            ),
    (13, 2, 0) : (
                196, # one side head
                3708 # other side
            ), 
    # Right hand
    (18, 0, 1): (# hand width
                6179, #r end pinkie
                6137 #r_middle finger end
            ),
    (18, 1, 0): ( # hand length
                5670, #r_wrist_middle,
                5906 #r_middle finger end
            ),
    # Left hand
    (23, 0, 1): (# hand width (Use the same vertex references as right hand)
                6179, #r end pinkie
                6137 #r_middle finger end
            ),
    (23, 1, 0): ( # hand length
                5670, #r_wrist_middle,
                5906 #r_middle finger end
            ),
    }