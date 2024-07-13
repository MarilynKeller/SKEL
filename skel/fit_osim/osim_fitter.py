# Small demo to show how to extract joints and markers trajectory from an OpenSim .osim and .mot file
# in prevision of fitting a SKEL model to the sequence

import yaml
from skel.fit_osim.osim_aug import OSIM


if __name__ == '__main__':
    
    osim_path ='/is/cluster/work/mkeller2/Data/TML/OpenSim/FullBodyModelwSampleSim.-latest/ModelWithSampleSimulations-4.0/SimulationDataAndSetupFiles-4.0/Rajagopal2015.osim'

    mot_path = '/is/cluster/work/mkeller2/Data/TML/OpenSim/FullBodyModelwSampleSim.-latest/ModelWithSampleSimulations-4.0/SimulationDataAndSetupFiles-4.0/IK/results_run/ik_output_run.mot'
    
    joint_mapping_dict_path = 'skel/fit_osim/mappings/full_body.yaml'
    
    osim = OSIM(osim_path, mot_path)
    
    osim_markers, osim_joints, osim_joints_ori = osim.run_fk()
    
    #Print the BSM model joints location
    print(f'Osim joints: {osim_joints}')
    
    print(f'joints shape: {osim_joints.shape}')
    print(f'joints_ori shape: {osim_joints_ori.shape}')
    print(f'markers shape: {osim_markers.shape}')
    print(f'markers_labels: {osim.markers_labels}')
    print(f'joints_labels: {osim.joints_labels}')
    
    # Joint mapping between SKEL and OpenSim
    skel_mapping = yaml.load(open(joint_mapping_dict_path, 'r'), Loader=yaml.FullLoader)
    import ipdb; ipdb.set_trace()
    skel_joints_mask = osim.skel_joint_mask(skel_mapping)
    print(f'mask: {skel_joints_mask}')
    print(f'mask shape: {len(skel_joints_mask)}')
    
    # Markers mapping between SKEL and OpenSim
    osim_mk_mask, skel_verts_mask = osim.skel_marker_mask(skel_mapping)
    for osim_marker_index, skel_verts in zip(osim_mk_mask, skel_verts_mask):
        print(f'the osim marker {osim.markers_labels[osim_marker_index]} corresponds to the skel vertices {skel_verts}')
    
    # TODO
    # At that point, SKEL should be fitted to the sequence with the following losses 
    # joint_losses = (skel_output.joints[:, mask, :] - osim_joints)
    # marker_losses = (skel_output.skin_verts[:, skel_verts_mask, :] - osim_markers[:, osim_mk_mask, :]) # ensure proper rotation of the forearm
    # By optimizing its betas and poses parameter
    