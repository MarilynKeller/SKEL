# Copyright (C) 2024  Max Planck Institute for Intelligent Systems Tuebingen, Marilyn Keller 

import os
import tqdm
import numpy as np
import yaml
import nimblephysics as nimble
from skel import kin_skel

class OSIM:
    
    def __init__(self, osim_path, motion_path):
        self.osim_path = osim_path
        self.motion_path = motion_path
        self.osim = OSIM.load_osim(osim_path)
        self.motion = OSIM.load_motion(self.osim, motion_path)
        self.n_frames = self.motion.shape[0]
        self.markers_labels = OSIM.get_markers_labels(self.osim)
        self.joints_labels = OSIM.get_joints_labels(self.osim)

    @classmethod
    def load_osim(cls, osim_path):
        """Load an osim file"""
        assert os.path.exists(osim_path), "Could not find osim file: {}".format(osim_path)
        osim : nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
        assert osim is not None, "Could not load osim file: {}".format(osim_path)
        return osim
    
    @classmethod
    def load_motion(cls, osim, motion_path):
        """Load a .mot motion file"""
        assert os.path.exists(motion_path), "Could not find motion file: {}".format(motion_path)
        mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(osim.skeleton, motion_path)
        assert mot is not None, "Could not load motion file: {}".format(motion_path)
        motion = np.array(mot.poses.T)    
        return motion
    
    @classmethod
    def get_markers_labels(cls, osim):
        """Return a list of markers in the osim file"""
        markers_labels = [ml for ml in osim.markersMap.keys()]
        # markers_labels.sort()
        return markers_labels
    
    @classmethod
    def get_joints_labels(cls, osim):
        node_names = [n.getName() for n in osim.skeleton.getBodyNodes()]
        return node_names
        
    def skel_joint_mask(self, skel_mapping):
        """ Return a mask to map the joints of the skel model to the joints of the osim model"""
        corresp_skel_joint_ids = [] # For each osim joint, the index of the corresponding skel joint
        
        joint_mapping_dict = skel_mapping['joints_mapping']
        skel_joint_names = kin_skel.skel_joints_name
        for osim_joint_name in joint_mapping_dict.keys():
            skel_joint_name = joint_mapping_dict[osim_joint_name]
            skel_joint_index = skel_joint_names.index(skel_joint_name)
            corresp_skel_joint_ids.append(skel_joint_index)
        
        import ipdb; ipdb.set_trace()
        return corresp_skel_joint_ids
    
    def skel_marker_mask(self, skel_mapping):
        
        markers_mapping_dict = skel_mapping['markers_mapping']
        
        osim_mask = []
        skel_verts_mask = []
        for osim_marker_name in self.markers_labels:
            osim_marker_index = self.markers_labels.index(osim_marker_name)
            osim_mask.append(osim_marker_index)
            
            skel_verts_mask.append(markers_mapping_dict[osim_marker_name])
        return osim_mask, skel_verts_mask
    
    def run_fk(self):
        """Return a list of markers in the osim file"""
        markers = np.zeros((self.n_frames, len(self.markers_labels), 3))
        joints = np.zeros([self.n_frames, len(self.joints_labels), 3])
        joints_ori = np.zeros([self.n_frames, len(self.joints_labels), 3, 3])
        
        for frame_id in (pbar := tqdm.tqdm(range(self.n_frames))):
            pbar.set_description("Running OpenSim forward kinematics")

            pose = self.motion[frame_id, :]

            # Pose osim
            self.osim.skeleton.setPositions(self.motion[frame_id, :])

            # Since python 3.6, dicts have a fixed order so the order of this list should be marching labels
            markers[frame_id, :, :] = np.vstack(self.osim.skeleton.getMarkerMapWorldPositions(self.osim.markersMap).values())
            #Sanity check for previous comment
            assert list(self.osim.skeleton.getMarkerMapWorldPositions(self.osim.markersMap).keys()) == self.markers_labels, "Marker labels are not in the same order"

            for ni, node_name in enumerate(self.joints_labels):
                # part transfo
                transfo = self.osim.skeleton.getBodyNode(node_name).getWorldTransform()
                    
                # Update joint                    
                joints[frame_id, ni, :] = transfo.translation()
                joints_ori[frame_id, ni, :, :] = transfo.rotation()

        return markers, joints, joints_ori
        

