import os

package_directory = os.path.dirname(os.path.abspath(__file__))

skel_folder = os.path.join(package_directory, '../models/skel_models_v1.1/')
smpl_folder = os.path.join(package_directory, '../models/')
fitting_mask_file = os.path.join(package_directory, 'alignment/riggid_parts_mask.pkl')
default_config_file = os.path.join(package_directory, 'alignment/default_config.yaml')