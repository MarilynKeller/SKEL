import os

package_directory = os.path.dirname(os.path.abspath(__file__))
# skel_folder = 'models/skel_models_v1.1/'
# smpl_folder = 'models/'

skel_folder = os.path.join(package_directory, '../models/skel_models_v1.1/')
smpl_folder = os.path.join(package_directory, '../models/')