# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import pickle

import trimesh

from skel.alignment.aligner import SkelFitter
from skel.alignment.utils import load_smpl_seq


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Align SKEL to a SMPL frame')
    
    parser.add_argument('--smpl_mesh_path', type=str, help='Path to the SMPL mesh to align to', default=None)
    parser.add_argument('--smpl_data_path', type=str, help='Path to the SMPL dictionary to align to (.pkl or .npz)', default=None)
    parser.add_argument('-o', '--out_dir', type=str, help='Output directory', default='output')
    parser.add_argument('-F', '--force-recompute', help='Force recomputation of the alignment', action='store_true')
    parser.add_argument('--gender', type=str, help='Gender of the subject (only needed if not provided with smpl_data_path)', default='female')
    parser.add_argument('--config', help='Yaml config file containing parameters for training. \
                    You can create a config tailored to align a specific sequence. When left to None, \
                        the default config will be used', default=None)
    
    args = parser.parse_args()

    smpl_data = load_smpl_seq(args.smpl_data_path, gender=args.gender, straighten_hands=False)
    
    if args.smpl_mesh_path is not None:
        subj_name = os.path.basename(args.smpl_seq_path).split(".")[0]
    elif args.smpl_data_path is not None:
        subj_name = os.path.basename(args.smpl_data_path).split(".")[0]
    else:
        raise ValueError('Either smpl_mesh_path or smpl_data_path must be provided')
    
    # Create the output directory
    subj_dir = os.path.join(args.out_dir, subj_name)
    os.makedirs(subj_dir, exist_ok=True)
    pkl_path = os.path.join(subj_dir, subj_name+'_skel.pkl')  
    
    subj_dir = subj_dir
    
    if os.path.exists(pkl_path) and not args.force_recompute:
        print('Previous aligned SKEL sequence found at {}. Will be used as initialization.'.format(subj_dir))
        skel_data_init = pickle.load(open(pkl_path, 'rb'))
    else:
        skel_data_init = None
    
    skel_fitter = SkelFitter(smpl_data['gender'], 
                             device='cuda:0', 
                             export_meshes=True, 
                             config_path=args.config)
    skel_seq = skel_fitter.run_fit(smpl_data['trans'], 
                               smpl_data['betas'], 
                               smpl_data['poses'],
                               batch_size=1,
                               skel_data_init=skel_data_init, 
                               force_recompute=args.force_recompute)
    
    print('Saved aligned SKEL to {}'.format(subj_dir))
    
    SKEL_skin_mesh = trimesh.Trimesh(vertices=skel_seq['skin_v'][0], faces=skel_seq['skin_f'])
    SKEL_skel_mesh = trimesh.Trimesh(vertices=skel_seq['skel_v'][0], faces=skel_seq['skel_f'])
    SMPL_mesh = trimesh.Trimesh(vertices=skel_seq['smpl_v'][0], faces=skel_seq['smpl_f'])
    
    SKEL_skin_mesh.export(os.path.join(subj_dir, subj_name + '_skin.obj'))
    SKEL_skel_mesh.export(os.path.join(subj_dir, subj_name + '_skel.obj'))
    SMPL_mesh.export(os.path.join(subj_dir, subj_name + '_smpl.obj'))
    
    pickle.dump(skel_seq, open(pkl_path, 'wb'))
    
    print('SKEL parameters saved to {}'.format(subj_dir))
    print('SKEL skin mesh saved to {}'.format(os.path.join(subj_dir, subj_name + '_skin.obj')))
    print('SKEL skel mesh saved to {}'.format(os.path.join(subj_dir, subj_name + '_skel.obj')))
    print('SMPL mesh saved to {}'.format(os.path.join(subj_dir, subj_name + '_smpl.obj')))
    