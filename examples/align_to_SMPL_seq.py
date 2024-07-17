# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import pickle

import trimesh

from skel.alignment.aligner import SkelFitter
from skel.alignment.utils import load_smpl_seq


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Align SKEL to a SMPL sequence')
    
    parser.add_argument('smpl_seq_path', type=str, help='Path to the SMPL sequence')
    parser.add_argument('-o', '--out_dir', type=str, help='Output directory', default='output')
    parser.add_argument('-F', '--force-recompute', help='Force recomputation of the alignment', action='store_true')
    parser.add_argument('-D', '--debug', help='Only run the fit on the first minibach to test', action='store_true')
    parser.add_argument('-B', '--batch_size', type=int, help='Batch size', default=3000)
    parser.add_argument('-w', '--watch_frame', type=int, help='Frame of the batch to display', default=0)
    parser.add_argument('--gender', type=str, help='Gender of the subject (only needed if not provided with smpl_data_path)', default='female')
    parser.add_argument('-m', '--export_meshes', choices=[None, 'mesh', 'pickle'], default=None, 
                        help='If not None, export the resulting meshes (skin and skeleton), either as .obj' \
                            +'files or as a pickle file containing the vertices and faces')
    parser.add_argument('--config', help='Yaml config file containing parameters for training. \
                    You can create a config tailored to align a specific sequence. When left to None, \
                        the default config will be used', default=None)
    
    args = parser.parse_args()
    
    smpl_seq = load_smpl_seq(args.smpl_seq_path, gender=args.gender, straighten_hands=False)
    
    # Create the output directory
    subj_name = os.path.basename(args.smpl_seq_path).split(".")[0]
    subj_dir = os.path.join(args.out_dir, subj_name)
    os.makedirs(subj_dir, exist_ok=True)
    pkl_path = os.path.join(subj_dir, subj_name+'_skel.pkl')  
     
    # Load the previous aligned SKEL sequence if it exists to use it as initialization
    if os.path.exists(pkl_path) and not args.force_recompute:
        print('Previous aligned SKEL sequence found at {}. Will be used as initialization.'.format(pkl_path))
        skel_data_init = pickle.load(open(pkl_path, 'rb'))
    else:
        skel_data_init = None
    
    skel_fitter = SkelFitter(smpl_seq['gender'], 
                             device='cuda:0', 
                             export_meshes=args.export_meshes is not None,
                             config_path=args.config)
    skel_seq = skel_fitter.run_fit(smpl_seq['trans'], 
                               smpl_seq['betas'], 
                               smpl_seq['poses'], 
                               batch_size = args.batch_size,
                               skel_data_init=skel_data_init, 
                               force_recompute=args.force_recompute,
                               debug=args.debug,
                               watch_frame=args.watch_frame)
    
    if args.export_meshes == 'mesh':
        # Remove the vertices and faces from  the pkl file to make it lighter
        skel_seq = {key: val for key, val in skel_seq.items() if key not in ['skel_v', 'skel_f', 'skin_v', 'skin_f', 'smpl_v', 'smpl_f']}
        
        for folder in ['SKEL_skin', 'SKEL_skel', 'SMPL']:
            os.makedirs(os.path.join(subj_dir, 'meshes', folder), exist_ok=True)
    
        mesh_folder = os.path.join(subj_dir, 'meshes')
        for i in range(skel_seq['skel_v'].shape[0]):
            skin_mesh = trimesh.Trimesh(vertices=skel_seq['skin_v'][i], faces=skel_seq['skin_f'])
            skin_mesh.export(os.path.join(mesh_folder, 'SKEL_skin', f'skel_skin_{i}.obj'))
            
            skel_mesh = trimesh.Trimesh(vertices=skel_seq['skel_v'][i], faces=skel_seq['skel_f'])
            skel_mesh.export(os.path.join(mesh_folder, 'SKEL_skel', f'skel_skel{i}.obj'))
            
            smpl_mesh = trimesh.Trimesh(vertices=skel_seq['smpl_v'][i], faces=skel_seq['smpl_f'])
            smpl_mesh.export(os.path.join(mesh_folder, 'SMPL', f'smpl_{i}.obj'))
        

    pickle.dump(skel_seq, open(pkl_path, 'wb'))
    print('Saved aligned SKEL sequence to {}'.format(pkl_path))