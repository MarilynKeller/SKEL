# Copyright (C) 2024  MPI IS, Marilyn Keller
import argparse
import os
import pickle

from skel.alignment.aligner import SkelFitter, load_smpl_seq


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Align SKEL to a SMPL sequence')
    
    parser.add_argument('smpl_seq_path', type=str, help='Path to the SMPL sequence')
    parser.add_argument('-o', '--out_dir', type=str, help='Output directory', default='output')
    parser.add_argument('-F', '--force-recompute', help='Force recomputation of the alignment', action='store_true')
    parser.add_argument('-D', '--debug', help='Only run the fit on the first minibach to test', action='store_true')
    parser.add_argument('-B', '--batch-size', type=int, help='Batch size', default=3000)
    parser.add_argument('-w', '--watch-frame', type=int, help='Frame of the batch to display', default=0)
    
    args = parser.parse_args()
    
    smpl_seq = load_smpl_seq(args.smpl_seq_path)
    
    outfile = os.path.basename(args.smpl_seq_path).split(".")[0] + '_skel.pkl'
    out_path = os.path.join(args.out_dir, outfile)
    
    if os.path.exists(out_path):
        print('Previous aligned SKEL sequence found at {}. Will be used as initialization.'.format(out_path))
        skel_data_init = pickle.load(open(out_path, 'rb'))
    else:
        skel_data_init = None
    
    skel_fitter = SkelFitter(smpl_seq['gender'], device='cuda:0')
    skel_seq = skel_fitter.fit(smpl_seq['trans'], 
                               smpl_seq['betas'], 
                               smpl_seq['poses'], 
                               batch_size = args.batch_size,
                               skel_data_init=skel_data_init, 
                               force_recompute=args.force_recompute,
                               debug=args.debug,
                               watch_frame=args.watch_frame)
    
    pickle.dump(skel_seq, open(out_path, 'wb'))
    print('Saved aligned SKEL sequence to {}'.format(out_path))