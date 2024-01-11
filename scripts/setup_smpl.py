import argparse
import os
import shutil
import zipfile


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Rename SMPL files to be loadable by smplx')
    
    parser.add_argument('smpl_zip', type=str, default='/path/to/SMPL_python_v.1.1.0.zip')
    
    args = parser.parse_args()
    
    model_folder = 'models'
    smpl_zip = args.smpl_zip
    
    # Unzip the SMPL model
    print('Unzipping SMPL model...')
    with zipfile.ZipFile(smpl_zip, 'r') as zip_ref:
        zip_ref.extractall(model_folder)

    os.makedirs(os.path.join(model_folder, 'smpl'), exist_ok=True)
    
    # Rename the SMPL files
    print('Renaming SMPL files...')
    shutil.move(os.path.join(model_folder, 'SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl'),
                os.path.join(model_folder, 'smpl', 'SMPL_FEMALE.pkl'))
    
    shutil.move(os.path.join(model_folder, 'SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl'),
            os.path.join(model_folder, 'smpl', 'SMPL_MALE.pkl'))
    
    shutil.move(os.path.join(model_folder, 'SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'),
        os.path.join(model_folder, 'smpl', 'SMPL_NEUTRAL.pkl'))
    
    print("Cleaning up...")
    shutil.rmtree(os.path.join(model_folder, 'SMPL_python_v.1.1.0'))
    print("Done!")
    print(" The SMPL models were placed in models/. You can now use the smplx library to load the SMPL model.")