"""
    Preprocessing of InterTVA  resting state fMRI data for the computing of the correlation matrix (voxels/ROIs)

    Processing
    ==========

    1 - Importation of resting state fMRI data (from Bastien Cagna)

    2 - Preprocessing of BOLD data using SPM 12:
        - Proc 1: Slice Timing
        - Proc 2: Compute VDM
        - Proc 3: Realign and Unwarp
        - Proc 4: Coregistration to already coregistred Voice Localizer BOLD image
        - Proc 5: Noise regressors estimation  with PhysioTapas

    3 - Create mask

    4 - Model specification: GLM

    5 - Use of explicit masks to calculate the GLM in SPM

    6 - Model estimation

    7 - Check if exist a nan values in nii files

    8 - Projection of nii files into gii files on freesurfer6 template

    9- Compute the correlation matrix:
        - Proc 1: Extract all matrix of gii files
        - Proc 2: ROIs averaging
        - Proc 3: Correlation Matrix (Voxels/ ROIs)


    Softwares
    =========

    This pipeline use:
        * Python 3
        * SPM12 toolbox in Matlab 2018
        * FSL 5.0 (fslmaths)
        * Freesurfer 6

    Arguments
   =========

    List of subjects' number (or ID)


    Example
    =======

    python preprocessing_rsfmri.py 4 5 6


    Next step
    =========

    After running this pipeline on a subject, you can visualize the obtained correlation matrix  with: python visualization.py sub-xx

"""

import sys
import numpy as np
import nibabel as nib
from utils import run,  spm_run_batch, matlab_run_script, matlab_define, run_script_matlab
from functions import project_epi, correlation, solve_nan, convert_mesh, correlation_voxel_voxel
# ************************** PIPELINE ******************************************
def import_data(src_bids, intertva_dir, subdir, sub):
    """
        Import all needed files for one subject

        :param src_bids:        Original BIDS directory (from Bastien Cagna)
        :param intertva_dir:    Already preprocessed and anaysed data directory
        :param subdir:          New directory for rsfMRI data
        :param sub:             Subject's ID
    """
    # Import data
    # Resting state fMRI
    print("Import resting state fMRI of " + sub)
    run('mkdir {}/{}/func -vp'.format(subdir, sub))
    run("cp {}/{}/func/{}_task-rest_bold.nii.gz {}/{}/func/".format(src_bids, sub, sub, subdir, sub))
    run('gunzip {}/{}/func/{}_task-rest_bold.nii.gz '.format(subdir, sub, sub))

    # fmap data
    print("Import fmap data of " + sub)
    run('mkdir {}/{}/fmap -vp'.format(subdir, sub))
    run("cp {}/{}/fmap/{}_acq-topup2_fieldmap.nii.gz {}/{}/fmap/".format(src_bids, sub, sub, subdir, sub))
    run('gunzip {}/{}/fmap/{}_acq-topup2_fieldmap.nii.gz '.format(subdir, sub, sub))
    run("cp {}/{}/fmap/{}_acq-topup2_magnitude.nii.gz {}/{}/fmap/".format(src_bids, sub, sub, subdir, sub))
    run('gunzip {}/{}/fmap/{}_acq-topup2_magnitude.nii.gz '.format(subdir, sub, sub))


    # Anatomical MRI
    print("Import Anatomical MRI of " + sub)
    run('mkdir {}/{}/anat -vp'.format(subdir, sub))
    run("cp {}/{}/anat/masked_msanlm_{}_T1w.nii {}/{}/anat/".format(intertva_dir, sub, sub, subdir, sub))
    run("cp {}/{}/anat/c2sanlm_{}_T1w.nii {}/{}/anat/".format(intertva_dir, sub, sub, subdir, sub))
    run("cp {}/{}/anat/c3sanlm_{}_T1w.nii {}/{}/anat/".format(intertva_dir, sub, sub, subdir, sub))

    # Localizer data
    print("Import localizer data of " + sub)
    run('mkdir {}/{}/func/localizer/vol -vp'.format(subdir, sub))  # Localizer files
    run('cp {}/{}/func/localizer/vol/u{}_task-localizer_bold.nii {}/{}/func/localizer/vol/ -v'.format(intertva_dir, sub, sub, subdir, sub))

    # Annotations files
    print("Import labeled data of " + sub)
    run('mkdir {}/{}/label -vp'.format(subdir, sub)) # Annotations files
    run('cp {}/{}/fs/{}/label/lh.aparc.a2009s.annot {}/{}/label/ -v'.format(intertva_dir, sub, sub, subdir, sub))
    run('cp {}/{}/fs/{}/label/rh.aparc.a2009s.annot {}/{}/label/ -v'.format(intertva_dir, sub, sub, subdir, sub))

def create_mask(root, sub):
    func_f = root+'/rsfmri/{}/func/ua{}_task-rest_bold.nii'.format(sub, sub)

    func_nii = nib.load(func_f)
    data = func_nii.get_data()
    affine = func_nii.get_affine()
    print(affine)
    print(data.shape)

    mask = np.ones(data.shape[:3])

    mask_nii = nib.Nifti1Image(mask, affine)
    print(mask_nii.get_data().shape)
    nib.save(mask_nii, func_f[:-4] + '_fullimage_mask.nii')
def projection(subdir, sub, template):
    gii_dir = subdir + "/" + sub + "/glm/noisefiltering/"
    fs_subdir = "/hpc/banco/cagna.b/my_intertva/surf/data/" + sub + "/fs"
    for ct in range(1, 621):
        filename = "Res_{:04d}".format(ct)
        nii_file = gii_dir + filename + ".nii"
        project_epi(fs_subdir, sub, nii_file, filename, gii_dir, tgt_subject=template, hem_list=['lh', 'rh'], sfwhm=0)



def pipeline(root, sub, src_bids, intertva_dir):
    subdir = root + "/rsfmri"
    matlabdir = root + "/scripts/matlab"


    #  load resting-state fMRI
    import_data(src_bids, intertva_dir, subdir, sub)

    # preprocessing of resting state fMRI: Slice Timing, Compute VDM, Realing & Unwrap, Coregistration, and PhysioTapas
     spm_run_batch(root + "/script_batch_rsfmri/"
                           "preprocess.m", {'subdir': subdir,'subname': sub}, display=False)
    #    PhysioTapas
     spm_run_batch(root + "/script_batch_rsfmri/"
                            "batch_physio.m", {'subdir': subdir,'subname': sub}, display=True)

    #  create mask
    create_mask(root, sub)


    # Model Specification with GLM
    spm_run_batch(root + "/script_batch_rsfmri/"
                               "model_specification.m", {'subdir': subdir,'subname': sub}, display=False)

    #  Use of explicit masks to calculate the GLM in SPM
    run_script_matlab(root + "/script_batch_rsfmri/"
                                "mask_spm.m", {'subdir': subdir,'subname': sub})

    # Model Estimation
    spm_run_batch(root + "/script_batch_rsfmri/"
                                "model_estimation.m", {'subdir': subdir,'subname': sub}, display=False)


    # Check and solve the nan values in nii files
    solve_nan(subdir, sub)

    # Projection of nii files into gii files on freesurfer6 template: sub (native space), fsaverage, fsaverage5, or fsaverage6
    template='fsaverage5'
    projection(subdir, sub, template)

    # Convert white mesh to gii format
    convert_mesh(subdir, sub)

    # Compute the correlation matrix (Voxels/ROIs)
    correlation(subdir, sub, template)

    # Compute the correlation matrix (Voxels/ROIs)
    correlation_voxel_voxel(subdir, sub, template)



# ************************ INTERPRETER *****************************************
if __name__ == "__main__":
    rt = "/hpc/banco/sellami.a/InterTVA"
    orig_bids = "/hpc/banco/cagna.b/my_intertva/openneuro/bids"
    intertva = "/hpc/banco/cagna.b/my_intertva/surf/data/"

    # Process each subject that specified in the command line
    for i in range(1, len(sys.argv)):
        pipeline(rt, sys.argv[i], orig_bids, intertva) # No Interactive mode
        #pipeline(rt, "sub-{:02d}".format(int(sys.argv[i])), orig_bids, intertva) # Interactive mode
