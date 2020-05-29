"""
    Preprocessing of InterTVA  resting state fMRI data for the computing of the correlation matrix (voxels/ROIs)

    Processing
    ==========

    1 - Avoid nan values

    2- Projection of nni files to gii files using white mesh

    3- Construction of activation matrix



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

    python preprocessing_tfmri.py 4 5 6




"""

import sys
import numpy as np
import nibabel as nib
from utils import run,  spm_run_batch, matlab_run_script, matlab_define, run_script_matlab
import nibabel.gifti as ng
from functions import project_epi, correlation, solve_nan, convert_mesh, project_epi_fsaverage5
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

    print("Import tfMRI of " + sub)
    run('mkdir {}/{}/ -vp'.format(subdir, sub))
    run("cp -r {}/{}/glm/vol/u{}_task-localizer_model-singletrial_denoised {}/{}/".format(intertva_dir, sub, sub, subdir, sub))

def solve_nan_tfmri(subdir, sub):

    gii_dir = subdir + "/{}/u{}_task-localizer_model-singletrial_denoised/".format(sub, sub)
    for ct in range(1,145):
        nii_file= "beta_{:04d}".format(ct)
        filename = gii_dir + nii_file + ".nii"
        print(filename)
        ex = filename
        nii = nib.load(ex)
        data = nii.get_data().copy()
        # print("Dimensions:", data.shape)
        # Test on NAN Values
        test = np.isnan(data[:])
        if True in test[:]:
            if ct == 1:
                print("NAN VALUES EXIST")
                print("Number of Nan Values:", np.sum(test))
            # Convert Boolean Matrix to int (0 or 1)
            mat = test.astype(int)
            # print(mat)
            [x, y, z] = mat.shape
            for k in range(0, z):
                for i in range(0, x):
                    for j in range(0, y):
                        if test[i, j, k] == 1:  # if NAN exist
                            if k == 0:  # if the first matrix (3D)
                                if i == 0:  # if the first line
                                    if j == 0:  # if the first column
                                        V = np.concatenate((data[i:i + 2, j:j + 2, k].flatten(),
                                                            data[i:i + 2, j:j + 2, k + 1].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])  # Extract non nan values
                                        data[i, j, k] = m
                                    elif j == y - 1:  # if the end column
                                        V = np.concatenate((data[i:i + 2, j - 1:j + 2, k].flatten(),
                                                            data[i:i + 2, j - 1:j + 2, k + 1].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                elif i == x - 1:  # if the end line
                                    if j == 0:  # if the first column
                                        V = np.concatenate((data[i - 1:i + 1, j:j + 2, k].flatten(),
                                                            data[i - 1:i + 1, j:j + 2, k + 1].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                    elif j == y - 1:  # if the end column
                                        V = np.concatenate((data[i - 1:i + 1, j - 1:j + 1, k].flatten(),
                                                            data[i - 1:i + 1, j - 1:j + 1, k + 1].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                else:
                                    if j == 0:  # If the first column
                                        V = np.concatenate((data[i - 1:i + 2, j:j + 2, k].flatten(),
                                                            data[i - 1:i + 2, j:j + 2, k + 1].flatten()), axis=None)
                                        # Extract no nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                    elif j == y - 1:  # end column
                                        V = np.concatenate((data[i - 1:i + 2, j - 1:j + 1, k].flatten(),
                                                            data[i - 1:i + 2, j - 1:j + 1, k + 1].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                    else:
                                        V = np.concatenate((data[i - 1:i + 2, j - 1:j + 2, k].flatten(),
                                                            data[i - 1:i + 2, j - 1:j + 2, k + 1].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])  # Extract non nan values
                                        data[i, j, k] = m
                            elif k == z - 1:  # if the end matrix
                                if i == 0:  # if the first line
                                    if j == 0:  # if the first column
                                        V = np.concatenate((data[i:i + 2, j:j + 2, k - 1].flatten(),
                                                            data[i:i + 2, j:j + 2, k].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])  # Extract non nan values
                                        data[i, j, k] = m
                                    elif j == y - 1:  # if the end column
                                        V = np.concatenate((data[i:i + 2, j - 1:j + 2, k - 1].flatten(),
                                                            data[i:i + 2, j - 1:j + 2, k].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                elif i == x - 1:  # if the end line
                                    if j == 0:  # if the first column
                                        V = np.concatenate((data[i - 1:i + 1, j:j + 2, k - 1].flatten(),
                                                            data[i - 1:i + 1, j:j + 2, k].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                    elif j == y - 1:  # if the end column
                                        V = np.concatenate((data[i - 1:i + 1, j - 1:j + 1, k - 1].flatten(),
                                                            data[i - 1:i + 1, j - 1:j + 1, k].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                else:
                                    if j == 0:  # If the first column
                                        V = np.concatenate((data[i - 1:i + 2, j:j + 2, k - 1].flatten(),
                                                            data[i - 1:i + 2, j:j + 2, k].flatten()), axis=None)
                                        # Extract no nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m

                                    elif j == y - 1:  # end column
                                        V = np.concatenate((data[i - 1:i + 2, j - 1:j + 1, k - 1].flatten(),
                                                            data[i - 1:i + 2, j - 1:j + 1, k].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                    else:
                                        V = np.concatenate((data[i - 1:i + 2, j - 1:j + 2, k - 1].flatten(),
                                                            data[i - 1:i + 2, j - 1:j + 2, k].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])  # Extract non nan values
                                        data[i, j, k] = m
                            else:
                                if i == 0:  # if the first line
                                    if j == 0:  # if the first column
                                        V = np.concatenate((data[i:i + 2, j:j + 2, k - 1].flatten(),
                                                            data[i:i + 2, j:j + 2, k].flatten(),
                                                            data[i:i + 2, j:j + 2, k + 1].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])  # Extract non nan values
                                        data[i, j, k] = m
                                    elif j == y - 1:  # if the end column
                                        V = np.concatenate((data[i:i + 2, j - 1:j + 2, k - 1].flatten(),
                                                            data[i:i + 2, j - 1:j + 2, k].flatten(),
                                                            data[i:i + 2, j - 1:j + 2, k + 1].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                elif i == x - 1:  # if the end line
                                    if j == 0:  # if the first column
                                        V = np.concatenate((data[i - 1:i + 1, j:j + 2, k - 1].flatten(),
                                                            data[i - 1:i + 1, j:j + 2, k].flatten(),
                                                            data[i - 1:i + 1, j:j + 2, k + 1].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                    elif j == y - 1:  # if the end column
                                        V = np.concatenate((data[i - 1:i + 1, j - 1:j + 1, k - 1].flatten(),
                                                            data[i - 1:i + 1, j - 1:j + 1, k].flatten(),
                                                            data[i - 1:i + 1, j - 1:j + 1, k + 1].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                else:
                                    if j == 0:  # If the first column
                                        V = np.concatenate((data[i - 1:i + 2, j:j + 2, k - 1].flatten(),
                                                            data[i - 1:i + 2, j:j + 2, k].flatten(),
                                                            data[i - 1:i + 2, j:j + 2, k + 1].flatten()), axis=None)
                                        # Extract no nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m
                                        # print(V)
                                        # print(i, j, k)
                                        # print(m)
                                    elif j == y - 1:  # end column
                                        V = np.concatenate((data[i - 1:i + 2, j - 1:j + 1, k - 1].flatten(),
                                                            data[i - 1:i + 2, j - 1:j + 1, k].flatten(),
                                                            data[i - 1:i + 2, j - 1:j + 1, k + 1].flatten()), axis=None)
                                        # Extract non nan values
                                        m = np.mean(V[~np.isnan(V)])
                                        data[i, j, k] = m

                                    else:
                                        V = np.concatenate((data[i - 1:i + 2, j - 1:j + 2, k - 1].flatten(),
                                                            data[i - 1:i + 2, j - 1:j + 2, k].flatten(),
                                                            data[i - 1:i + 2, j - 1:j + 2, k + 1].flatten()), axis=None)
                                        m = np.mean(V[~np.isnan(V)])  # Extract non nan values
                                        data[i, j, k] = m
        else:
            print("NO NAN :) ")
        array_img = nib.Nifti1Image(data, nii.affine, nii.header)
        # new_img.to_filename(new_fname)
        nib.save(array_img, filename)
        print("nan solved on:", filename)





def gifti_files(subdir, sub):
    """"
    This code allows to construct the activation matrix using gifti files


    """
    trgt_fsaverage5 = '/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage5/label/'

    subname = sub
    gii_dir = subdir + "/{}/u{}_task-localizer_model-singletrial_denoised/".format(sub, sub)

    """""
    STEP 1: MATRIX CONSTRUCTION  OF lh.Gifti AND rh.Gifti files 
    """""
    print("STEP 1: MATRIX CONSTRUCTION  OF lh.Gifti AND rh.Gifti files")
    hem_list = ['lh', 'rh']
    gii_matrix = np.empty([])
    for hem in hem_list:
        print("load of " + hem + ".gifti files")
        for i in range(1, 145):
            filename = gii_dir + "/beta_{:04d}.{}_fsaverage5.gii".format(i, hem)
            gii = ng.read(filename)
            data = np.array([gii.darrays[0].data])
            if i == 1:
                gii_matrix = data
            else:
                gii_matrix = np.concatenate((gii_matrix, data), axis=0)
        gii_matrix = np.transpose(gii_matrix)
        print("Size of Matrix " + hem + ":", gii_matrix.shape)
        if hem == 'lh':
            file = gii_dir+ "/gii_matrix_fsaverage5_lh.npy"
            np.save(file, gii_matrix)
        else:
            file = gii_dir + "/gii_matrix_fsaverage5_rh.npy"
            np.save(file, gii_matrix)

def projection(subdir, sub):
    gii_dir = subdir + "/{}/u{}_task-localizer_model-singletrial_denoised/".format(sub, sub)
    fs_subdir = "/hpc/banco/cagna.b/my_intertva/surf/data/" + sub + "/fs"
    for ct in range(1,145):
        filename = "beta_{:04d}".format(ct)
        nii_file = gii_dir + filename + ".nii"
        project_epi(fs_subdir, sub, nii_file, filename, gii_dir, tgt_subject=sub, hem_list=['lh', 'rh'], sfwhm=0)

def projection_fsaverage(subdir, sub):
    gii_dir = subdir + "/{}/u{}_task-localizer_model-singletrial_denoised/".format(sub, sub)
    fs_subdir = "/hpc/banco/cagna.b/my_intertva/surf/data/" + sub + "/fs"

    for ct in range(1, 145):
        filename = "beta_{:04d}".format(ct)
        nii_file = gii_dir + filename + ".nii"
        project_epi(fs_subdir, sub, nii_file, filename, gii_dir, tgt_subject='fsaverage', hem_list=['lh', 'rh'],
                    sfwhm=0)

def projection_fsaverage5(subdir, sub):
    gii_dir = subdir + "/{}/u{}_task-localizer_model-singletrial_denoised/".format(sub, sub)
    fs_subdir = "/hpc/banco/cagna.b/my_intertva/surf/data/" + sub + "/fs"

    for ct in range(1, 145):
        filename = "beta_{:04d}".format(ct)
        nii_file = gii_dir + filename + ".nii"
        project_epi_fsaverage5(fs_subdir, sub, nii_file, filename, gii_dir, tgt_subject='fsaverage5', hem_list=['lh', 'rh'],
                    sfwhm=0)


def pipeline(root, sub, src_bids, intertva_dir):
    subdir = root + "/tfmri"
    matlabdir = root + "/scripts/matlab"


    # Importation of data
    import_data(src_bids, intertva_dir, subdir, sub)


    # Check and solve the nan values in nii files
    solve_nan_tfmri(subdir, sub)

    # Projection of nii files into gii files on freesurfer6 template
    projection_fsaverage5(subdir, sub)

    # Convert white mesh to gii format
    convert_mesh(subdir, sub)


    # Construct the activation matrix

    gifti_files(subdir, sub)


# ************************ INTERPRETER *****************************************
if __name__ == "__main__":
    rt = "/hpc/banco/sellami.a/InterTVA"
    orig_bids = "/hpc/banco/cagna.b/my_intertva/openneuro/bids"
    intertva = "/hpc/banco/cagna.b/my_intertva/surf/data"

    # Process each subject that specified in the command line
    for i in range(1, len(sys.argv)):
        pipeline(rt, sys.argv[i], orig_bids, intertva) # No Interactive mode
        #pipeline(rt, "sub-{:02d}".format(int(sys.argv[i])), orig_bids, intertva) # Interactive mode
