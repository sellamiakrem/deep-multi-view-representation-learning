

import os
import numpy as np
import nibabel as nib
import nibabel.gifti as ng
import time
from scipy.stats import pearsonr


# ***** PROJECTION *************************************************************
def project_epi(fs_subdir, sub, nii_file, filename, gii_dir,
                tgt_subject, hem_list=['lh', 'rh'], sfwhm=0):
    """
        Project one Nifti file (3D image) to surface saved as Gifti file.

        Projection is done for left and right
        :param fs_subdir: FreeSurfer subjects directory
        :param sub: Subject name
        :param nii_file: Splitted .nii directory
        :param gii_dir: Output directory
        :param gii_sfx: Gifti files suffix (add to the hemisphere name)
        :param tgt_subject: Name of target subject
        :param hem_list: Hemispheres (default: left and right)
        :param sfwhm: Surface smoothing (default = 0mm)
    """
    for hem in hem_list:
        gii_file = "{}/{}.{}_{}.gii".format(gii_dir, filename, hem, tgt_subject)

        cmd = '$FREESURFER_HOME/bin/mri_vol2surf --src {} --o {} ' \
              '--out_type gii --regheader {} --hemi {} '\
              '--projfrac-avg 0 1 0.1 --surf-fwhm {:d} --sd {} ' \
              '--trgsubject {}'.format(
               nii_file, gii_file, sub, hem, sfwhm, fs_subdir, tgt_subject)
        os.system(cmd)
def project_epi_fsaverage5(fs_subdir, sub, nii_file, filename, gii_dir,
                tgt_subject, hem_list=['lh', 'rh'], sfwhm=0):
    """
        Project one Nifti file (3D image) to surface saved as Gifti file.

        Projection is done for left and right
        :param fs_subdir: FreeSurfer subjects directory
        :param sub: Subject name
        :param nii_file: Splitted .nii directory
        :param gii_dir: Output directory
        :param gii_sfx: Gifti files suffix (add to the hemisphere name)
        :param tgt_subject: Name of target subject
        :param hem_list: Hemispheres (default: left and right)
        :param sfwhm: Surface smoothing (default = 0mm)
    """
    for hem in hem_list:
        gii_file = "{}/{}.{}_fsaverage5.gii".format(gii_dir, filename, hem)

        cmd = '$FREESURFER_HOME/bin/mri_vol2surf --src {} --o {} ' \
              '--out_type gii --regheader {} --hemi {} ' \
              '--projfrac-avg 0 1 0.1 --surf-fwhm {:d} --sd {} ' \
              '--trgsubject {}'.format(
            nii_file, gii_file, sub, hem, sfwhm, fs_subdir, tgt_subject)
        os.system(cmd)

# ***** CORRELATION MATRIX ******************************************************************
def correlation(subdir, sub, template):
    """"
    This code allows to compute the correlation bewteen vowels and ROIs.
    It needs a set of labels (annotation files) and gii files.
    The code is decomposed into three phases (procedures)
        :proc  1: matrix construction of gii file (each line is a voxel, and the column is the j time serie)
        :proc  2: : for each ROI, we save the set of selected voxels based on the annotation file (labels)
        !proc  3: Coorelation matrix, for each voxel we compute their correlation with the average value of each ROI

    """
    trgt_fsaverage= '/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage/label/'
    trgt_fsaverage5 = '/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage5/label/'
    trgt= '/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/{}/label/'.format(template)

    subname = sub
    start = time.time()

    # FS AVERAGE 5
    """""
    STEP 1: MATRIX CONSTRUCTION  OF lh.Gifti AND rh.Gifti files
    """""
    print("STEP 1: MATRIX CONSTRUCTION  OF lh.Gifti AND rh.Gifti files")
    hem_list = ['lh', 'rh']
    gii_matrix = np.empty([])
    for hem in hem_list:
        print("load of " + hem + ".gifti files")
        for i in range(1, 621):
            filename = subdir + "/" + subname + "/glm/noisefiltering/Res_{:04d}.{}_{}.gii".format(i, hem, template)
            gii = ng.read(filename)
            data = np.array([gii.darrays[0].data])
            if i == 1:
                gii_matrix = data
            else:
                gii_matrix = np.concatenate((gii_matrix, data), axis=0)
        gii_matrix = np.transpose(gii_matrix)
        print("Size of Matrix " + hem + ":", gii_matrix.shape)
        if hem == 'lh':
            file = subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_{}_lh.npy".format(template)
            np.save(file, gii_matrix)
        else:
            file = subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_{}_rh.npy".format(template)
            np.save(file, gii_matrix)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Elapsed time Step 1: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        start = time.time()
        """""
        Step 2 : ROI averaging.
        """""
        # Read Annotation LH file
        print("STEP 2: ROI AVERAGING")
        print("Read annotation {} file: {}.aparc.a2009s.annot".format(hem, hem))
        #filepath = subdir + "/" + subname + "/label/{}.aparc.a2009s.annot".format(hem) # espace natif
        filepath = trgt+ "{}.aparc.a2009s.annot".format(hem)    # espace fsaverage
        [labels, ctab, names] = nib.freesurfer.io.read_annot(filepath, orig_ids=False)
        print("labels {}".format(hem), labels)
        print("Size of labels", labels.shape)
        # print("List of names", names)
        print("Number of ROIs", len(names))
        # ID Regions Extraction
        print("ID ROI extraction")
        Id_ROI = np.asarray(sorted(set(labels)))
        print("Ids ROIs:", Id_ROI)
        print("ROIs dimensions:", Id_ROI.shape)
        # print("len",len(labels))

        # Extract time series for each ROI (averaging)
        print('Extract time series for each ROI by averaging operation')
        roi_avg = np.empty((len(Id_ROI), gii_matrix.shape[1]))
        for i in range(0, len(Id_ROI)):
            print("*********************************************************")
            print("ID ROI:", Id_ROI[i])
            mask = np.where(labels == Id_ROI[i])
            roi_timeseries = gii_matrix[mask, :].mean(1)
            roi_avg[i, :] = roi_timeseries
        print("*********************************************************")
        print("********** Results: **********")
        print("Size of the Average Matrix of ALL ROIs", roi_avg.shape)
        # Save the average matrix of all ROIs
        if hem == 'lh':
            file = subdir + "/" + subname + "/glm/noisefiltering/roi_avg_lh_{}.npy".format(template)
            np.save(file, roi_avg)
        else:
            file = subdir + "/" + subname + "/glm/noisefiltering/roi_avg_rh_{}.npy".format(template)
            np.save(file, roi_avg)
        print("*********************************************************")
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Elapsed time Step 2: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        print("*********************************************************")
        start = time.time()
    """""
    Step 3 : Correlation Matrix
    """""
    print("STEP 3: COMPUTING OF THE CORRELATION MATRIX ")
    roi_avg_lh=np.load(subdir + "/" + subname + "/glm/noisefiltering/roi_avg_lh_{}.npy".format(template))
    roi_avg_rh = np.load(subdir + "/" + subname + "/glm/noisefiltering/roi_avg_rh_{}.npy".format(template))
    roi_avg=np.concatenate((roi_avg_lh, roi_avg_rh))
    print("roi avg shape", roi_avg.shape)
    gii_matrix_lh=np.load(subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_{}_lh.npy".format(template))
    gii_matrix_rh=np.load(subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_{}_rh.npy".format(template))
    gii_matrix=np.concatenate((gii_matrix_lh, gii_matrix_rh))
    print("gii matrix shape", gii_matrix.shape)
    correlation_matrix = np.empty((gii_matrix.shape[0], roi_avg.shape[0]))
    print("correlation matrix shape", correlation_matrix.shape)
    for n in range(gii_matrix.shape[0]):
        for m in range(roi_avg.shape[0]):
            correlation_matrix[n, m] = pearsonr(gii_matrix[n, :], roi_avg[m, :])[0]
    correlation_matrix[np.where(np.isnan(correlation_matrix[:]))] = 0
    file = subdir + "/" + subname + "/glm/noisefiltering/correlation_matrix_{}.npy".format(template)
    np.save(file, correlation_matrix)
    print("********** Results: **********")
    print("Dimensions of the correlation Matrix:", correlation_matrix.shape)
    print("Computing of the correlation matrix, DONE!:", subname)
    test = np.isnan(correlation_matrix[:])
    if True in test[:]:
        print("Nan values exist in correlation matrix")
        print("Number of Nan Values:", np.sum(test))
    else:
        print("No Nan values in correlation matrix")
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time Step 3: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds ))

def correlation_voxel_voxel(subdir, sub, template):
    """"
    This code allows to compute the correlation bewteen vowels and ROIs.
    It needs a set of labels (annotation files) and gii files.
    The code is decomposed into three phases (procedures)
        :proc  1: matrix construction of gii file (each line is a voxel, and the column is the j time serie)
        :proc  2: : for each ROI, we save the set of selected voxels based on the annotation file (labels)
        !proc  3: Coorelation matrix, for each voxel we compute their correlation with the average value of each ROI

    """
    trgt_fsaverage = '/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage/label/'
    trgt_fsaverage5 = '/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/fsaverage5/label/'
    trgt = '/hpc/soft/freesurfer/freesurfer_6.0.0/subjects/{}/label/'.format(template)

    subname = sub
    start = time.time()

    """""
    Step 1 : Correlation Matrix
    """""
    print("STEP 1: COMPUTING OF THE CORRELATION MATRIX ")
    # roi_avg_lh = np.load(subdir + "/" + subname + "/glm/noisefiltering/roi_avg_lh_{}.npy".format(template))
    # roi_avg_rh = np.load(subdir + "/" + subname + "/glm/noisefiltering/roi_avg_rh_{}.npy".format(template))
    # roi_avg = np.concatenate((roi_avg_lh, roi_avg_rh))
    # print("roi avg shape", roi_avg.shape)
    gii_matrix_lh = np.load(subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_{}_lh.npy".format(template))
    gii_matrix_rh = np.load(subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_{}_rh.npy".format(template))
    gii_matrix = np.concatenate((gii_matrix_lh, gii_matrix_rh))
    print("gii matrix shape", gii_matrix.shape)
    correlation_matrix = np.empty((gii_matrix.shape[0], gii_matrix.shape[0]))
    print("correlation matrix shape", correlation_matrix.shape)
    for n in range(gii_matrix.shape[0]):
        for m in range(gii_matrix.shape[0]):
            correlation_matrix[n, m] = pearsonr(gii_matrix[n, :], gii_matrix[m, :])[0]
    correlation_matrix[np.where(np.isnan(correlation_matrix[:]))] = 0
    file = subdir + "/" + subname + "/glm/noisefiltering/correlation_matrix_voxel_voxel_{}.npy".format(template)
    np.save(file, correlation_matrix)
    print("********** Results: **********")
    print("Dimensions of the correlation Matrix:", correlation_matrix.shape)
    print("Computing of the correlation matrix, DONE!:", subname)
    test = np.isnan(correlation_matrix[:])
    if True in test[:]:
        print("Nan values exist in correlation matrix")
        print("Number of Nan Values:", np.sum(test))
    else:
        print("No Nan values in correlation matrix")
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time Step 3: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    # Fsaverage

    # print("STEP 1: MATRIX CONSTRUCTION  OF lh.Gifti AND rh.Gifti files")
    # hem_list = ['lh', 'rh']
    # gii_matrix = np.empty([])
    # for hem in hem_list:
    #     print("load of " + hem + ".gifti files")
    #     for i in range(1, 621):
    #         filename = subdir + "/" + subname + "/glm/noisefiltering/Res_{:04d}.{}.gii".format(i, hem)
    #         gii = ng.read(filename)
    #         data = np.array([gii.darrays[0].data])
    #         if i == 1:
    #             gii_matrix = data
    #         else:
    #             gii_matrix = np.concatenate((gii_matrix, data), axis=0)
    #     gii_matrix = np.transpose(gii_matrix)
    #     print("Size of Matrix " + hem + ":", gii_matrix.shape)
    #     if hem == 'lh':
    #         file = subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_lh.npy"
    #         np.save(file, gii_matrix)
    #     else:
    #         file = subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_rh.npy"
    #         np.save(file, gii_matrix)
    #     end = time.time()
    #     hours, rem = divmod(end - start, 3600)
    #     minutes, seconds = divmod(rem, 60)
    #     print("Elapsed time Step 1: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    #     start = time.time()
    #     """""
    #     Step 2 : ROI averaging.
    #     """""
    #     # Read Annotation LH file
    #     print("STEP 2: ROI AVERAGING")
    #     print("Read annotation {} file: {}.aparc.a2009s.annot".format(hem, hem))
    #     # filepath = subdir + "/" + subname + "/label/{}.aparc.a2009s.annot".format(hem) # espace natif
    #     filepath = trgt_fsaverage5 + "{}.aparc.a2009s.annot".format(hem)  # espace fsaverage
    #     [labels, ctab, names] = nib.freesurfer.io.read_annot(filepath, orig_ids=False)
    #     print("labels {}".format(hem), labels)
    #     print("Size of labels", labels.shape)
    #     # print("List of names", names)
    #     print("Number of ROIs", len(names))
    #     # ID Regions Extraction
    #     print("ID ROI extraction")
    #     Id_ROI = np.asarray(sorted(set(labels)))
    #     print("Ids ROIs:", Id_ROI)
    #     print("ROIs dimensions:", Id_ROI.shape)
    #     # print("len",len(labels))
    #
    #     # Extract time series for each ROI (averaging)
    #     print('Extract time series for each ROI by averaging operation')
    #     roi_avg = np.empty((len(Id_ROI), gii_matrix.shape[1]))
    #     for i in range(0, len(Id_ROI)):
    #         print("*********************************************************")
    #         print("ID ROI:", Id_ROI[i])
    #         mask = np.where(labels == Id_ROI[i])
    #         roi_timeseries = gii_matrix[mask, :].mean(1)
    #         roi_avg[i, :] = roi_timeseries
    #     print("*********************************************************")
    #     print("********** Results: **********")
    #     print("Size of the Average Matrix of ALL ROIs", roi_avg.shape)
    #     # Save the average matrix of all ROIs
    #     if hem == 'lh':
    #         file = subdir + "/" + subname + "/glm/noisefiltering/roi_avg_lh.npy"
    #         np.save(file, roi_avg)
    #     else:
    #         file = subdir + "/" + subname + "/glm/noisefiltering/roi_avg_rh.npy"
    #         np.save(file, roi_avg)
    #     print("*********************************************************")
    #     end = time.time()
    #     hours, rem = divmod(end - start, 3600)
    #     minutes, seconds = divmod(rem, 60)
    #     print("Elapsed time Step 2: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    #     print("*********************************************************")
    #     start = time.time()
    #
    # """""
    # Step 3 : Correlation Matrix
    # """""
    # print("STEP 3: COMPUTING OF THE CORRELATION MATRIX ")
    # roi_avg_lh=np.load(subdir + "/" + subname + "/glm/noisefiltering/roi_avg_lh.npy")
    # roi_avg_rh = np.load(subdir + "/" + subname + "/glm/noisefiltering/roi_avg_rh.npy")
    # roi_avg=np.concatenate((roi_avg_lh, roi_avg_rh))
    # print("roi avg shape", roi_avg.shape)
    # gii_matrix_lh=np.load(subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_lh.npy")
    # gii_matrix_rh=np.load(subdir + "/" + subname + "/glm/noisefiltering/gii_matrix_rh.npy")
    # gii_matrix=np.concatenate((gii_matrix_lh, gii_matrix_rh))
    # print("gii matrix shape", gii_matrix.shape)
    # correlation_matrix = np.empty((gii_matrix.shape[0], roi_avg.shape[0]))
    # print("correlation matrix shape", correlation_matrix.shape)
    # for n in range(gii_matrix.shape[0]):
    #     for m in range(roi_avg.shape[0]):
    #         correlation_matrix[n, m] = pearsonr(gii_matrix[n, :], roi_avg[m, :])[0]
    # file = subdir + "/" + subname + "/glm/noisefiltering/correlation_matrix.npy"
    # np.save(file, correlation_matrix)
    # print("********** Results: **********")
    # print("Dimensions of the correlation Matrix:", correlation_matrix.shape)
    # print("Computing of the correlation matrix, DONE!:", subname)
    # end = time.time()
    # hours, rem = divmod(end - start, 3600)
    # minutes, seconds = divmod(rem, 60)
    # print("Elapsed time Step 3: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))



def convert_mesh(subdir, sub, hem_list=['lh', 'rh']):
    """
        Convert whit mesh to to surface saved as Gifti file.

        Projection is done for left and right
        :param fs_subdir: FreeSurfer subjects directory
        :param sub: Subject name
        :param nii_file: Splitted .nii directory
        :param gii_dir: Output directory
        :param gii_sfx: Gifti files suffix (add to the hemisphere name)
        :param tgt_subject: Name of target subject
        :param hem_list: Hemispheres (default: left and right)
        :param sfwhm: Surface smoothing (default = 0mm)
    """
    for hem in hem_list:
        fs_subdir = "/hpc/banco/cagna.b/my_intertva/surf/data/" + sub + "/fs/" +sub+ "/surf/{}.white".format(hem)
        gii_subdir = subdir + "/" + sub + "/glm/noisefiltering/{}.white.gii".format(hem)

        cmd = '$FREESURFER_HOME/bin/mris_convert {} {}'.format(fs_subdir, gii_subdir)

        os.system(cmd)


def solve_nan(subdir, sub):
    start = time.time()
    subname=sub
    for ct in range(1, 621):
        filename = subdir + "/" + subname + "/glm/noisefiltering/Res_{:04d}.nii".format(ct)
        print(filename)
        ex = os.path.join(filename)
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
    # elapsed_time_fl = (time.time() - start)
    # print("elapsed time:", elapsed_time_fl)
    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
