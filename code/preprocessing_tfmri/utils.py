import nibabel.gifti as ng
from os import system

# ***** GIFTI ******************************************************************
def gii_convert_to_texture(gii_f, out_gii_f=None, meta=None, verbose=0):
    """ Change intent of first data array and create a new Gifti """
    # TODO: verify the that description is good and function is general
    orig_gii = ng.read(gii_f)

    data = orig_gii.darrays[len(orig_gii.darrays) - 1].data[0]
    darray = ng.GiftiDataArray(data=data, intent='NIFTI_INTENT_ESTIMATE')
    gii = ng.GiftiImage(darrays=[darray])

    if meta:
        gii.meta = ng.GiftiMetaData().from_dict(meta)

    out_f = out_gii_f if out_gii_f else gii_f
    ng.write(gii, out_f)

    if verbose > 0:
        print("Texture saved at: {}".format(out_f))


# ***** Freesurfer *************************************************************
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
        gii_file = "{}/{}.{}.gii".format(gii_dir, filename, hem)

        cmd = '$FREESURFER_HOME/bin/mri_vol2surf --src {} --o {} ' \
              '--out_type gii --regheader {} --hemi {} '\
              '--projfrac-avg 0 1 0.1 --surf-fwhm {:d} --sd {} ' \
              '--trgsubject {}'.format(
               nii_file, gii_file, sub, hem, sfwhm, fs_subdir, tgt_subject)
        system(cmd)

# ***** SHELL ******************************************************************
def run(cmd, verbose=True, execute=True):
    """ Print and run a shell command """
    if verbose:
        print(cmd)

    if execute:
        system(cmd)


# ***** SPM ********************************************************************
def spm_run_batch(batch, variables, display=True, n_jobs=-1):
    """ Run the SPM batch script file in Matlab

        :param variables: Variables set in matlab
        :param batch: SPM Batch file (.m)
        :param display: Open Matlab's GUI if is True. (Default: True)
        :param n_jobs: Set the max number of jobs in Matlab (default: maximum)
        :return: Nothing
    """

    m_script = matlab_define(variables)
    if n_jobs > -1:
        m_script += ' LASTN = maxNumCompThreads({:d});'.format(n_jobs)
    m_script += "spm fmri; run('{}');".format(batch)
    m_script += "spm_jobman('run',matlabbatch); clear variables;"

    matlab_run_script(m_script, display=display, exit_after=True)


# ***** MATLAB *****************************************************************

def run_script_matlab(script, var, display=True ):
    m_script = matlab_define(var)
    m_script += "run('{}');".format(script)
    matlab_run_script(m_script, display=display, exit_after=True)

def matlab_run_script(mat_script, exit_after=True, display=True):
    """ Start matlab throught the terminal and run the given script

        :param mat_script: Matlab script.
        :param exit_after: (opt.) Add the matlab exit command at the end of the
                            script. Default: False.
        :param display: Open Matlab's GUI if is True. (Default: True)
        :return: Nothing
    """
    if exit_after is True:
        mat_script += " exit();"

    cmd = "matlab "
    if display is False:
        cmd += "-nodisplay "
    cmd += '-r "{}"'.format(mat_script)
    run(cmd)


def matlab_val(val):
    """ Return val formatted for Matlab """
    if type(val) == str:
        var = "'" + val + "'"
    elif type(val) == dict:
        var = matlab_format_dict(val)
    elif type(val) == bool:
        var = int(val)
    elif type(val) == list:
        if len(val) == 0:
            var = '[]'
        elif type(val[0]) == str or type(val[0]) == list:
            var = matlab_format_str_array(val)
        else:
            var = matlab_format_array(val)
    else:
        var = val
    return var


def matlab_format_array(array):
    """ Return Matlab array that define an array """
    m_tab = "["
    for val in array:
        m_tab += "{} ".format(matlab_val(val))
    if m_tab[:-2] == ", ":
        m_tab = m_tab[:-2]
    m_tab += "]"
    return m_tab


def matlab_format_str_array(array):
    """ Return Matlab script that define an array of strings """
    m_tab = "{"
    for val in array:
        m_tab += "{}, ".format(matlab_val(val))
    if m_tab[:-2] == ", ":
        m_tab = m_tab[:-2]
    m_tab += "}"
    return m_tab


def matlab_format_dict(data):
    """ Return Matlab script that define a dictionnary as cells """
    m_dict = "{"
    for k in data.keys():
        m_dict += "struct('key', '{}', 'val', {}), ".format(
            k, matlab_format_array(data[k]))
    m_dict += "}"
    return m_dict


def matlab_define(variables):
    """ Return Matlab script that define all variables

    :param variables: dictionnary of variables (names as keys)
    """
    m_script = ""
    for v_name in variables.keys():
        m_script += "{}={}; ".format(v_name, matlab_val(variables[v_name]))
    return m_script





