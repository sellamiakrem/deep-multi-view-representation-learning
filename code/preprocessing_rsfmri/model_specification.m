

tr = 0.955;
n_slices = 60;
subdirectory = [subdir, '/', subname];
fmap_file = [subdirectory, '/fmap/', subname, '_acq-topup2_fieldmap.nii,1'];
mag_fmap_file = [subdirectory, '/fmap/', subname, '_acq-topup2_magnitude.nii,1'];
epi_ref = [subdirectory, '/func/a', subname, '_task-rest_bold.nii,1'];

T1_ref = [subdirectory, '/anat/masked_msanlm_', subname, '_T1w.nii,1'];


bold_img = [subdirectory, '/func/', subname, '_task-rest_bold.nii'];
realigned_bold = [subdirectory, '/func/ua', subname, '_task-rest_bold.nii'];
coreg_ref=[subdirectory, '/func/localizer/vol/u', subname, '_task-localizer_bold.nii,1'];

c2_img = [subdirectory, '/anat/c2sanlm_', subname, '_T1w.nii,1'];
c3_img = [subdirectory, '/anat/c3sanlm_', subname, '_T1w.nii,1'];

rp_file = [subdirectory, '/func/rest/rp_a', subname, '_task-rest_bold.txt'];



n_comp = 12;

% % ------------------------------------------------------------------------
job_id = 1;
% % 


%% Model Specification (GLM)
%% Make Directory
f_parent=[subdirectory,'/glm'];
matlabbatch{job_id}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = {f_parent};
matlabbatch{job_id}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'noisefiltering';
job_id = job_id + 1;
%% Model Specification: GLM
spec_dir=[subdirectory ,'/glm/noisefiltering'];
scan=[subdirectory, '/func/ua', subname ,'_task-rest_bold.nii'];
reg=[subdirectory ,'/func/', subname ,'_task-rest_noise.txt'];
fmask=[subdirectory, '/func/ua', subname ,'_task-rest_bold_fullimage_mask.nii'];
matlabbatch{job_id}.spm.stats.fmri_spec.dir = {spec_dir};
matlabbatch{job_id}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{job_id}.spm.stats.fmri_spec.timing.RT = 0.955;
matlabbatch{job_id}.spm.stats.fmri_spec.timing.fmri_t = 12;
matlabbatch{job_id}.spm.stats.fmri_spec.timing.fmri_t0 = 6;
matlabbatch{job_id}.spm.stats.fmri_spec.sess.scans = {scan};
matlabbatch{job_id}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
matlabbatch{job_id}.spm.stats.fmri_spec.sess.multi = {''};
matlabbatch{job_id}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
matlabbatch{job_id}.spm.stats.fmri_spec.sess.multi_reg = {reg};
matlabbatch{job_id}.spm.stats.fmri_spec.sess.hpf = 128;
matlabbatch{job_id}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{job_id}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{job_id}.spm.stats.fmri_spec.volt = 1;
matlabbatch{job_id}.spm.stats.fmri_spec.global = 'None';
matlabbatch{job_id}.spm.stats.fmri_spec.mthresh = 0;
matlabbatch{job_id}.spm.stats.fmri_spec.mask = {fmask};
matlabbatch{job_id}.spm.stats.fmri_spec.cvi = 'AR(1)';
job_id = job_id + 1;

