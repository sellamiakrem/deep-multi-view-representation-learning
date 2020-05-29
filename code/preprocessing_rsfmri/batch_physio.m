
tr = 0.955;
n_slices = 60;
%subdir='/hpc/banco/sellami.a/InterTVA/rsfmri'
%subname='sub-06'
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

%% Physio TAPAS
matlabbatch{job_id}.spm.tools.physio.save_dir = {[subdirectory, '/func']};
matlabbatch{job_id}.spm.tools.physio.log_files.vendor = 'Siemens';
matlabbatch{job_id}.spm.tools.physio.log_files.cardiac = '';
matlabbatch{job_id}.spm.tools.physio.log_files.respiration = '';
matlabbatch{job_id}.spm.tools.physio.log_files.scan_timing = {''};
matlabbatch{job_id}.spm.tools.physio.log_files.sampling_interval = [0.005 0.02];
matlabbatch{job_id}.spm.tools.physio.log_files.relative_start_acquisition = 0;
matlabbatch{job_id}.spm.tools.physio.log_files.align_scan = 'last';
matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.Nslices = 60;
matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.NslicesPerBeat = [];
matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.TR = 0.955;
matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.Ndummies = 0;
matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.Nscans = 620;
matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.onset_slice = 10;
matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.time_slice_to_slice = 0;
matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.Nprep = [];
matlabbatch{job_id}.spm.tools.physio.scan_timing.sync.scan_timing_log = struct([]);
matlabbatch{job_id}.spm.tools.physio.preproc.cardiac.modality = 'PPU';
matlabbatch{job_id}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.3;
matlabbatch{job_id}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = '';
matlabbatch{job_id}.spm.tools.physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
matlabbatch{job_id}.spm.tools.physio.model.output_multiple_regressors = [subname, '_task-rest_noise.txt'];
matlabbatch{job_id}.spm.tools.physio.model.output_physio = [subname, '_task-rest_noise.mat'];
matlabbatch{job_id}.spm.tools.physio.model.orthogonalise = 'none';
matlabbatch{job_id}.spm.tools.physio.model.retroicor.no = struct([]);
matlabbatch{job_id}.spm.tools.physio.model.rvt.no = struct([]);
matlabbatch{job_id}.spm.tools.physio.model.hrv.no = struct([]);
matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.fmri_files = {realigned_bold};
matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.roi_files = { 
    c2_img,
    c3_img
};
matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.thresholds = 0.99;
matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.n_voxel_crop = [1 0];
matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.n_components = n_comp;
matlabbatch{job_id}.spm.tools.physio.model.movement.yes.file_realignment_parameters = {rp_file};
matlabbatch{job_id}.spm.tools.physio.model.movement.yes.order = 6;
matlabbatch{job_id}.spm.tools.physio.model.movement.yes.outlier_translation_mm = Inf;
matlabbatch{job_id}.spm.tools.physio.model.movement.yes.outlier_rotation_deg = Inf;
matlabbatch{job_id}.spm.tools.physio.model.other.no = struct([]);
matlabbatch{job_id}.spm.tools.physio.verbose.level = 2;
matlabbatch{job_id}.spm.tools.physio.verbose.fig_output_file = [subname, '_task-rest_physio.fig'];
matlabbatch{job_id}.spm.tools.physio.verbose.use_tabs = false;
job_id = job_id + 1;
