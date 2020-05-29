
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
%% Expand 4D resting bold file
matlabbatch{job_id}.spm.util.exp_frames.files = {bold_img};
matlabbatch{job_id}.spm.util.exp_frames.frames = 1:999;
job_id = job_id + 1;


%% Slice Timing
matlabbatch{job_id}.spm.temporal.st.scans{1}(1) = cfg_dep('Expand image frames: Expanded filename list.', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','files'));
matlabbatch{job_id}.spm.temporal.st.nslices = n_slices;
matlabbatch{job_id}.spm.temporal.st.tr = tr;
matlabbatch{job_id}.spm.temporal.st.ta = tr - (tr/n_slices); %0.939083333333333;
matlabbatch{job_id}.spm.temporal.st.so = [875 795 717.5 637.5 557.5 477.5 397.5 317.5 240 160 80 0 875 795 717.5 637.5 557.5 477.5 397.5 317.5 240 160 80 0 875 795 717.5 637.5 557.5 477.5 397.5 317.5 240 160 80 0 875 795 717.5 637.5 557.5 477.5 397.5 317.5 240 160 80 0 875 795 717.5 637.5 557.5 477.5 397.5 317.5 240 160 80 0];
matlabbatch{job_id}.spm.temporal.st.refslice = 437.5;
matlabbatch{job_id}.spm.temporal.st.prefix = 'a';
job_id = job_id + 1;
%% Compute VDM
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.data.precalcfieldmap.precalcfieldmap = {fmap_file};
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.data.precalcfieldmap.magfieldmap = {mag_fmap_file};
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.et = [4.37 6.83];
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.maskbrain = 1;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.blipdir = -1;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.tert = 58;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.epifm = 0;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.ajm = 0;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.uflags.method = 'Mark3D';
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.uflags.fwhm = 10;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.uflags.pad = 0;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.uflags.ws = 1;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.mflags.template = {
    '/hpc/soft/matlab_tools/spm/spm12/toolbox/FieldMap/T1.nii'
};
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.mflags.fwhm = 4;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.mflags.nerode = 2;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.mflags.ndilate = 4;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.mflags.thresh = 0.5;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.defaults.defaultsval.mflags.reg = 0.02;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.session.epi = {epi_ref};
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.matchvdm = 0;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.sessname = 'session';
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.writeunwarped = 0;
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.anat = {T1_ref};
matlabbatch{job_id}.spm.tools.fieldmap.calculatevdm.subj.matchanat = 0;
job_id = job_id + 1;

% 
 %% Realing & Unwarp
matlabbatch{job_id}.spm.spatial.realignunwarp.data.scans(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 1)', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
matlabbatch{job_id}.spm.spatial.realignunwarp.data.pmscan(1) = {[subdirectory, '/fmap/vdm5_', subname ,'_acq-topup2_fieldmap.nii']};
matlabbatch{job_id}.spm.spatial.realignunwarp.eoptions.quality = 0.9;
matlabbatch{job_id}.spm.spatial.realignunwarp.eoptions.sep = 4;
matlabbatch{job_id}.spm.spatial.realignunwarp.eoptions.fwhm = 5;
matlabbatch{job_id}.spm.spatial.realignunwarp.eoptions.rtm = 0;
matlabbatch{job_id}.spm.spatial.realignunwarp.eoptions.einterp = 2;
matlabbatch{job_id}.spm.spatial.realignunwarp.eoptions.ewrap = [0 0 0];
matlabbatch{job_id}.spm.spatial.realignunwarp.eoptions.weight = '';
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.basfcn = [12 12];
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.regorder = 1;
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.lambda = 100000;
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.jm = 0;
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.fot = [4 5];
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.sot = [];
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.uwfwhm = 4;
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.rem = 1;
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.noi = 5;
matlabbatch{job_id}.spm.spatial.realignunwarp.uweoptions.expround = 'Average';
matlabbatch{job_id}.spm.spatial.realignunwarp.uwroptions.uwwhich = [2 1];
matlabbatch{job_id}.spm.spatial.realignunwarp.uwroptions.rinterp = 4;
matlabbatch{job_id}.spm.spatial.realignunwarp.uwroptions.wrap = [0 0 0];
matlabbatch{job_id}.spm.spatial.realignunwarp.uwroptions.mask = 1;
matlabbatch{job_id}.spm.spatial.realignunwarp.uwroptions.prefix = 'u';
job_id = job_id + 1;


%% Coregistration to already coregistred Voice Localizer BOLD image
matlabbatch{job_id}.spm.spatial.coreg.estimate.ref = {coreg_ref};
matlabbatch{job_id}.spm.spatial.coreg.estimate.source = {[realigned_bold, ',1']};
matlabbatch{job_id}.spm.spatial.coreg.estimate.other = {realigned_bold};
matlabbatch{job_id}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
matlabbatch{job_id}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
matlabbatch{job_id}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{job_id}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
job_id = job_id + 1;


%% Physio TAPAS
% matlabbatch{job_id}.spm.tools.physio.save_dir = {[subdirectory, '/func']};
% matlabbatch{job_id}.spm.tools.physio.log_files.vendor = 'Siemens';
% matlabbatch{job_id}.spm.tools.physio.log_files.cardiac = '';
% matlabbatch{job_id}.spm.tools.physio.log_files.respiration = '';
% matlabbatch{job_id}.spm.tools.physio.log_files.scan_timing = {''};
% matlabbatch{job_id}.spm.tools.physio.log_files.sampling_interval = [0.005 0.02];
% matlabbatch{job_id}.spm.tools.physio.log_files.relative_start_acquisition = 0;
% matlabbatch{job_id}.spm.tools.physio.log_files.align_scan = 'last';
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.Nslices = 60;
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.NslicesPerBeat = [];
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.TR = 0.955;
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.Ndummies = 0;
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.Nscans = 620;
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.onset_slice = 10;
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.time_slice_to_slice = 0;
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sqpar.Nprep = [];
% matlabbatch{job_id}.spm.tools.physio.scan_timing.sync.scan_timing_log = struct([]);
% matlabbatch{job_id}.spm.tools.physio.preproc.cardiac.modality = 'PPU';
% matlabbatch{job_id}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.3;
% matlabbatch{job_id}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = '';
% matlabbatch{job_id}.spm.tools.physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
% matlabbatch{job_id}.spm.tools.physio.model.output_multiple_regressors = [subname, '_task-rest_noise.txt'];
% matlabbatch{job_id}.spm.tools.physio.model.output_physio = [subname, '_task-rest_noise.mat'];
% matlabbatch{job_id}.spm.tools.physio.model.orthogonalise = 'none';
% matlabbatch{job_id}.spm.tools.physio.model.retroicor.no = struct([]);
% matlabbatch{job_id}.spm.tools.physio.model.rvt.no = struct([]);
% matlabbatch{job_id}.spm.tools.physio.model.hrv.no = struct([]);
% matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.fmri_files = {realigned_bold};
% matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.roi_files = { 
%     c2_img,
%     c3_img
% };
% matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.thresholds = 0.99;
% matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.n_voxel_crop = [1 0];
% matlabbatch{job_id}.spm.tools.physio.model.noise_rois.yes.n_components = n_comp;
% matlabbatch{job_id}.spm.tools.physio.model.movement.yes.file_realignment_parameters = {rp_file};
% matlabbatch{job_id}.spm.tools.physio.model.movement.yes.order = 6;
% matlabbatch{job_id}.spm.tools.physio.model.movement.yes.outlier_translation_mm = Inf;
% matlabbatch{job_id}.spm.tools.physio.model.movement.yes.outlier_rotation_deg = Inf;
% matlabbatch{job_id}.spm.tools.physio.model.other.no = struct([]);
% matlabbatch{job_id}.spm.tools.physio.verbose.level = 2;
% matlabbatch{job_id}.spm.tools.physio.verbose.fig_output_file = [subname, '_task-rest_physio.fig'];
% matlabbatch{job_id}.spm.tools.physio.verbose.use_tabs = false;
% job_id = job_id + 1;
