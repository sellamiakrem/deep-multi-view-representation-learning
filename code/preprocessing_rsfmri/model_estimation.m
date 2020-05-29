
%% Model Estimation
job_id=1;
spmat=[subdir, '/', subname, '/glm/noisefiltering/SPM.mat'];
matlabbatch{job_id}.spm.stats.fmri_est.spmmat = {spmat};
matlabbatch{job_id}.spm.stats.fmri_est.write_residuals = 1;
matlabbatch{job_id}.spm.stats.fmri_est.method.Classical = 1;