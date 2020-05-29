
subdirectory = [subdir, '/', subname, '/glm/noisefiltering/SPM.mat' ];
load(subdirectory)
SPM.xM.TH = -Inf(size(SPM.xM.TH));
save(subdirectory,'SPM', '-v6');