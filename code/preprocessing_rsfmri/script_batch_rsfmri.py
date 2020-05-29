"""
This script demonstrates how to use firoul cluster to perform
a set of preprocessing task on resting-state fMRI data.
39 subjects are available. You can download these multimodal fMRI data here:

https://openneuro.org/datasets/ds001771/versions/1.0.2

"""
import os.path as op
import subprocess
import datetime as dt


root_dir = op.realpath(op.dirname(__file__))
subdir = '/hpc/banco/sellami.a/InterTVA/rsfmri'


for s in range(3,43): # subnames index
    if s == 36:  # Avoid missing data
        sub = "sub-{0:02d}".format(s)
        cmd = "/hpc/soft/anaconda3/bin/python " + root_dir + "/preprocessing_rsfmri.py {0}".format(sub)
        log_dir = op.join(subdir, sub, 'logs')
        std = log_dir + "/{0}_{1}_%jobid%_preprocessing.".format(sub, d_str)
        fb_cmd = "mkdir {0} -p; frioul_batch \"{1}\" -C {2}cmd -O {3}out -E {4}err -n 18,19,20 ".format(log_dir, cmd, std, std,
                                                                                       std)
        print(fb_cmd)
        subprocess.run(fb_cmd, shell=True)
