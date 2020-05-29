
"""""""""""""""
       PCA  code
"""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from keras.models import load_model
# import numpy as np
# import keras
# from keras.layers import Input, Dense, concatenate
# from keras.models import Model
# from keras import backend as k
# from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
plt.switch_backend('agg')
import sys as os
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

# loading multimodal fMRI data
def load_data(sub, view):

    # Import Task fMRI data
    if view == 1:
        view_tfmri = np.load(os.path.join(path, "tfmri/{}/gii_matrix_fsaverage5.npy".format(sub)))
        return view_tfmri

    # Import Resting-State fMRI data
    if view == 2:
        view_rsfmri = np.load(os.path.join(path, "rsfmri/{}/correlation_matrix_fsaverage5.npy".format(sub)))
        return view_rsfmri

    # Import concatenated fMRI data
    if view ==3:
        view_rsfmri = np.load(os.path.join(path, "rsfmri/{}/correlation_matrix_fsaverage5.npy".format(sub)))
        view_tfmri = np.load(os.path.join(path, "tfmri/{}/gii_matrix_fsaverage5.npy".format(sub)))
        fmri_data =np.concatenate([view_tfmri, view_rsfmri], axis=1)
        return fmri_data

# normalization to range [-1, 1]

def normalization(data):
    normalized_data= 2* (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
    return normalized_data

# Path
path = "/home/asellami/data_fsaverage5"

missing_data=[36]
index_subjects=np.arange(3,43)
index_subjects = np.delete(index_subjects, np.argwhere(index_subjects == missing_data))

view=2
if view == 1:
    v = 'tfmri'
elif view == 2:
    v = 'rsfmri'
else:
    v = 'concat'
# MSE
mse_train = []
mse_test = []
# RMSE
rmse_train = []
rmse_test = []
#
# Standard deviation MSE
std_mse_train = []
std_mse_test = []
# Standard deviation RMSE
std_rmse_train = []
std_rmse_test = []
ar = np.arange(2,101)
#dimensions = ar[ar%2==0]
dimensions=[2, 6, 10, 16, 20, 26, 30, 36, 40, 46, 50, 56, 60, 66, 70, 76, 80, 86, 90, 96, 100]

for dim in dimensions:
    # Cross Validation
    kf = KFold(n_splits=10)
    print(kf.get_n_splits(index_subjects))
    print("number of splits:", kf)
    print("number of features:", dimensions)
    cvscores_mse_test = []
    cvscores_rmse_test = []
    cvscores_mse_train = []
    cvscores_rmse_train = []
    fold=0
    for train_index, test_index in kf.split(index_subjects):
        fold+=1
        print(f"Fold #{fold}")
        print("TRAIN:", index_subjects[train_index], "TEST:", index_subjects[test_index])
        # load training and testing data
        print('Load training data... (view {})'.format(view))
        train_data = np.concatenate([load_data(sub, view) for sub in index_subjects[train_index]])
        print("Shape of the training data:", train_data.shape)
        print('Load testdata... (view {})'.format(view))
        test_data = np.concatenate([load_data(sub, view) for sub in index_subjects[test_index]])
        print("Shape of the test data:", test_data.shape)
        # Data normalization to range [-1, 1]
        # print("Data normalization to range [-1, 1]")
        scaler = MinMaxScaler()
        normalized_train_data = scaler.fit_transform(train_data)
        normalized_test_data = scaler.fit_transform(test_data)
        # intialize pca
        pca = PCA(n_components=dim)

        # fit PCA on training set
        pca.fit(normalized_train_data)

        # Apply the mapping (transform) to both the training set and the test set
        X_train_pca = pca.transform(normalized_train_data)
        X_test_pca = pca.transform(normalized_test_data)
        print("Original shape:   ", normalized_train_data.shape)
        print("Transformed shape:", X_train_pca.shape)

        # Reconstruction of training data
        print("Reconstruction of training data... ")
        X_train_new = pca.inverse_transform(X_train_pca)
        print("Reconstructed matrix shape:", X_train_new.shape)
        mse = mean_squared_error(normalized_train_data, X_train_new)
        print('Reconstruction MSE : ', mse)
        cvscores_mse_train.append(mse)
        rms = sqrt(mse)
        print('Reconstruction RMSE : ', rms)
        cvscores_rmse_train.append(rms)
        # Reconstruction of test data
        print("Reconstruction of test data... ")
        X_test_new = pca.inverse_transform(X_test_pca)
        print("Reconstructed matrix shape:", X_test_new.shape)
        mse = mean_squared_error(normalized_test_data, X_test_new)
        cvscores_mse_test.append(mse)
        print('Reconstruction MSE : ', mse)
        rms = sqrt(mse)
        print('Reconstruction RMSE : ', rms)
        cvscores_rmse_test.append(rms)

        # Apply dimensionality reduction
        directory = '../../../regression/pca/{}/{}/fold_{}/'.format(v, dim, fold)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for sub in index_subjects:
            subject=load_data(sub, view)
            normalized_subject = scaler.fit_transform(subject)
            transformed_subject = pca.transform(normalized_subject)
            file = directory + "X_{}.npy".format(sub)
            np.save(file, transformed_subject)
            print('Shape of Latent representation:', transformed_subject.shape)
            print('Transpose of latent representation', transformed_subject.T.shape)
    print("shape of vector mse train", np.array([cvscores_mse_train]).shape)
    print(cvscores_mse_train)
    np.save('cvscores_mse_train_pca_dim_{}.npy'.format(dim), np.array([cvscores_mse_train]))

    print("shape of vector mse test", np.array([cvscores_mse_test]).shape)
    print(cvscores_mse_test)
    np.save( 'cvscores_mse_test_pca_dim_{}.npy'.format(dim), np.array([cvscores_mse_train]))

    print("shape of vector rmse train", np.array([cvscores_rmse_train]).shape)
    print(cvscores_rmse_train)
    np.save( 'cvscores_mse_train_pca_dim_{}.npy'.format(dim), np.array([cvscores_rmse_train]))

    print("shape of vector rmse test", np.array([cvscores_rmse_test]).shape)
    print(cvscores_rmse_test)
    np.save( 'rmse_test_pca_dim_{}.npy'.format(dim), np.array([cvscores_rmse_test]))
    print("%.2f%% (+/- %.5f%%)" % (np.mean(cvscores_mse_test), np.std(cvscores_mse_test)))
    mse_train.append(np.mean(cvscores_mse_train))
    mse_test.append(np.mean(cvscores_mse_test))
    rmse_train.append(np.mean(cvscores_rmse_train))
    rmse_test.append(np.mean(cvscores_rmse_test))
    std_mse_train.append(np.std(cvscores_mse_train))
    std_mse_test.append(np.std(cvscores_mse_test))
    std_rmse_train.append(np.std(cvscores_rmse_train))
    std_rmse_test.append(np.std(cvscores_rmse_test))
np.save( 'mse_test_mean_pca.npy', np.array([mse_test]))
np.save( 'rmse_test_mean_pca.npy', np.array([rmse_test]))
np.save( 'std_mse_mean_pca.npy', np.array([std_mse_test]))
np.save( 'std_rmse_mean_pca.npy', np.array([std_rmse_test]))
# plotting the mse train
# setting x and y axis range
#plt.xlim(1, 120)
plt.plot(dimensions, mse_train, label="mse_train")
plt.plot(dimensions, mse_test, label="mse_test")
plt.xlabel('Encoding dimension')
plt.ylabel('Reconstruction error (MSE)')
# showing legend
plt.legend()
plt.savefig('reconstruction_error_mse_pca_tfmri.pdf')
plt.savefig('reconstruction_error_mse_pca_tfmri.png')
plt.close()

# plotting the rmse train
# setting x and y axis range
#plt.xlim(1, 120)
plt.plot(dimensions, rmse_train, label="rmse_train")
plt.plot(dimensions, rmse_test, label="rmse_test")
plt.xlabel('Encoding dimension')
plt.ylabel('Reconstruction error (RMSE)')
# showing legend
plt.legend()
plt.savefig('reconstruction_error_rmse_pca_tfmri.pdf')
plt.savefig('reconstruction_error_rmse_pca_tfmri.png')
plt.close()

