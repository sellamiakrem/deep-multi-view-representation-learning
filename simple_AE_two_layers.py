
"""""""""""""""
Monomodal Autoenconder  code
"""""""""""""""
import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys as os
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.optimizers import SGD, Adadelta, Adam
import numpy as np
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from keras.models import load_model




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





# Path
path = "/home/asellami/data_fsaverage5"

print('View 1: task-fMRI')
print('View 2: resting-state fMRI')
print('View=3: concatenated views (task-fMRI + rest-fMRI)')

# view =1: tfmri, view =2: rsfmri, view=3: concatenated views (task-fMRI + rest-fMRI)
view=1

# activation functions
hidden_layer='linear'
output_layer='linear'

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
# missing data
missing_data=[36]
index_subjects=np.arange(3,43)
index_subjects = np.delete(index_subjects, np.argwhere(index_subjects == missing_data))

#ar = np.arange(75,101)
#dimensions = ar[ar%2==0]

dimensions=[2, 6, 10, 16, 20, 26, 30, 36, 40, 46, 50, 56, 60, 66, 70, 76, 80, 86, 90, 96, 100]
batch_1=dimensions[0:6]
batch_2=dimensions[6:12]
batch_3=dimensions[12:17]
batch_4=dimensions[17:21]
for dim in batch_1:
    # create directory
    directory = '{}'.format(dim)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Cross Validation
    kf = KFold(n_splits=10)
    print(kf.get_n_splits(index_subjects))
    print("number of splits:", kf)
    print("number of features:", dimensions)
    cvscores_mse_test = []
    cvscores_rmse_test = []
    cvscores_mse_train = []
    cvscores_rmse_train = []
    fold = 0
    for train_index, test_index in kf.split(index_subjects):
        fold += 1
        # create directory
        directory = '{}/fold_{}'.format(dim, fold)
        if not os.path.exists(directory):
            os.makedirs(directory)
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
        print("Data normalization to range [0, 1]")
        scaler = MinMaxScaler()
        normalized_train_data = scaler.fit_transform(train_data)
        normalized_test_data = scaler.fit_transform(test_data)

        # Apply linear autoencoder
        # Inputs Shape
        input_train_data = Input(shape=(normalized_train_data[0].shape))
        # Create linear AE
        encoded = Dense(110, activation=hidden_layer)(input_train_data)
        encoded = Dense(dim, activation=hidden_layer)(encoded)
        decoded = Dense(110, activation=hidden_layer)(encoded)
        decoded = Dense(normalized_train_data[0].shape[0], activation=output_layer)(decoded)
        # This model maps an input to its reconstruction
        autoencoder = Model(input_train_data, decoded)
        adam= Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        autoencoder.compile(optimizer=adam, loss='mse')
        print(autoencoder.summary())
        # fit Autoencoder on training set
        history = autoencoder.fit(normalized_train_data, normalized_train_data,
                                  epochs=70,
                                  batch_size=1000,
                                  shuffle=True,
                                  validation_data=(normalized_test_data, normalized_test_data), verbose=1)
        # list all data in history
        print(history.history.keys())
        # use our encoded layer to encode the training input
        encoder = Model(input_train_data, encoded)
        #  create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer_1 = autoencoder.layers[-2]
        decoder_layer_2 = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer_2(decoder_layer_1(encoded_input)))

        # save models
        autoencoder.save('{}/fold_{}/autoencoder.h5'.format(dim, fold))
        encoder.save('{}/fold_{}/encoder.h5'.format(dim, fold))
        decoder.save('{}/fold_{}/decoder.h5'.format(dim, fold))

        # plot our loss
        plt.plot(history.history['loss'], label='loss_fold_{}'.format(fold))
        plt.plot(history.history['val_loss'], label='val_loss_fold_{}'.format(fold))
        print("vector of val_loss", history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('{}/fold_{}/loss.png'.format(dim, fold))
        plt.savefig('{}/fold_{}/loss.pdf'.format(dim, fold))
        plt.close()

        # Apply the mapping (transform) to both the training set and the test set
        X_train_AE = encoder.predict(normalized_train_data)
        X_test_AE = encoder.predict(normalized_test_data)

        print("Original shape:   ", normalized_train_data.shape)
        print("Transformed shape:", X_train_AE.shape)

        # Reconstruction of training data
        print("Reconstruction of training data... ")
        X_train_new = autoencoder.predict(normalized_train_data)
        print("Max value of predicted training data", np.max(X_train_new))
        print("Min value of predicted training data", np.min(X_train_new))
        print("Reconstructed matrix shape:", X_train_new.shape)
        val_mse_train = mean_squared_error(normalized_train_data, X_train_new)
        print('Value of MSE (train) : ', val_mse_train)
        cvscores_mse_train.append(val_mse_train)
        val_rmse = sqrt(val_mse_train)
        print('Value of RMSE (train)  : ', val_rmse)
        cvscores_rmse_train.append(val_rmse)
        # Reconstruction of test data
        print("Reconstruction of test data... ")
        X_test_new = autoencoder.predict(normalized_test_data)
        print("Max value of predicted test data", np.max(X_test_new))
        print("Min value of predicted test data", np.min(X_test_new))
        print("Reconstructed matrix shape:", X_test_new.shape)
        val_mse = mean_squared_error(normalized_test_data, X_test_new)
        cvscores_mse_test.append(val_mse)
        print('Value of MSE (test) : ', val_mse)
        val_rmse = sqrt(val_mse)
        print('Value of MSE (test) : ', val_rmse)
        cvscores_rmse_test.append(val_rmse)
        # transform and save latent representation


    print("shape of  mse vector (train):", np.array([cvscores_mse_train]).shape)
    print(cvscores_mse_train)
    np.save('{}/cvscores_mse_train.npy'.format(dim), np.array([cvscores_mse_train]))
    print("shape of  mse vector(test):", np.array([cvscores_mse_test]).shape)
    print(cvscores_mse_test)
    np.save('{}/cvscores_mse_test.npy'.format(dim), np.array([cvscores_mse_test]))
    print("shape of rmse vector (train):", np.array([cvscores_rmse_train]).shape)
    print(cvscores_rmse_train)
    np.save('{}/cvscores_rmse_train.npy'.format(dim), np.array([cvscores_rmse_train]))
    print("shape of rmse vector (test):", np.array([cvscores_rmse_test]).shape)
    print(cvscores_rmse_test)
    np.save('{}/cvscores_rmse_test.npy'.format(dim), np.array([cvscores_rmse_test]))
    print("%.3f%% (+/- %.5f%%)" % (np.mean(cvscores_mse_test), np.std(cvscores_mse_test)))
    mse_train.append(np.mean(cvscores_mse_train))
    std_mse_train.append(np.std(cvscores_mse_train))
    mse_test.append(np.mean(cvscores_mse_test))
    std_mse_test.append(np.std(cvscores_mse_test))
    rmse_train.append(np.mean(cvscores_rmse_train))
    std_rmse_train.append(np.std(cvscores_rmse_train))
    rmse_test.append(np.mean(cvscores_rmse_test))
    std_rmse_test.append(np.std(cvscores_rmse_test))

# save MSE, RMSE, and STD vectors for training and test sets
np.save('mse_train_mean.npy', np.array([mse_train]))
np.save('rmse_train_mean.npy', np.array([rmse_train]))
np.save('std_mse_train_mean.npy', np.array([std_mse_train]))
np.save('std_rmse_train_mean.npy', np.array([std_rmse_train]))

np.save('mse_test_mean.npy', np.array([mse_test]))
np.save('rmse_test_mean.npy', np.array([rmse_test]))
np.save('std_mse_test_mean.npy', np.array([std_mse_test]))
np.save('std_rmse_test_mean.npy', np.array([std_rmse_test]))
# plotting the mse train
# setting x and y axis range
# plotting the mse train
plt.plot(dimensions, mse_train, label="mse_train")
plt.plot(dimensions, mse_test, label="mse_test")
plt.xlabel('Encoding dimension')
plt.ylabel('Reconstruction error (MSE)')
# showing legend
plt.legend()
plt.savefig('reconstruction_error_mse.pdf')
plt.savefig('reconstruction_error_mse.png')
plt.close()
# plotting the rmse train
# setting x and y axis range
plt.plot(dimensions, rmse_train, label="rmse_train")
plt.plot(dimensions, rmse_test, label="rmse_test")
plt.xlabel('Encoding dimension')
plt.ylabel('Reconstruction error (RMSE)')
# showing legend
plt.legend()
plt.savefig('reconstruction_error_rmse.pdf')
plt.savefig('reconstruction_error_rmse.png')
plt.close()

