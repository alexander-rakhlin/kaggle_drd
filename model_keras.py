from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.constraints import nonneg
from keras.regularizers import l1, l1l2
from keras.optimizers import SGD

from sklearn.preprocessing import LabelEncoder
from kappa import quadratic_weighted_kappa
from sklearn.metrics import accuracy_score, confusion_matrix

import os
import constants
from scaler import variance, scale, fit2distribution
import h5py
from utils import dump_prediction


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def pick_activations(model, X, net_type):
    input_dim = model.get_config()[0]['input_dim']
    output_dim = model.get_config()[0]['output_dim']
    dropout = model.get_config()[2]['p']
    m = Sequential()
    m.add(Dense(input_dim, output_dim, weights=model.layers[0].get_weights()))
    m.add(Activation('relu'))
    m.add(Dropout(dropout))

    if net_type == 'softmax':
        loss='categorical_crossentropy'
    elif net_type == 'regression':
        loss='mean_squared_error'
    m.compile(loss=loss, optimizer='adam')

    activations = m.predict(X, verbose=0)
    
    return activations
    

def validate(n_epochs, n_models, n_steps=5, activations=False):
    with h5py.File(constants.train_features_scaled_strat_file, "r") as fi:
        labels_train = fi.get("y_train")[:60000]
        X_train = fi.get("X_train")[:60000]
        y_train, _ = preprocess_labels(labels_train, categorical=(net_type=='softmax'))
        
        labels_test = fi.get("y_test")[()]
        X_test = fi.get("X_test")[()]
        y_test, _ = preprocess_labels(labels_test, categorical=(net_type=='softmax'))
    
        y_train = y_train/5.0/2+0.5
        y_test = y_test/5.0/2+0.5
    
        if net_type == 'softmax':
            n_classes = y_train.shape[1]
        elif net_type == 'regression':
            n_classes = 1
        print(n_classes, 'classes')
        
        n_dims = X_train.shape[1]
        print(n_dims, 'dims')
    
        cum_blend = 0
        models = range(1, n_models+1)
        for i in models:
            print("\n-------------- Model %d --------------\n" % i)
        
            model = model_factory(n_classes, n_dims, net_type)
            for n in range(0, n_epochs, n_steps):
                model.fit(X_train, y_train, nb_epoch=n_steps, batch_size=128,
                          show_accuracy=False, verbose=2)#, validation_data=(X_test, y_test))
    
                # validate individual net
                if net_type == 'softmax':
                    y_pred = model.predict_classes(X_test, verbose=0)
                elif net_type == 'regression':        
                    y_pred = model.predict(X_test, verbose=0)
                    y_pred = np.floor((y_pred-0.5)*2*5.0).flatten()
                    y_pred[y_pred<0] = 0
                    y_pred[y_pred>4] = 4
    
                print('Epoch: %d. Accuracy: %0.2f%%. Kappa: %0.2f' %
                (n+n_steps,
                 100 * accuracy_score(labels_test, y_pred),
                 quadratic_weighted_kappa(labels_test, y_pred)))
            
    
            # validate ensemble
            if net_type == 'softmax':
                cum_blend += model.predict_proba(X_test, verbose=0)
                y_pred = np.argmax(cum_blend, axis=1)
            elif net_type == 'regression':  
                cum_blend += model.predict(X_test, verbose=0)
                y_pred = np.floor((cum_blend/i-0.5)*2*5.0).flatten()
                y_pred[y_pred<0] = 0
                y_pred[y_pred>4] = 4
    
            print('\nBlend %d. Accuracy: %0.2f%%. Kappa: %0.2f' %
            (i, 100 * accuracy_score(labels_test, y_pred),
             quadratic_weighted_kappa(labels_test, y_pred)))
            print('Confusion matrix:\n', confusion_matrix(labels_test, y_pred))
            
            fitted = fit2distribution(labels_test, cum_blend)
            print('\nFitted. Accuracy: %0.2f%%. Kappa: %0.2f' %
            (100 * accuracy_score(labels_test, fitted),
            quadratic_weighted_kappa(labels_test, fitted)))
            print('Confusion matrix:\n', confusion_matrix(labels_test, fitted))
            
            if activations:
                F_train = pick_activations(model, X_train, net_type)
                F_test = pick_activations(model, X_test, net_type)
                
                fout = os.path.join(constants.features_NN_dir,
                                    features_NN_prefix + format(i,'02d') +'.hd5')
                with h5py.File(fout, "w") as fo:
                    fo.create_dataset("X_train", data=F_train)
                    fo.create_dataset("y_train", data=labels_train)
                    fo.create_dataset("X_test", data=F_test)
                    fo.create_dataset("y_test", data=labels_test)
                    
                with h5py.File(fout, "r") as fi:
                    X = fi.get("X_train")
                    y = fi.get("y_train")
                    XX = fi.get("X_test")
                    yy = fi.get("y_test")
                    print(X.shape, y.shape, XX.shape, yy.shape)        


def full_train(n_epochs, n_models, activations=False):
    with h5py.File(constants.train_features_scaled_file, "r") as fi:
        labels_train = fi.get("y")[()]
        X_train = fi.get("X")[()]
        y_train, _ = preprocess_labels(labels_train, categorical=(net_type=='softmax'))
    
        y_train = y_train/5.0/2+0.5
    
        if net_type == 'softmax':
            n_classes = y_train.shape[1]
        elif net_type == 'regression':
            n_classes = 1
        print(n_classes, 'classes')
        
        n_dims = X_train.shape[1]
        print(n_dims, 'dims')
    
        cum_blend = 0
        models = range(1, n_models+1)
        for i in models:
            print("\n-------------- Model %d --------------\n" % i)
            model = model_factory(n_classes, n_dims, net_type)
            model.fit(X_train, y_train, nb_epoch=n_epochs, batch_size=128,
                          show_accuracy=False, verbose=2)
    
            # make ensemble prediction
            with h5py.File(constants.test_features_scaled_file, "r") as ft:
                X_test = ft.get("X")
                if net_type == 'softmax':
                    cum_blend += model.predict_proba(X_test, verbose=0)
                elif net_type == 'regression':  
                    cum_blend += model.predict(X_test, verbose=0)
                    
                dump_prediction(cum_blend/i, constants.test_packed_list,
                                constants.pred_dump)


def model_factory(n_classes, n_dims, net_type):
    print('Building model. Net type is %s.' % net_type)
    
    layer1_sz = 1024
    layer2_sz = 1024

    model = Sequential()
    model.add(Dense(n_dims, layer1_sz, init='glorot_uniform'))
    model.add(Activation('relu'))
#    model.add(Dropout(0.5))

    model.add(Dense(layer1_sz, layer2_sz, init='glorot_uniform'))
    model.add(Activation('relu'))
#    model.add(Dropout(0.5))
    
    model.add(Dense(layer2_sz, n_classes, init='glorot_uniform'))
    
    if net_type == 'softmax':
        model.add(Activation('softmax'))
    elif net_type == 'regression':
        pass

    if net_type == 'softmax':
        loss='categorical_crossentropy'
    elif net_type == 'regression':
        loss='mean_squared_error'
    model.compile(loss=loss, optimizer='adam')
    
    return model

  
#np.random.seed(1337) # for reproducibility

net_type = 'regression'  # softmax|regression
mode = 'validate'  # full_train|validate
activations = False
features_NN_prefix = net_type + '.'
      
# Scale
if not (os.path.exists(constants.train_features_scaled_file) and
        os.path.exists(constants.test_features_scaled_file)):
    var, mean = variance(constants.train_features_file)
    sd = np.sqrt(var)
    if not os.path.exists(constants.train_features_scaled_file):
        print('Scaling ' + constants.train_features_scaled_file)
        scale(constants.train_features_file, constants.train_features_scaled_file, sd, mean)
    if not os.path.exists(constants.test_features_scaled_file):
        print('Scaling ' + constants.test_features_scaled_file)
        scale(constants.test_features_file, constants.test_features_scaled_file, sd, mean)

n_epochs = 200
n_models = 10

if mode == 'validate':
    validate(n_epochs, n_models)
elif mode == 'full_train':
    full_train(n_epochs, n_models)
else:
    print('Unknown mode. Exiting.')
    

        