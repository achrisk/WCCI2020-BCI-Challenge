import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.constraints import unit_norm

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from pyriemann.clustering import Potato

from data_utils import *

def eegnet(F1=8, C1=256, D=2, P1=4, t=1, P2=8, drop=0.5):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(F1, (1, C1), padding="same", input_shape=(12, 1536, 1), use_bias=False))
    model.add(layers.BatchNormalization())
    C = num_channels
    model.add(layers.DepthwiseConv2D((C, 1), padding="valid", depth_multiplier=D, depthwise_constraint=unit_norm(), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling2D(pool_size=(1, P1), padding="valid"))
    model.add(layers.Dropout(drop))
    F2 = D*F1
    C2 = int((512/P1)*t)
    model.add(layers.SeparableConv2D(F2, (1, C2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling2D(pool_size=(1, P2), padding="valid"))
    model.add(layers.Dropout(drop))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
    
    return model

def get_model(in_shape=32, num_hidden=16, activation='relu'):
    #build model
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(in_shape, )))
    model.add(layers.Dense(num_hidden, activation=activation))
    model.add(layers.Dense(1, activation='sigmoid'))

    #compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    return model

def holdout(model_fn=get_model, data_fn=get_riemann_ts, mode="within", patients=[1], num_epochs=None, num_hid=None, flt_size=None, test_size=0.2, trim=(-1536,0), plot="off", vf=0, ve=1):
    for i in patients:
        if mode=="within":
            X, y, X_eval = get_data_within(i, trim, low=8, high=35)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            if flt_size is None:
                X_train = data_fn(X_train)
                X_test = data_fn(X_test)
                X_eval = data_fn(X_eval)
            print(X_train.shape)
            
        else:
            X_train, y_train, X_test, y_test, X_eval = get_data_inter(i, low=8, high=35)
            if flt_size is None:
                X_train = data_fn(X_train)
                X_test = data_fn(X_test)
                X_eval = data_fn(X_eval)
            print(X_train.shape, X_test.shape)
        if num_epochs is not None:
            if flt_size is not None:
                if X_train.ndim == 3:
                    X_train = np.expand_dims(X_train, axis=3)
                if X_test.ndim == 3:
                    X_test = np.expand_dims(X_test, axis=3)
                if X_eval.ndim == 3:
                    X_eval = np.expand_dims(X_eval, axis=3)
                model = model_fn(C1=flt_size)
            else:
                model = model_fn(in_shape=X_train.shape[-1], num_hidden=num_hid)
            history = model.fit(X_train, y_train, epochs=num_epochs, verbose=vf)
            # evaluate
            loss, train_acc = model.evaluate(X_train, y_train, verbose=ve)
            loss, test_acc = model.evaluate(X_test, y_test, verbose=ve)
            y_pred = (np.rint(np.squeeze(model.predict(X_test)))).astype(int)
            kappa = metrics.cohen_kappa_score(y_pred, y_test)
            print('Patient=%d'%(i))
            print('Kappa Score: %f | Training accuracy: %f | Testing accuracy: %f'%(kappa, train_acc, test_acc))
            y_eval = (np.rint(np.squeeze(model.predict(X_eval)))).astype(int)
            for y in y_eval:
                print(y, end=' ')
            print('\nX_eval stats:\n#0:%d  |  #1:%d'%(sum(1 for x in y_eval if x==0), sum(1 for x in y_eval if x==1)))
            if plot=="on":
                # Plot training accuracy values
                plt.plot(history.history['binary_accuracy'])
                plt.title('Model training accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.show()

                # Plot training loss values
                plt.plot(history.history['loss'])
                plt.title('Model training loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.show()
        else:
            # train
            model = model_fn()
            model.fit(X_train, y_train)
            # evaluate
            y_pred = model.predict(X_train)
            train_acc = metrics.accuracy_score(y_train, y_pred)
            y_pred = model.predict(X_test)
            print(y_test)
            print(y_pred)
            kappa = metrics.cohen_kappa_score(y_test, y_pred)
            test_acc = metrics.accuracy_score(y_test, y_pred)
            y_eval = model.predict(X_eval)
            print('Patient=%d'%(i))
            print('Kappa Score: %f | Training accuracy: %f | Testing accuracy: %f'%(kappa, train_acc, test_acc))
            for y in y_eval:
                print(y, end=' ')
            print('\nX_eval stats:\n#0:%d  |  #1:%d'%(sum(1 for x in y_eval if x==0), sum(1 for x in y_eval if x==1)))
    return kappa
            
def cross_val(model_fn=get_model, data_fn=get_riemann_ts, patients=[1], num_folds=5, num_epochs=None, num_hid=None, flt_size=None, trim=(-1536,0), plot="off", vf=0, ve=1):
    for i in patients:
        X, y, X_eval = get_data_within(i, trim, low=8, high=35)
        if flt_size is None:
            X_eval = data_fn(X_eval)
        if num_epochs is not None:
            val_acc = []
            kappas = []
            kf = StratifiedKFold(num_folds, shuffle=True)
            it=0
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                if flt_size is not None:
                    model = model_fn(C1=flt_size)
                    if X_train.ndim == 3:
                        X_train = np.expand_dims(X_train, axis=3)
                    if X_test.ndim == 3:
                        X_test = np.expand_dims(X_test, axis=3)
                    if X_eval.ndim == 3:
                        X_eval = np.expand_dims(X_eval, axis=3)
                else:
                    X_train = data_fn(X_train)
                    X_test = data_fn(X_test)
                    model = model_fn(in_shape=X_train.shape[-1], num_hidden=num_hid)

                history = model.fit(X_train, y_train, epochs=num_epochs, verbose=vf)
                # evaluate
                loss, train_acc = model.evaluate(X_train, y_train, verbose=ve)
                loss, test_acc = model.evaluate(X_test, y_test, verbose=ve)
                y_pred = (np.rint(np.squeeze(model.predict(X_test)))).astype(int)
                kappa = metrics.cohen_kappa_score(y_pred, y_test)
                val_acc.append(test_acc)
                kappas.append(kappa)
                it+=1
                print('Fold %d done | Kappa: %f | Accuracy: %f'%(it, kappa, test_acc))
                if plot=="on":
                    # Plot training accuracy values
                    plt.plot(history.history['accuracy'])
                    plt.title('Model training accuracy')
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.show()

                    # Plot training loss values
                    plt.plot(history.history['loss'])
                    plt.title('Model training loss')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.show()
                y_eval = (np.rint(np.squeeze(model.predict(X_eval)))).astype(int)
                for ye in y_eval:
                    print(ye, end=' ')
                print('\nX_eval stats:\n#0:%d  |  #1:%d'%(sum(1 for x in y_eval if x==0), sum(1 for x in y_eval if x==1)))
            print('Patient=%d | Kappa: %f | Accuracy=%f\n'%(i, np.mean(np.array(kappas)), np.mean(np.array(val_acc))))
        else:
            val_acc = []
            kappas = []
            kf = KFold(num_folds, shuffle=True)
            it=0
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                X_train = data_fn(X_train)
                X_test = data_fn(X_test)
                # train
                model = model_fn()
                model.fit(X_train, y_train)
                # evaluate
                y_pred = model.predict(X_test)
                test_acc = metrics.accuracy_score(y_test, y_pred)
                val_acc.append(test_acc)
                kappa = metrics.cohen_kappa_score(y_test, y_pred)
                kappas.append(kappa)
                it+=1
                print('Fold %d done | Kappa: %f | Accuracy: %f'%(it, kappa, test_acc))
                y_eval = model.predict(X_eval)
                for ye in y_eval:
                    print(ye, end=' ')
                print('\nX_eval stats:\n#0:%d  |  #1:%d'%(sum(1 for x in y_eval if x==0), sum(1 for x in y_eval if x==1)))
            print('Patient=%d | Kappa=%f | Accuracy=%f\n'%(i, np.mean(np.array(kappas)), np.mean(np.array(val_acc))))
    return np.array(kappas)
            
def eval_methods(model_fn=get_model, data_fn=get_riemann_ts, mode="within", patient=1, num_epochs=None, num_hid=None, num_runs=9):
    if mode=="within":
        X, y, X_eval = get_data_within(patient, low=8, high=35)
    else:
        X, y, X_eval = get_data_inter(patient, low=8, high=35, mode="eval")
    X = data_fn(X)
    X_eval = data_fn(X_eval)
    print(X.shape, X_eval.shape)
    if num_epochs is not None:
        eval_accum = np.zeros((X_eval.shape[0],1), dtype="float64")
        for j in range(num_runs):
            # train
            model = model_fn(in_shape=78, num_hidden=num_hid)
            history = model.fit(X, y, epochs=num_epochs, verbose=0)
            # evaluate
            loss, train_acc = model.evaluate(X, y, verbose=0)
#             print('RUN: %d DONE'%(j+1))
#             print('Training accuracy: %f'%(train_acc))
            y_eval = (np.rint(np.squeeze(model.predict(X_eval)))).astype(int)
            eval_accum += model.predict(X_eval)
#             for ye in y_eval:
#                 print(ye, end=' ')
#             print('\nX_eval stats:\n#0:%d  |  #1:%d'%(sum(1 for x in y_eval if x==0), sum(1 for x in y_eval if x==1)))
        eval_accum /= num_runs
        eval_accum = (np.rint(np.squeeze(eval_accum))).astype(int)
#         for ye in eval_accum:
#                 print(ye, end=' ')
#         print('\nPatient %d Accumulated stats:\n#0:%d  |  #1:%d\n'%(patient, sum(1 for x in eval_accum if x==0), sum(1 for x in eval_accum if x==1)))
        return eval_accum
    else:
        # train
        model = model_fn()
        model.fit(X, y)
        # evaluate
        y_pred = model.predict(X)
        train_acc = metrics.accuracy_score(y, y_pred)
        y_eval = model.predict(X_eval)
#         print('Training accuracy: %f'%(train_acc))
#         for ye in y_eval:
#             print(ye, end=' ')
#         print('\nPatient %d Accumulated stats:\n#0:%d  |  #1:%d\n'%(patient, sum(1 for x in y_eval if x==0), sum(1 for x in y_eval if x==1)))
        return y_eval
    
def eval_eegnet(mode="within", patient=1, num_epochs=300, flt_size=128, num_runs=9):
    if mode=="within":
        X, y, X_eval = get_data_within(patient, low=8, high=24)
    else:
        X, y, X_eval = get_data_inter(patient, low=8, high=24, mode="eval")
    print(X.shape, X_eval.shape)
    if X.ndim == 3:
        X = np.expand_dims(X, axis=3)
    if X_eval.ndim == 3:
        X_eval = np.expand_dims(X_eval, axis=3)

    eval_accum = np.zeros((X_eval.shape[0],1), dtype="float64")
    for j in range(num_runs):
        # train model
        model = eegnet(C1=flt_size, drop=0.5)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
        history = model.fit(X, y, epochs=num_epochs, verbose=0)
        #evaluate
        loss, train_acc = model.evaluate(X, y, verbose=0)
        print('RUN: %d DONE'%(j+1))
#         print('Training accuracy: %f'%(train_acc))
        y_eval = (np.rint(np.squeeze(model.predict(X_eval)))).astype(int)
        eval_accum += model.predict(X_eval)
#         for ye in y_eval:
#             print(ye, end=' ')
#         print('\nX_eval stats:\n#0:%d  |  #1:%d'%(sum(1 for x in y_eval if x==0), sum(1 for x in y_eval if x==1)))
    eval_accum /= num_runs
    eval_accum = (np.rint(np.squeeze(eval_accum))).astype(int)
#     for j in range(eval_accum.size):
#         if eval_accum[j]>int(num_runs/2):
#             eval_accum[j] = 1
#         else:
#             eval_accum[j] = 0
#     for ye in eval_accum:
#                 print(ye, end=' ')
#     print('\nPatient %d Accumulated stats:\n#0:%d  |  #1:%d\n'%(patient, sum(1 for x in eval_accum if x==0), sum(1 for x in eval_accum if x==1)))
    return eval_accum