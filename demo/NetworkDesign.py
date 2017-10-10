import tensorflow as tf
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint,TensorBoard

def LSTMTrain(x_train, y_train, x_test, y_test, TotalTimeStep, FeaDim, name, model_history, init_epoch):

    print('ready')
    print(TotalTimeStep)
    print(FeaDim)

    batch_size = 200
    epochs = init_epoch+60

    print(x_train.shape)
    print(x_test.shape)
    #   D tensor with shape (batch_size, timesteps, input_dim)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(TotalTimeStep, FeaDim)))#dropout=0.2, recurrent_dropout=0.3
    model.add(GlobalAveragePooling1D())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))
    model.summary()


    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  # optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])

    if(model_history != ''):
        print('load history')
        model.load_weights(model_history)

    root_path = '/mi/share/scratch/xiez/PSLSTMDyadic/'

    history = model.fit(x_train, y_train,
                        callbacks = [TensorBoard(log_dir=root_path+'LSTMLog/'+name+'/', histogram_freq=0),
                                    ModelCheckpoint(root_path+'LSTMLog/'+name+'/weights_{epoch:02d}.hdf5', monitor = 'val_acc', period = 30)],
                                    # ModelCheckpoint('./LSTMLog/'+name+'/weights_{epoch:02d}_{val_acc:.2f}.hdf5', monitor = 'val_acc', period = 30)],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        shuffle=True,
                        validation_data=(x_train, y_train),
                        initial_epoch=init_epoch)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1];
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])





def keras_twoLayerNet(x_train, y_train, x_test, y_test, dim):

    dropout_rate = 0.5
    print('dropout rate: ' + str(dropout_rate))
    batch_size = 200
    epochs = 60

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(dim,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  # optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_train, y_train))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]



def keras_oneLayerNet(x_train, y_train, x_test, y_test, dim):

    dropout_rate = 0
    print('dropout rate: ' + str(dropout_rate))
    batch_size = 200
    epochs = 80

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=(dim,)))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  # optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_train, y_train))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score[1]

