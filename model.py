import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Activation, Flatten
from keras.layers import LSTM,SimpleRNN,GRU
from keras.layers.normalization import BatchNormalization

from keras.layers.core import Lambda, Dropout
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf

from getData import *
import scipy.io as sio

FLAGS = tf.app.flags.FLAGS

def hash_loss(args):
    hash_a, hash_p, hash_n = args
    dis_ap = K.sum(K.square(hash_a - hash_p), axis = 1)
    dis_an = K.sum(K.square(hash_a - hash_n), axis = 1)
    return (dis_an - dis_ap)

def triplet_loss(y_true, y_pred):
    return K.maximum(0.0 , 1 - y_pred)
   
class SNN_basic(object):
    def __init__(self):
        if FLAGS.use_RNN:
            sensor_v = Input(shape = (FLAGS.time_steps, 1))
        else:
            sensor_v = Input(shape = (FLAGS.time_steps, ))
        input = sensor_v
        if FLAGS.use_RNN:
            for width in FLAGS.struct:
                input = SimpleRNN(1,activation = "sigmoid", return_sequences = True, unroll = True)(input)
        else:
            for width in FLAGS.struct:
                input = Dense(width, activation = "sigmoid")(input)
        if FLAGS.use_RNN:        
            input = Flatten()(input)
        self.get_hash = Model(inputs = sensor_v, outputs = input)
        pred = Dense(7, activation = "softmax")(input)
        self.model = Model(inputs = sensor_v, outputs = pred)
        self.model.compile(optimizer = 'rmsprop',
              loss = "categorical_crossentropy",
              metrics = ['accuracy'])
    
    def train(self, train_data, test_data):
        batch_size = FLAGS.batch_size
        return self.model.fit_generator(train_data.generate_batches_C(batch_size), 
                    steps_per_epoch = int(train_data.N / batch_size), 
                    epochs = FLAGS.epochs,
                    validation_data = test_data.generate_batches_C(batch_size),
                    validation_steps = test_data.N / batch_size)

    def predict(self, data):
        if FLAGS.use_RNN:
            data = data.reshape(-1,FLAGS.time_steps,1)
        p1 = self.model.predict(data)
        p1 = np.argmax(p1, axis = 1)
        return p1

    def get_hash_code(self, data):
        return np.sign(self.get_hash.predict(data) - 0.5)
   
class SNN(object):
    def __init__(self):
        sensor_a = Input(shape = (FLAGS.time_steps,))
        sensor_p = Input(shape = (FLAGS.time_steps,))
        sensor_n = Input(shape = (FLAGS.time_steps,))

        hash_model = Sequential()
        for i,width  in enumerate(FLAGS.struct):
            if i == 0:
                hash_model.add(Dense(width, input_shape = (FLAGS.time_steps,), activation = 'sigmoid'))
            else:
                hash_model.add(Dropout(0.1))
                hash_model.add(Dense(width, activation = 'sigmoid'))
       
        
        feature_a = hash_model(sensor_a)
        feature_p = hash_model(sensor_p)
        feature_n = hash_model(sensor_n)

        hash_a = Lambda(lambda x : K.sign(x - 0.5), name = "hash_code")(feature_a)
        hash_p = Lambda(lambda x : K.sign(x - 0.5))(feature_p)
        hash_n = Lambda(lambda x : K.sign(x - 0.5))(feature_n)

        loss_out = Lambda(hash_loss, output_shape = (1,), name = 'hash_loss')([hash_a, hash_p, hash_n])

        classifier = Dense(7, activation = 'softmax', name = 'classifier')
        predict = classifier(feature_a)

        self.model = Model(inputs = [sensor_a, sensor_p, sensor_n], outputs = [predict, loss_out])
        self.model.compile(optimizer = 'rmsprop',
                      loss = {
                      'hash_loss':triplet_loss, 
                      'classifier':'categorical_crossentropy'},
                      loss_weights = [1, 1],
                      metrics = ['accuracy'])

    def train(self, train_data, test_data):
        batch_size = FLAGS.batch_size
        return self.model.fit_generator(train_data.generate_batches(batch_size), 
                    steps_per_epoch = int(train_data.N / batch_size), 
                    epochs = FLAGS.epochs,
                    validation_data = test_data.generate_batches(batch_size),
                    validation_steps = test_data.N / batch_size)

    def predict(self, data):
        p1,  _ = self.model.predict([data, data, data])
        p1 = np.argmax(p1, axis = 1)
        return p1
        
    def get_hash(self, data):
        hash_model = Model(inputs = self.model.input, outputs = self.model.get_layer('hash_code').output)
        return np.sign(hash_model.predict([data, data, data]) - 0.5)
        
class SNN_REG(object):
    def __init__(self):
        def in_range(y_true, y_pred):
            a = K.cast(K.greater(y_true * 1.1, y_pred), "float32") 
            b = K.cast(K.less(y_true * 0.9, y_pred), "float32") 
            return a * b
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        set_session(tf.Session(config = tf_config))

        FLAGS = FLAGS
        if FLAGS.load_model:
            try:
                self.model = keras.models.load_model(FLAGS.old_model)
                print("load model successfully")
                return
            except:
                print("fail to load model")
                pass
        self.model = Sequential()
        
        for i,length  in enumerate(FLAGS.struct):
            if i == 0:
                self.model.add(Dense(length, input_shape = (FLAGS.time_steps,), activation = 'sigmoid'))
            else:
                self.model.add(Dense(length, activation = 'sigmoid'))
        self.model.add(Dense(1))

        self.model.compile(optimizer = 'adam',
                      loss = "mean_squared_error",
                      metrics = [in_range])

    def train(self, train_data, test_data):
        batch_size = FLAGS.batch_size
        return self.model.fit_generator(train_data.generate_batches_reg(batch_size), 
                    steps_per_epoch = int(train_data.N / batch_size), 
                    epochs = FLAGS.epochs,
                    validation_data = test_data.generate_batches_reg(batch_size),
                    validation_steps = test_data.N / batch_size)

    def train_PT(self, train_data, test_data):
        batch_size = FLAGS.batch_size
        return self.model.fit_generator(train_data.generate_batches_PT(batch_size), 
                    steps_per_epoch = int(train_data.N / batch_size), 
                    epochs = FLAGS.epochs,
                    validation_data = test_data.generate_batches_PT(batch_size),
                    validation_steps = test_data.N / batch_size)
                    
    def predict(self, data):
        return self.model.predict([data])


    def __del__(self):
        K.clear_session()





        