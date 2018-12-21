import os
import numpy as np
import scipy.io as sio
import random
from numpy.fft import fft
from keras.utils import to_categorical
import pywt
cmap = {}   
cnt = 0
cate = []

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
class DataSet(object):
    def __init__(self, sensor_data, labels,  time_start, time_step, context_data = []):
        self.sensor_data = sensor_data
        self.labels = labels
        self.contexts = context_data
        if context_data == []:
            self.contexts = [None]*len(self.sensor_data)
        self.N = len(self.sensor_data)
        self.len = len(self.sensor_data[0])
        self.data = [(self.sensor_data[i], self.labels[i], self.contexts[i]) for i in range(self.N)]
        np.random.shuffle(self.data)
        self.start_index = 0
        self.time_start = time_start
        self.time_step = time_step
    
    def new_get_dis(self, A, B, type):
        if type == 0:
            return self.get_dis(A[0],B[0])
        else:
            return np.sum(np.square(A[2] - B[2]))
        
    def get_dis(self, A, B):
        fA = abs(fft(A))[list(range(int(len(A)/16)))]
        fB = abs(fft(B))[list(range(int(len(B)/16)))]
        # fA = pywt.wavedec(A, 'haar')[0]
        # fB = pywt.wavedec(B, 'haar')[0]
        return np.sqrt(np.sum(np.square(fA - fB)))
    
    def process(self, sensor_data):
        sensor_data = sensor_data[:,self.time_start:self.time_start + self.time_step]
        threshold = sensor_data[:,-1].reshape(-1,1)
        sensor_data = sensor_data - threshold
        sensor_data[:,-1] = threshold.reshape(-1)
        return sensor_data
        
    def generate_batches_C(self, batch_size):
        # generate_batches for classification problem.
        while 1:
            sensor_data = []
            values = []
            for t in range(batch_size):
                i = self.start_index
                self.start_index = (self.start_index + 1)
                if (self.start_index == self.N):
                    self.start_index = 0
                    np.random.shuffle(self.data)

                sensor_data.append(self.data[i][0])
                values.append(self.data[i][1])
            if FLAGS.use_RNN:
                yield [self.process(np.array(sensor_data)).reshape(batch_size,-1,1)], [np.array(to_categorical(values, num_classes=7))]
            else:
                yield [self.process(np.array(sensor_data))], [np.array(to_categorical(values, num_classes=7))]
     
    def generate_batches(self, batch_size, lstm_length = 0):
        while 1:
            sensor_a = []
            sensor_b = []
            sensor_c = []
            labels = []
            similarity = []
            for t in range(batch_size):
                i = self.start_index
                self.start_index = (self.start_index + 1)
                if (self.start_index == self.N):
                    self.start_index = 0
                    np.random.shuffle(self.data)
                
                j = random.randint(0, self.N - 1)
                k = random.randint(0, self.N - 1)
                
                # while (self.new_get_dis(self.data[i], self.data[j], 0) > 15000):
                    # j = random.randint(0, self.N - 1)
               
                # while (self.new_get_dis(self.data[i], self.data[k], 0) < 15000):  
                    # k = random.randint(0, self.N - 1)

                    
                dis_ij = self.new_get_dis(self.data[i], self.data[j], 0)
                dis_ik = self.new_get_dis(self.data[i], self.data[k], 0)
                if (dis_ij > dis_ik):
                   j,k = k,j
                sensor_a.append(self.data[i][0])
                sensor_b.append(self.data[j][0])
                sensor_c.append(self.data[k][0])
                labels.append(self.data[i][1])

                similarity.append(abs(dis_ij - dis_ik))
            sensor_a = self.process(np.array(sensor_a))
            sensor_b = self.process(np.array(sensor_b))
            sensor_c = self.process(np.array(sensor_c))
            similarity = np.array(similarity)
            labels = np.array(to_categorical(labels, num_classes=7))
            # labels_b = np.array(to_categorical(labels_b, num_classes=7))
            # labels_c = np.array(to_categorical(labels_c, num_classes=7))
            yield [sensor_a, sensor_b, sensor_c], [labels, similarity]
                
        
        
def get_label(mess):
    if mess[0] == "Tolerance" and mess[1] == 'High':
        return 0
    if mess[0] == "Tolerance" and mess[1] == 'Low':
        return 1
    if mess[0] == "Warning" and mess[1] == 'High':
        return 2
    if mess[0] == "Warning" and mess[1] == 'Low':
        return 3
    if mess[0] == "Alarm" and mess[1] == 'High':
        return 4
    if mess[0] == "Alarm" and mess[1] == 'Low':
        return 5
        
def get_sensor_data(file):
    print(file)
    tmp_data = []
    tmp_labels = []
    time_map = {}
    sensor_data = {}
    with open(file) as fin:
        for index, line in enumerate(fin.readlines()):
            line = line.strip().split(' ')
            if index % 2 == 1:
                line = [float(i) for i in line]
                line.extend([line[-1]] * (720 - len(line)))
                sensor_data[start_time] = line
            else:
                label = get_label(line)
                start_time = int(line[3])
                if start_time in time_map:
                    time_map[start_time] = max(label, time_map[start_time])
                else:
                    time_map[start_time] = label
    for k,v in time_map.items():
        tmp_data.append(sensor_data[k])
        tmp_labels.append(v)
    return np.array(tmp_data), np.array(tmp_labels)

def get_normal_data(file):
    tmp_data = []
    with open(file) as fin:
        for line in fin.readlines():
            line = [float(i) for i in line.strip().split()]
            line.extend([line[-1]] * (720 - len(line)))
            tmp_data.append(line)
    return np.array(tmp_data)



def do_map(x):
    global cnt
    if x not in cmap:
        cate.append(x)
        cmap[x] = cnt
        cnt += 1
    
def new_load_data():
    try:
        sensor_data = sio.loadmat("." + os.sep + "data" + os.sep + "sensor_data.mat")["sensor_data"]
        labels = sio.loadmat("." + os.sep + "data" + os.sep + "labels.mat")["labels"][0]
        contexts = sio.loadmat("." + os.sep + "data" + os.sep + "contexts.mat")["contexts"]
        print("data size", len(sensor_data))
        print("senor data length", len(sensor_data[0]))
        return sensor_data, labels, contexts
    except:
        pass
    
    data = []
    labels = []
    contexts = []
    fin = open("data/data.txt","r", encoding = 'utf8')
    ordinal = []
    for line in fin.readlines():
        line = line.strip().split()
        label = get_label((line[5], line[6]))
            
        sensor_data = [float(line[i]) for i in range(9, len(line))]
        ## TODO: multi-sample rate 
        sensor_data.extend([sensor_data[-1]] * (720 - len(sensor_data)))
        if len(sensor_data) > 720:
            continue
        context = [line[i] for i in range(3, 5)]
        #sensor_data = np.array(sensor_data)
        #if data == []:
        #    data = sensor_data
        #else:
        #    data = np.vstack((data, sensor_data))
        data.append(sensor_data)
        labels.append(label)
        contexts.append(context)
    for i in range(0,2):
        for c in contexts:
            do_map(c[i])
    for i in range(len(contexts)):
        tmp = [0] * cnt
        for j in range(0,2):
            tmp[cmap[contexts[i][j]]] = 1
        contexts[i] = tmp
        
    data = np.array(data)
    sio.savemat("sensor_data.mat", {'sensor_data':data})
    sio.savemat("labels.mat",{"labels":labels})
    sio.savemat("contexts.mat",{"contexts":contexts})
    return new_load_data(time_start, time_steps)
    
def load_data(time_start, time_steps):
    try:
        sensor_data = sio.loadmat("data-all.mat")["sensor_data"][:,time_start:time_start + time_steps]
        for i in range(len(sensor_data)):
            threshold = sensor_data[i][-1]
            sensor_data[i] = sensor_data[i] - threshold
            sensor_data[i][-1] = threshold
            # for k in range(len(sensor_data[i])- 1, 0, -1):
                # sensor_data[i][k] = sensor_data[i][k] - sensor_data[i][k - 1]
            # sensor_data[i][0] = 0
        return sensor_data,  sio.loadmat("data-all.mat")["labels"][0]
    except:
        pass
    s = os.sep 
    root = os.getcwd() + s + '..' + s + 'siemens-normalData'
    for rt, dirs, files in os.walk(root):
        for index,file in enumerate(files):
            tmp_data = get_normal_data(root + s + file)
            if index == 0:
                data = tmp_data
                labels = [6] * len(tmp_data)
            else:
                data = np.vstack((data,tmp_data))
                labels = np.hstack((labels, [6] * len(tmp_data)))
    root = os.getcwd() + s + '..' + s + 'siemens-issueData'
    for rt, dirs, files in os.walk(root):
        for index, file in enumerate(files):
            tmp_data, tmp_labels = get_sensor_data(root + s + file)
            data = np.vstack((data, tmp_data))
            labels = np.hstack((labels, tmp_labels))
    sio.savemat("data-all.mat", {'sensor_data':data, "labels":labels})
    return load_data(time_start, time_steps)
