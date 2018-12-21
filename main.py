from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np

from getData import *
from model import *
import scipy.io as sio
import flags

def read_file(filename, time_start, time_steps):
    data = []
    with open(filename) as fin:
        for line in fin.readlines():
            line = [float(i) for i in line.strip().split()]
            line = line[time_start:time_start + time_steps]
            threshold = line[-1]
            line = [i - line[-1] for i in line]
            line[-1] = threshold
            # for k in range(len(line)-1, 0, -1):
                # line[k] = line[k] - line[k - 1]
            # line[0] = 0
            data.append(line)
    return np.array(data)
    
issues = [''] * 7
issues[0] = 'Tolerance High'
issues[1] = 'Tolerance Low'
issues[2] = 'Warning High'
issues[3] = 'Warning Low'
issues[4] = 'Alarm High'
issues[5] = 'Alarm Low'
issues[6] = 'normal'

FLAGS = tf.app.flags.FLAGS

def main(argv = None):
    model = SNN_basic()
    

    sensor_data, labels, contexts = new_load_data()
    s_train, s_test, l_train, l_test = train_test_split(sensor_data, labels, test_size = 0.1)
    _, _, c_train, c_test = train_test_split(sensor_data, contexts, test_size = 0.1)
    train_data = DataSet(s_train, l_train,  FLAGS.time_start, FLAGS.time_steps, c_train)
    test_data = DataSet(s_test, l_test, FLAGS.time_start, FLAGS.time_steps, c_test)
    print("data prepared")
    model.train(train_data, test_data)
    
    issue_ind = model.predict(train_data.process(sensor_data))
    print(issues[issue_ind[0]])
    hash_code = model.get_hash_code(train_data.process(sensor_data))
        

if __name__ == '__main__':
    tf.app.run()

                

        
