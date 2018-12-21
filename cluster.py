import scipy.io as sio
from numpy.fft import fft
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pywt

data = sio.loadmat("data-all.mat")['sensor_data']
labels = sio.loadmat("data-all.mat")['labels'][0]

new_data = []
for sensor,label in zip(data, labels):
    feature = abs(fft(sensor))[list(range(int(len(sensor)/5)))]
    
    #haar = pywt.wavedec(sensor, 'haar')
    #feature = haar[0]
    #for i in haar[1:]:
    #    feature = np.hstack((feature, i))

    #feature = sensor
    
    new_data.append((feature,label))

np.random.shuffle(new_data)
new_data = np.array(new_data)[list(range(int(len(new_data) / 10)))]

new_feature = np.array([f for f,label in new_data])

print("feature done")

embedding = TSNE(n_components = 2).fit_transform(new_feature)
print("tsne done")

vis_x = embedding[:,0]
vis_y = embedding[:,1]

with open("fft.txt","w") as fout:
    print(len(vis_x))
    for i in vis_x:
        print(i, end=' ', file=fout)
    print("\n", end=' ', file=fout)
    for i in vis_y:
        print(i, end=' ', file=fout)
    print('\n', end=' ', file=fout)
    for f,l in new_data:
        print(l, end=' ', file=fout)


