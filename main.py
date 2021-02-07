import tensorflow as tf
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import numpy as np
import pickle

# from sklearn.datasets import fetch_mldata

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def Load_Cifar10():
    data = np.array([])
    labels = []
    with open('DS/cifar-10-batches-py/data_batch_1','rb') as fo:
        dic = pickle.load(fo, encoding='bytes')

    data_b1 = dic[b'data'] # is already a numpy array
    labels_t = dic[b'labels'] # is a list of numbers
    # filenames = dic[b'filenames']

    # reshape all the data
    # data_b1  = data_b1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    labels.extend(labels_t)


    # Batch2
    with open("DS/cifar-10-batches-py/data_batch_2", 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    data_b2 = dic[b'data']  # is already a numpy array
    labels_t = dic[b'labels']  # is a list of numbers

    labels.extend(labels_t)

    data = np.append(data_b1, data_b2)
    data = data.reshape(20000, 3, 32,32).transpose(0,2,3,1)
    return data,labels




if __name__ == '__main__':
    data, labels = Load_Cifar10()

    print("data contains: ", len(labels), "labels")
    # print("filenames: ", filenames[0:3])
    print("first 3 labels: ", labels[0:3])
    print("data 0 dimension: ", data[0].shape)

    print(data.shape)
    print(class_names[labels[44]])

    plt.figure()
    plt.imshow(data[44])
    plt.show()

    
