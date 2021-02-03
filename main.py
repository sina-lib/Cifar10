import tensorflow as tf
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import numpy as np
import pickle

# from sklearn.datasets import fetch_mldata

def Load_Cifar10():
    with open('DS/cifar-10-batches-py/data_batch_1','rb') as fo:
        dic = pickle.load(fo, encoding='bytes')

    data = dic[b'data'] # is already a numpy array
    labels = dic[b'labels'] # is a list of numbers
    filenames = dic[b'filenames']

    # reshape all the data
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

    return data,labels,filenames




if __name__ == '__main__':
    data, labels, filenames = Load_Cifar10()

    print("data contains: ", len(labels), "labels")
    print("filenames: ", filenames[0:3])
    print("labels: ", labels[0:3])
    print("data dimension: ", data[0].shape)

    plt.figure()

    for i in range(3):
        for j in range(3):
            plt.subplot(3, 3, 3 * i + j + 1)
            plt.imshow(data[3 * i + j])
    plt.show()
