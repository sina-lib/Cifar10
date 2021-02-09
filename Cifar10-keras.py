import tensorflow as tf
from tensorflow.keras import datasets,layers, models
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

# from sklearn.datasets import fetch_mldata

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def Load_Cifar10():
    data = np.array([])
    test_data = np.array([])
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


    # test batch
    with open("DS/cifar-10-batches-py/test_batch", 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    data_t = dic[b'data']
    labels_t = dic[b'labels']

    test_data = data_t.reshape(10000, 3, 32,32).transpose(0,2,3,1)

    data = np.append(data_b1, data_b2)
    data = data.reshape(20000, 3, 32,32).transpose(0,2,3,1)
    return (data,labels),(test_data,labels_t)




if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = Load_Cifar10()

    print("data contains: ", len(train_labels), "labels")
    # print("filenames: ", filenames[0:3])
    print("first 3 labels: ", train_labels[0:3])
    print("data 0 dimension: ", train_data[0].shape)

    print(train_data.shape)
    print(class_names[train_labels[44]]) # for example No. 44

    # plt.figure()
    # plt.imshow(train_data[44])
    # plt.show()

    # convert output class number to one-hot vector:
    enc_10c = OneHotEncoder(sparse=False)
    train_labels = enc_10c.fit_transform(np.array(train_labels).reshape(-1,1))
    # test_labels = enc_10c.fit_transform(np.array(test_labels).reshape(-1,1))

    # build a Model
    model = models.Sequential()

    # define layers
    model.add(layers.Conv2D( 32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D( 120, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D( 60, kernel_size=(3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(30,activation='relu'))
    model.add(layers.Dense(50,activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    # train the network
    model.fit(train_data,train_labels, batch_size=100, validation_split=0.2, verbose=1, epochs=10)

    predict_labels = model.predict(test_data)
    predict_labels = enc_10c.inverse_transform(predict_labels)

    # demonstrate some predictions:
    plt.figure()
    k=4
    for i in range(k):
        for j in range(k):
            plt.subplot(k,k, i*k+j+1)
            plt.imshow(test_data[i*k+j])
            plt.title( class_names[predict_labels[ i*k+j , 0]] )

    plt.show()
