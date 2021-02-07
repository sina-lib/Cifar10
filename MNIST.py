from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,models,optimizers
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder

# prepare dataset ---------------------------------------------------------
(trainX, trainY), (testX,testY) = mnist.load_data()
trainY = trainY.reshape(-1,1)
testY  = testY.reshape(-1,1)

print("X dimension: ",trainX.shape)
print("Y dimension: ",trainY.shape)

# import matplotlib.pyplot as plt
# plt.imshow(testX[0],cmap=plt.get_cmap("gray"))
# plt.show()
enc = OneHotEncoder(sparse=False)
trainY = enc.fit_transform(trainY)

# build the model ----------------------------------------------------------
model = models.Sequential()

# Model 1:
# model.add(layers.Conv2D(32,kernel_size=(3,3),activation='relu', input_shape=(28,28,1)))
# model.add(layers.Conv2D(64,kernel_size=(3,3),activation='relu'))
# model.add(layers.MaxPool2D(pool_size=(2,2)))
# model.add(layers.Dropout(0.25))
# model.add(layers.Flatten())
# model.add(layers.Dense(128,activation='relu'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(10, activation='softmax'))
# Model 2:

# model.add(layers.Dense(40,activation='sigmoid'))
model.add(layers.Conv2D(10,kernel_size=(3,3),activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(30,activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


trainX = trainX.reshape(60000,28,28,1)
testX = testX.reshape(10000,28,28,1)
print(trainY)

model.fit(trainX,trainY, batch_size=128, validation_split=0.3, epochs=1, verbose=1)


predictY = model.predict(testX)
predictY = enc.inverse_transform(predictY)

import matplotlib.pyplot as plt

for i in range(3):
    for j in range(3):
        plt.subplot(3,3, 3*i+j+1)
        plt.imshow(testX[3*i+j])
        plt.title(predictY[3*i+j])

# print("test labels: ",testY[:7])
# print("test prediction: ",predictY[:7])
plt.show()

