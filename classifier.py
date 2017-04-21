from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential
import keras
from data_utils import *
import numpy as np

train_X, train_y, test_X, test_y = load_CIFAR10("cifar-10-batches-py")



N = 50000
D = 32 * 32 * 3

input_shape = (32, 32, 3)

print test_X.shape
#train_X = train_X.reshape(N, D)
#test_X = test_X.reshape(10000, D)

print train_X.shape

model = Sequential()

model.add(Conv2D(32, kernel_size=(4, 4),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.3))

model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25w))


model.add(Flatten())
model.add(Dense(256, activation='relu'))

model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.fit(train_X, train_y, epochs=10, batch_size=64)

loss_and_metrics = model.evaluate(test_X, test_y, batch_size=128)
print loss_and_metrics
