import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import argparse
import matplotlib.pyplot as plt

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape((train_x.shape[0],-1))
test_x = test_x.reshape((test_x.shape[0],-1))
y_train_encoded = to_categorical(train_y, num_classes=10)
y_test_encoded = to_categorical(test_y, num_classes=10)
model = Sequential([
    Dense(1000, activation = 'tanh', input_shape = (train_x.shape[1],)),
    # Dense(500, activation = 'tanh'),
    Dense(10, activation = 'softmax')
])
model.compile(optimizer=SGD(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, y_train_encoded, epochs = 50, batch_size = 256, validation_split = 0.2)

loss, accuracy = model.evaluate(test_x, y_test_encoded)
print(f'Test Accuracy: {accuracy:.4f}')
y_pred_probs = model.predict(test_x)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# SGD: 0.8879 Adam: 0.8915