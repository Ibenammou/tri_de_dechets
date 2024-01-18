import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoardp
import numpy as np

# Assuming x_train and y_train are your training data
# Replace this with your actual data
x_train = np.random.rand(100, 10)
y_train = np.random.randint(2, size=(100,))

# Create a simple neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10,)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up TensorBoard callback
tensorboard_callback = TensorBoard(log_dir="./logs")

# Fit the model
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])
