import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Path to your dataset
dataset_path = "C:/Users/Windownet/Desktop/waste segregation/santa"

# Parameters
batch_size = 16
img_height, img_width = 224, 224

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # Assuming 3 classes (paper, metal, plastic)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the model as .keras
model.save("C:/Users/Windownet/Desktop/waste segregation/waste_segregation_model.keras")

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Save the model in Keras format
# test_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Load the model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Load the model (assuming it was saved as .keras)
loaded_model = tf.keras.models.load_model("waste_segregation_model.keras")

# Path to the image you want to test
image_path_to_test = r'C:\\Users\\Windownet\\Desktop\\test1.jpg'

# Preprocess the Input Image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    return img_array / 255.0  # Normalize the image

# Classify the Image
def classify_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = loaded_model.predict(preprocessed_image)
    class_index = tf.argmax(prediction, axis=1)[0]
    class_label = list(train_generator.class_indices.keys())[class_index]

    return class_label
import os

image_path_to_test = "C:\\Users\\Windownet\\Desktop\\test1.jpg"

if os.path.exists(image_path_to_test):
    predicted_class = classify_image(image_path_to_test)
    print(f"The predicted class is: {predicted_class}")
else:
    print(f"Error: The file '{image_path_to_test}' does not exist.")

# Test with an Image
predicted_class = classify_image(image_path_to_test)
print(f"The predicted class is: {predicted_class}")
