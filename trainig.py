import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import serial
import time

# Path to the trained model
model_path = "C:/Users/windownet/waste_segregation_model.keras"

# Load the trained model
loaded_model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path, img_height=224, img_width=224):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    return img_array / 255.0  # Normalize the image

# Function to classify the image
def classify_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = loaded_model.predict(preprocessed_image)
    class_index = tf.argmax(prediction, axis=1)[0]

    # Define class labels based on your model's output classes
    class_labels = ['paper_waste', 'plastic_waste', 'metal_waste']
    class_label = class_labels[class_index]

    # Adjust the predicted class based on your requirements
    if class_label == 'metal_waste':
        adjusted_class_label = 'plastic_waste'  # Map metal to plastic
    elif class_label == 'plastic_waste':
        adjusted_class_label = 'paper_waste'    # Map plastic to paper
    elif class_label == 'paper_waste':
        adjusted_class_label = 'metal_waste'     # Map paper to metal
    else:
        adjusted_class_label = class_label  # Default to the original label

    return adjusted_class_label

# Function to map the predicted class to the servo angle
def map_predicted_class_to_servo_angle(predicted_class):
    if predicted_class == 'paper_waste':
        return 110
    elif predicted_class == 'plastic_waste':
        return 140
    elif predicted_class == 'metal_waste':
        return 180
    else:
        return 90  # Default angle for other classes

# Function to send angle to Arduino
def send_angle_to_arduino(angle):
    try:
        ser = serial.Serial('COM10', 9600)  # Replace 'COM10' with the appropriate serial port
        time.sleep(2)  # Wait for serial communication to be established
        ser.write(str(angle).encode())
        ser.close()
    except Exception as e:
        print(f"Error communicating with Arduino: {e}")

# Path of the image to test
image_path_to_test = r"C:/Users/mustapha/Desktop/santa/plastic_waste/00000000.jpg"

# Classification of the image
predicted_class = classify_image(image_path_to_test)
print(f"The predicted class is: {predicted_class}")

# Map the class to the servo angle
angle_to_send = map_predicted_class_to_servo_angle(predicted_class)

# Send the angle to Arduino
send_angle_to_arduino(angle_to_send)
