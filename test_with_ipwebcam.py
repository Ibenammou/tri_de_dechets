import cv2
import requests
import numpy as np
import tensorflow as tf

# IP Webcam configuration
ip_webcam_url = 'http://192.168.1.8:8080/shot.jpg'

# Load TensorFlow model
# Load pre-trained Inception model
model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)

def capture_image():
    # Access the IP Webcam stream and capture an image
    response = requests.get(ip_webcam_url)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    return img

def preprocess_image(img):
    # Resize image to match the input size expected by the model
    resized_img = cv2.resize(img, (224, 224))

    # Normalize image to be in the range [0, 1]
    normalized_img = resized_img / 255.0

    # Expand dimensions to create a batch of size 1
    preprocessed_img = np.expand_dims(normalized_img, axis=0)

    return preprocessed_img

def classify_waste_type(model, preprocessed_img):
    # Run inference on the preprocessed image using your TensorFlow model
    predictions = model(preprocessed_img)
    
    # Map predictions to waste types
    waste_type = map_predictions_to_waste_type(predictions)

    return waste_type

def map_predictions_to_waste_type(predictions):
    # For simplicity, let's assume a binary classification (e.g., recyclable or non-recyclable)
    # You need to customize this based on your model's output
    if predictions[0, 0] > 0.5:
        waste_type = "plastic_waste"
    elif predictions[0, 2] > 0.5:  # Assuming the third class corresponds to metal
        waste_type = "metal_waste"
    else:
        waste_type = "paper_waste"

    return waste_type

def control_servo_motor(waste_type):
    # Implement logic to control servo motors based on waste type
    # For demonstration purposes, just print the waste type
    print(f"Waste Type: {waste_type}")

if __name__ == "__main__":
    while True:
        # Capture image from IP Webcam
        img = capture_image()

        # Preprocess image
        preprocessed_img = preprocess_image(img)

        # Classify waste type
        waste_type = classify_waste_type(model, preprocessed_img)

        # Control servo motors based on waste type
        control_servo_motor(waste_type)
