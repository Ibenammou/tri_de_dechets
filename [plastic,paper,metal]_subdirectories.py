import os
import argparse
import requests
import cv2
from imutils import paths

def create_subdirectories(root_directory, class_names):
    for class_name in class_names:
        class_directory = os.path.join(root_directory, class_name)
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)

def download_images(urls_file, output_directory):
    rows = open(urls_file).read().strip().split("\n")
    total = 0

    for url in rows:
        try:
            r = requests.get(url, timeout=60)

            # Save the image to the appropriate subdirectory
            class_name = input("Enter the class name for the image (plastic, paper, metal): ").lower()
            output_subdirectory = os.path.join(output_directory, class_name)
            if not os.path.exists(output_subdirectory):
                os.makedirs(output_subdirectory)

            image_path = os.path.sep.join([output_subdirectory, "{}.jpg".format(str(total).zfill(8))])
            with open(image_path, "wb") as f:
                f.write(r.content)

            print("[INFO] downloaded: {}".format(image_path))
            total += 1

        except Exception as e:
            print("[INFO] error downloading {}...skipping".format(url))
            print(e)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True, help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

# List of class names (plastic, paper, metal)
class_names = ["plastic", "paper", "metal"]

# Create subdirectories for each class
create_subdirectories(args["output"], class_names)

# Download images and save to appropriate subdirectories
download_images(args["urls"], args["output"])