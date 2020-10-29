import argparse
import pickle
import random

import imageio
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
from keras.applications.vgg16 import VGG16

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False, default="model", help="name of trained model")
ap.add_argument("-t", "--testset", required=False, default="testset\\", help="path to test images")
ap.add_argument("-r", "--resize", required=False, default=96, help="frame resize number")
args = vars(ap.parse_args())

TESTSET_FOLDER_PATH = args["testset"]
RESIZE = int(args["resize"])
MODEL_NAME = args["model"]

# Read labels for classes to recognize
with open("classes\\class.lbl", "rb") as file:
    CLASS_LABELS = pickle.load(file)

# Load the trained network
model = load_model("models\\" + MODEL_NAME + ".h5")
model.summary()

video_paths = sorted(list(paths.list_files(TESTSET_FOLDER_PATH)))

random.seed()
random.shuffle(video_paths)

for video_path in video_paths:
    # cap = cv2.VideoCapture(video_path)
    reader = imageio.get_reader(video_path)

    FRAME_RATE = int(reader.get_meta_data()["fps"])

    frames = []
    frames_labels = []
    for frame_id, im in enumerate(reader):
        image = im.copy()
        output = imutils.resize(image, width=400)
        image = cv2.resize(image, (RESIZE, RESIZE))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Classify the input image
        predict = model.predict(image)[0]

        # Find the winner class and the probability
        probability = predict * 100
        winners_indexes = np.argsort(probability)[::-1]

        # Build the label
        for (i, index) in enumerate(winners_indexes):
            label = "{}: {:.6f}%".format(CLASS_LABELS[index], probability[index])

            # Draw the label on the image
            cv2.putText(output, label, (10, (i * 30) + 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

        # Show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(500) & 0xFF

        if key == ord("q"):
            break
