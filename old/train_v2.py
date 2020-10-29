import argparse
import os
import pickle
import random
import imageio
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from skimage.transform import resize
from keras import Sequential, Model, Input
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, LSTM, Reshape
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default="dataset\\videos\\", help="path to input dataset")
ap.add_argument("-e", "--epoch", required=False, default=15, help="number of epochs")
ap.add_argument("-m", "--model", required=False, default="model", help="name of output model")
ap.add_argument("-r", "--resize", required=False, default=96, help="frame resize number")
ap.add_argument("-b", "--batch_size", required=False, default=32, help="batch size")
ap.add_argument("-f", "--from_file", required=False, action='store_true', help="get data from files")
ap.add_argument("-df", "--data_file_name", required=False, default="data", help="name of data file")
ap.add_argument("-lf", "--labels_file_name", required=False, default="labels", help="name of label file")
ap.add_argument("-fs", "--frame_per_second", required=False, default=2, help="frame/second")
args = vars(ap.parse_args())

VIDEOS_FOLDER_PATH = args["dataset"]
MODEL_NAME = args["model"]
FROM_FILE = args["from_file"]
DATA_FILE_NAME = "data\\" + args["data_file_name"]
LABELS_FILE_NAME = "data\\" + args["labels_file_name"]

RESIZE = int(args["resize"])
EPOCHS = int(args["epoch"])
BS = int(args["batch_size"])
NO_FRAMES = int(args["frame_per_second"])
IMG_DEPTH = 3
NO_CLASSES = 2

if not FROM_FILE:
    # Get all video paths and shuffle them randomly
    video_paths = sorted(list(paths.list_files(VIDEOS_FOLDER_PATH)))

    random.seed()
    random.shuffle(video_paths)

    # Initialize the data and labels
    data = []
    labels = []
    labels_text = []
    for i, video_path in enumerate(video_paths):
        # capturing the video from the given path
        reader = imageio.get_reader(video_path)

        FRAME_RATE = int(reader.get_meta_data()["fps"])

        frames = []
        frames_labels = []
        for frame_id, im in enumerate(reader):
            # Get n frames per second
            if frame_id % (FRAME_RATE // NO_FRAMES) == 0:
                frame = im.copy()
                frame = resize(frame, (RESIZE, RESIZE))
                frame = img_to_array(frame)

                label_text = video_path.split(os.path.sep)[-2]

                frames.append(frame)
                frames_labels.append(int(label_text == "Violence"))
            if frame_id > FRAME_RATE*2:
                break

        print("Path:", video_path, "| Label:", label_text, "| Num:", i)

        data.append(frames)
        labels.append(frames_labels)

    data = np.array(data, dtype="float") / 255.0
    labels = to_categorical(labels, num_classes=NO_CLASSES)
    data = preprocess_input(data, mode='tf')

    with open(DATA_FILE_NAME + ".txt", 'wb') as file:
        pickle.dump(data, file)
    with open(LABELS_FILE_NAME + ".txt", 'wb') as file:
        pickle.dump(labels, file)
else:
    with open(DATA_FILE_NAME + ".txt", 'rb') as file:
        data = pickle.load(file)
    with open(LABELS_FILE_NAME + ".txt", 'rb') as file:
        labels = pickle.load(file)


t_num_vid, t_num_fram, t_width, t_height, t_depth = data.shape
reshape_1d = t_width * t_height

data = data.reshape(t_num_vid, t_num_fram, reshape_1d, t_depth)

data = data/data.max()


print("---Prepare VGG16 model---")
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(t_num_fram, reshape_1d, t_depth))

# Disable first top 10 layers
for layer in base_model.layers[:10]:
    layer.trainable = False

print("---Create LSTM model---")
model = base_model.get_layer("block5_pool").output
model = Reshape(target_shape=(3*3, 512))(model)
model = LSTM(256, return_sequences=True)(model)
model = LSTM(256)(model)
model = Dropout(0.5)(model)
model = Dense(NO_CLASSES, activation='softmax')(model)

# Connect VGG16 and LSTM model
model = Model(base_model.input, model)
model.summary()

model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

print("---Fit generator---")
H = model.fit(data, labels, batch_size=BS, epochs=EPOCHS)

# Save model to disk
model.save("models\\" + MODEL_NAME + ".h5")

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("plots\\" + MODEL_NAME + ".png")
