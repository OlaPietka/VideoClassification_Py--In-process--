import argparse
import os
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import img_to_array
from skimage.transform import resize
import nn_arch
from sklearn.model_selection import train_test_split


def prepare_data(video_path):
    # Initialize the data and labels
    data = []
    labels = []

    # capturing the video from the given path
    reader = imageio.get_reader(video_path)

    FRAME_RATE = int(reader.get_meta_data()["fps"])

    for frame_id, im in enumerate(reader):
        # Get n frames per second
        if frame_id % (FRAME_RATE // NO_FRAMES) == 0:
            frame = im.copy()
            frame = resize(frame, (RESIZE, RESIZE))
            frame = img_to_array(frame)

            label_text = video_path.split(os.path.sep)[-2]

            data.append(frame)
            labels.append(int(label_text == "Violence"))

        if frame_id > FRAME_RATE * 6:
            break

    data = np.array(data, dtype="float") / 255.0
    labels = to_categorical(labels, num_classes=NO_CLASSES)
    data = preprocess_input(data, mode='tf')

    return data, labels


def data_generator(pathes, batch_size):
    while True: #generators for keras must be infinite
        for path in pathes:
            x_train, y_train = prepare_data(path)

            totalSamps = x_train.shape[0]
            batches = totalSamps // batch_size

            if totalSamps % batch_size > 0:
                batches += 1

            for batch in range(batches):
                section = slice(batch*batch_size,(batch+1)*batch_size)
                yield x_train[section], y_train[section]


def batch_model_1(video_paths):
    for epoch in range(EPOCHS):
        print("Epoch: {}".format(epoch))
        for i, video_path in enumerate(video_paths):
            data, labels = prepare_data(video_path)

            print("---Batch {}---".format(i))
            H = model.train_on_batch(data, labels, reset_metrics=False)
        print(H)
        print(model.metrics_names)
    return H


def fit_model_1(video_paths):
    gen = data_generator(video_paths, BS)
    H = model.fit_generator(gen, steps_per_epoch=len(video_paths) * 30 // BS, epochs=EPOCHS)
    return H


def fit_model_2(video_paths):
    first = True
    for i, video_path in enumerate(video_paths):
        data, labels = prepare_data(video_path)

        # Split to train and validation data
        (train_data, valid_data, train_labels, valid_labels) = train_test_split(data, labels, random_state=42, test_size=0.25)

        if not first:
            model.load_weights('check point\\weight\\weight' + str(i - 1) + '.h5')

        # define the checkpoint
        filepath = "check point\\{}.h5".format(MODEL_NAME)
        checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='min')

        print("---Fit generator---")
        H = model.fit(train_data, train_labels, batch_size=BS, epochs=EPOCHS, validation_data=(valid_data, valid_labels), callbacks=[checkpoint])
        model.save_weights('check point\\weight\\weight' + str(i) + '.h5')

        first = False
    return H


def save_model(model, H):
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


if __name__ == "__main__":
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
    ap.add_argument("-fs", "--frame_per_second", required=False, default=1, help="frame/second")
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

    print("---Prepare VGG16+LSTM model---")
    model = nn_arch.VGG16_LSTM.build(RESIZE, RESIZE, IMG_DEPTH, NO_CLASSES)
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

    if not FROM_FILE:
        # Get all video paths and shuffle them randomly
        video_paths = sorted(list(paths.list_files(VIDEOS_FOLDER_PATH)))

        random.seed()
        random.shuffle(video_paths)

        H = fit_model_2(video_paths)

        save_model(model, H)






