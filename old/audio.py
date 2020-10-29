import os
import random
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import moviepy.editor
import numpy as np
from imutils import paths
from keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import img_to_array
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import nn_arch

VIDEOS_FOLDER_PATH = "dataset\\videos - audio\\"
AUDIOS_FOLDER_PATH = "dataset\\audios\\"
SPECTOGRAMS_FOLDER_PATH = "dataset\\spectograms\\"

ALL_VIDEOS_FOLDERS_PATHS = ["dataset\\videos\\", "dataset\\videos - audio\\", "dataset\\temp\\", "dataset\\temp2\\", "temp\\"]

RESIZE = 96
EPOCHS = 1000
BS = 32
IMG_DEPTH = 3
NO_CLASSES = 2


def extract_all_audio():
    for video_folder_path in ALL_VIDEOS_FOLDERS_PATHS:
        video_paths = sorted(list(paths.list_files(video_folder_path)))

        for video_path in video_paths:
            video = moviepy.editor.VideoFileClip(video_path)
            audio = video.audio

            if audio is None:
                continue

            label, fname = video_path.split(os.path.sep)[-2:]
            fname = fname.split(".")[0]

            audio_path = "\\".join([label, fname])

            audio.write_audiofile("dataset\\audios\\{}.wav".format(audio_path))


def extract_audio():
    video_paths = sorted(list(paths.list_files(VIDEOS_FOLDER_PATH)))

    for video_path in video_paths:
        video = moviepy.editor.VideoFileClip(video_path)
        audio = video.audio

        if audio is None:
            continue

        label, fname = video_path.split(os.path.sep)[-2:]
        fname = fname.split(".")[0]

        audio_path = "\\".join([label, fname])

        audio.write_audiofile("dataset\\audios\\{}.wav".format(audio_path))


def extract_spectograms():
    def create_spectrogram(filename, name):
        clip, sample_rate = librosa.load(filename, sr=None)

        fig = plt.figure(figsize=[0.72, 0.72])

        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)

        S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

        plt.savefig(name, dpi=400, bbox_inches='tight', pad_inches=0)

    audio_paths = sorted(list(paths.list_files(AUDIOS_FOLDER_PATH)))

    for audio_path in audio_paths:

        if "Non" in audio_path:
            print(audio_path)
            label, fname = audio_path.split(os.path.sep)[-2:]
            fname = fname.split(".")[0]

            spectogram_path = "\\".join([label, fname])

            create_spectrogram(audio_path, "dataset\\spectograms\\{}.jpg".format(spectogram_path))


def prepare_data():
    specogram_paths = sorted(list(paths.list_files(SPECTOGRAMS_FOLDER_PATH)))

    random.seed()
    random.shuffle(specogram_paths)

    data = []
    labels = []
    for specogram_path in specogram_paths:
        specogram = cv2.imread(specogram_path)
        specogram = resize(specogram, (RESIZE, RESIZE))
        specogram = img_to_array(specogram)

        label = specogram_path.split(os.path.sep)[-2]

        data.append(specogram)
        labels.append(int(label == "Violence"))

    data = np.array(data).astype(int)
    print(labels)
    labels = to_categorical(labels)
    print(labels)

    data = preprocess_input(data, mode='tf')

    return data, labels


if __name__ == "__main__":
    data, labels = prepare_data()

    print(data.shape)
    print(labels.shape)
    train_data, valid_data, train_labels, valid_labels = train_test_split(data, labels, random_state=42, test_size=0.25)

    model = nn_arch.SmallerVGGNet.build(RESIZE, RESIZE, IMG_DEPTH, NO_CLASSES)

    model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

    print("---Fit generator---")
    H = model.fit(train_data, train_labels, batch_size=BS, epochs=EPOCHS, validation_data=(valid_data, valid_labels))

    # Save model to disk
    model.save("models\\audio.h5")

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
    plt.savefig("plots\\audio.png")

#extract_audio()
#extract_spectograms()