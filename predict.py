import argparse
from imutils import paths
from keras.models import load_model
import numpy as np
import imutils
import cv2


def getOpticalFlow(video):
    # initialize the list of optical flows
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (64, 64, 1)))

    flows = []
    for i in range(0, len(video) - 1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # subtract the mean in order to eliminate the movement of camera
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # Add into list
        flows.append(flow)

    # Padding the last frame as empty array
    flows.append(np.zeros((64, 64, 2)))

    return np.array(flows, dtype=np.float32)

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=False, default="model_at_epoch_30", help="name of trained model")
    ap.add_argument("-t", "--testset", required=False, default="testset\\", help="path to test images")
    ap.add_argument("-r", "--resize", required=False, default=64, help="frame resize number")
    args = vars(ap.parse_args())

    TESTSET_FOLDER_PATH = args["testset"]
    RESIZE = int(args["resize"])
    MODEL_NAME = args["model"]

    # Load the trained network
    model = load_model("Logs\\" + MODEL_NAME + ".h5")
    model.summary()

    video_paths = list(paths.list_files(TESTSET_FOLDER_PATH))

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        len_frames = int(cap.get(7))

        print(len_frames)
        frames = []
        for i in range(len_frames - 1):
            _, frame = cap.read()
            output = imutils.resize(frame, width=400)
            frame = cv2.resize(frame, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (RESIZE, RESIZE, 3))
            frames.append(frame)

            print(len(frames))

            if len(frames) == 64:
                frames = np.array(frames)

                flows = getOpticalFlow(frames)

                print(frames.shape)
                print(flows.shape)

                result = np.zeros((len(flows), RESIZE, RESIZE, 5))
                result[..., :3] = frames
                result[..., 3:] = flows

                data = np.float32(result)

                # normalize rgb images and optical flows, respectively
                data[..., :3] = normalize(data[..., :3])
                data[..., 3:] = normalize(data[..., 3:])

                print(data.shape)

                data = np.array(data)

                # Classify the input image
                predict = model.predict(data)[0]

                print(predict)

                # Find the winner class and the probability
                probability = predict * 100
                winners_indexes = np.argsort(probability)[::-1]

                # Build the label
                for (i, index) in enumerate(winners_indexes):
                    label = "{}: {:.6f}%".format(index, probability[index])

                    # Draw the label on the image
                    cv2.putText(output, label, (10, (i * 30) + 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

                frames = []

                # Show the output image
            cv2.imshow("Output", output)

            key = cv2.waitKey(500) & 0xFF
            if key == ord("q"):
                break
