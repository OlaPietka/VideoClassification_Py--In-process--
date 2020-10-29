from keras import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM, Dropout, Dense, MaxPooling2D, Flatten, Conv2D, \
    AveragePooling2D, BatchNormalization, Activation, Reshape


class VGG16_LSTM:
    def build(width, height, depth, classes):
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(width, height, depth))

        # Disable first top 10 layers
        for layer in base_model.layers:
            layer.trainable = False

        print("---Create LSTM model---")
        model = base_model.get_layer("block5_pool").output
        model = Reshape(target_shape=(3*3, 512))(model)
        model = LSTM(256, return_sequences=False)(model)
        model = Dropout(0.5)(model)
        model = Dense(classes, activation='softmax')(model)

        # Connect VGG16 and LSTM model
        return Model(base_model.input, model)

class RNN:
    def build(batch_size, width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        model.add(LSTM(128, input_shape=inputShape, activation="relu", return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(classes, activation="softmax"))

        return model


class LeNet5:

    def build(width, height, depth, classes):
        # Network architecture:
        # INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC => SOFTMAX

        model = Sequential()
        inputShape = (height, width, depth)
        # C1 Convolutional Layer. Padding='same' gives the same output as input, so the input images could be treated as 32x32 px
        model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=inputShape, padding='same'))
        # S2 Pooling Layer
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # C3 Convolutional Layer
        model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        # S4 Pooling Layer
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        # C5 Fully Connected Convolutional Layer
        model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        # Flatten the CNN output so that we can connect it with fully connected layers
        model.add(Flatten())
        # FC6 Fully Connected Layer
        model.add(Dense(84, activation='tanh'))
        #Output Layer with softmax activation
        model.add(Dense(classes, activation='softmax'))
        
        return model
        
        
class SmallerVGGNet:

    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # sigmoid activation for multi-label classification
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        return model        
        

class FullyConnectedForIMG:

    def build(width, height, depth, classes, hidden):
        # Network architecture:
        # INPUT => FC => RELU => FC => SOFTMAX

        # initialize the model
        model = Sequential()

        # FC => RELU layer
        model.add(Flatten(input_shape=(height, width, depth)))
        model.add(Dense(hidden))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
