import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *

from config import IMG_SHAPE


def squared_difference(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.square(featsA - featsB)
    # return the euclidean distance between the vectors
    return sumSquared


def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))

    loss = 1 - K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss


def build_vgg_19_sister_network(shape, base_weights):
    inputs = tf.keras.layers.Input(shape)
    base_model = tf.keras.applications.vgg19.VGG19(input_shape=shape, include_top=False, weights=base_weights)
    sister_fc_model = build_sister_fc_model((7, 7, 512))  # VGG19 model output is (7,7,512)
    x = base_model(inputs)
    outputs = sister_fc_model(x)
    model = tf.keras.Model(inputs, outputs)
    # model.summary()
    return model, base_model, sister_fc_model


def build_sister_fc_model(shape):
    inputs = tf.keras.layers.Input(shape)

    outputs = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    model = tf.keras.Model(inputs, outputs)
    # model.summary()

    return model


def build_distance_model(shape):
    inputs = tf.keras.layers.Input(shape)

    fc1 = tf.keras.layers.Dense(256, activation="relu")(inputs)
    fc2 = tf.keras.layers.Dense(64, activation="relu")(fc1)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(fc2)

    model = tf.keras.Model(inputs, outputs)
    # model.summary()

    return model


def build_vgg_facenet_sister_network(shape, embedding=64, fine_tune=False, train_percentage=0.0):
    base_model = Sequential()
    base_model.add(ZeroPadding2D((1, 1), input_shape=shape))
    base_model.add(Convolution2D(64, (3, 3), activation='relu'))
    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(64, (3, 3), activation='relu'))
    base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(128, (3, 3), activation='relu'))
    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(128, (3, 3), activation='relu'))
    base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(256, (3, 3), activation='relu'))
    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(256, (3, 3), activation='relu'))
    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(256, (3, 3), activation='relu'))
    base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(512, (3, 3), activation='relu'))
    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(512, (3, 3), activation='relu'))
    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(512, (3, 3), activation='relu'))
    base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(512, (3, 3), activation='relu'))
    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(512, (3, 3), activation='relu'))
    base_model.add(ZeroPadding2D((1, 1)))
    base_model.add(Convolution2D(512, (3, 3), activation='relu'))
    base_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    base_model.add(Convolution2D(4096, (7, 7), activation='relu'))
    base_model.add(Dropout(0.5))
    base_model.add(Convolution2D(4096, (1, 1), activation='relu'))
    base_model.add(Dropout(0.5))
    base_model.add(Convolution2D(2622, (1, 1)))
    base_model.add(Flatten())
    base_model.add(Activation('softmax'))

    base_model.load_weights('../inputs/vgg_face_weights.h5')
    if fine_tune == False:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - int(len(base_model.layers) * train_percentage)
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True

    return base_model


def build_siamese_network(backbone_network_base_weights):
    input_a = tf.keras.layers.Input(shape=IMG_SHAPE)
    input_b = tf.keras.layers.Input(shape=IMG_SHAPE)

    # sister_network = build_vgg_facenet_sister_network(IMG_SHAPE, 64, True, .3)
    sister_network, back_bone, sister_fc_model = build_vgg_19_sister_network(IMG_SHAPE, backbone_network_base_weights)
    distance_network = build_distance_model(512)

    feature_a = sister_network(input_a)
    feature_b = sister_network(input_b)
    # finally, construct the siamese network
    distance = tf.keras.layers.Lambda(squared_difference)([feature_a, feature_b])

    outputs = distance_network(distance)

    # fc1 = tf.keras.layers.Dense(1024, activation="relu")(distance)
    # fc2 = tf.keras.layers.Dense(512, activation="relu")(fc1)
    # outputs = tf.keras.layers.Dense(1, activation="sigmoid")(fc2)
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=outputs)

    optimizer = tf.keras.optimizers.SGD(lr=0.001)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"], )
    # model.summary()

    return model, back_bone, distance_network, sister_fc_model