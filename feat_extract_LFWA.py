"""
Script to extract discriminative features from LFWA dataset.

cristiano.patricio@ubi.pt
"""
import os.path

from tensorflow import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
import glob
import pandas as pd

tf.config.run_functions_eagerly(True)


#########################################################################
#   AUXILIARY FUNCTIONS
#########################################################################


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    array = array.astype('float32')
    array = array / 255.

    return array


@tf.function
def compute_grads(img_array, model, last_conv_layer_name, classifier_layer_names, idx):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape(persistent=True) as tape:
        # Compute activations of the last conv layer and make the tape watch it
        input_tensor = tf.convert_to_tensor(img_array)
        last_conv_layer_output = last_conv_layer_model(input_tensor)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)

        sorted_preds = tf.argsort(preds[0], direction='DESCENDING')
        top_class_channel = preds[:, sorted_preds[idx]]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    return last_conv_layer_output, pooled_grads


@tf.function
def get_feature_map(img_array, model, last_conv_layer_name):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape(persistent=True) as tape:
        # Compute activations of the last conv layer and make the tape watch it
        input_tensor = tf.convert_to_tensor(img_array)
        last_conv_layer_output = last_conv_layer_model(input_tensor)

    return last_conv_layer_output


def extract_features(features_map, attention_map):
    bz, h, w, nc = features_map.shape
    # New Feature Map F' is computed by F' = (F * AAM) + F
    new_f = (np.squeeze(features_map).reshape((nc, h, w)) * attention_map) + np.squeeze(features_map).reshape(
        (nc, h, w))
    input_tensor = tf.convert_to_tensor(new_f.reshape((1, h, w, nc)))
    new_feature = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)[0]

    return new_feature.numpy()


def compute_CAM(last_conv_layer_output, pooled_grads):
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap


if __name__ == '__main__':

    # Load model
    #model = keras.models.load_model('models/weights-FC40-VGGFace-0.87.hdf5')
    #model = keras.models.load_model('models/weights-FC40-VGGFace-0.82.hdf5')
    model = keras.models.load_model('models/weights-FC40-VGGFace-weights-celeba-0.86.hdf5')

    # Define layer names
    last_conv_layer_name = "conv5_3"
    classifier_layer_names = [
        "pool5",
        "global_average_pooling2d",
        "dense",
        "batch_normalization",
        "dropout",
        "dense_1"
    ]

    # Define range
    start = 0
    stop = 13143

    #######################################################################
    # EXTRACT FEATURES
    #######################################################################

    # List files
    images_df = pd.read_csv("list_eval_partition.csv", sep=",")
    files = images_df["image_id"].tolist()
    print(files)

    main_folder = "/home/cristianopatricio/Documents/Datasets/LFWA/LFWA+/lfw-deepfunneled/"

    feats_list = []
    img_size = (224, 224)
    # Compute features
    for count, img_path in enumerate(files[start:stop]):
        # Clear keras session
        tf.keras.backend.clear_session()

        print("Processing {0}...".format(img_path))

        # Convert raw image to array
        img_array = get_img_array(os.path.join(main_folder, img_path), size=img_size)

        # Generate Class Activation Maps (CAMs)
        CAMs = [0] * 10
        for idx in range(0, 10):
            last_conv_layer_output, pooled_grads = compute_grads(img_array, model, last_conv_layer_name,
                                                                 classifier_layer_names, idx)

            CAMs[idx] = compute_CAM(last_conv_layer_output, pooled_grads)

        CAMs = np.asarray(CAMs)

        # Attribute Attention Map (AAM) is generated by maximum operation over the CAMs
        attention_map = np.amax(CAMs, axis=0)

        # Get features map
        features_map = get_feature_map(img_array, model, last_conv_layer_name)

        # Extract features
        feats = extract_features(features_map, attention_map)
        feats_list.append(feats)

        # Save to file each 1000 ite
        if count % 1000 == 0:
            with open("feats-VGGFace-Weights-CelebA-LFWA-" + str(start) + "_" + str(stop) + ".npy",
                      "wb") as f:
                np.save(f, np.array(feats_list))

    # Save features into .npy file
    feats_list = np.asarray(feats_list)
    with open("feats-VGGFace-Weights-CelebA-LFWA-" + str(start) + "_" + str(stop) + ".npy", "wb") as f:
        np.save(f, feats_list)
