"""
Script to generate CAMs and AAM to create more discriminative features

cristiano.patricio@ubi.pt
"""
import keras_vggface.vggface
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras_vggface import utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display
import cv2
import glob
import gc
import time
import pandas as pd
import os


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    array = array.astype('float32')
    array = array / 255.
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names, idx):
    tf.keras.backend.clear_session()
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape(persistent=True) as tape:
        tf.keras.backend.clear_session()
        # Compute activations of the last conv layer and make the tape watch it
        input_tensor = tf.convert_to_tensor(img_array)
        last_conv_layer_output = last_conv_layer_model(input_tensor)
        features_map = np.copy(last_conv_layer_output)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)

        sorted_preds = tf.argsort(preds[0], direction='DESCENDING')
        top_class_channel = preds[:, sorted_preds[idx]]

        # top_pred_index = tf.argmax(preds[0])

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # print(last_conv_layer_output.shape)

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap, features_map


def my_decode_predictions(preds, top=3):
    """

    :return:
    """

    celeb_attributes_names = ["5_o_Clock_Shadow",
                              "Arched_Eyebrows",
                              "Attractive",  # Remove for MobileNet
                              "Bags_Under_Eyes",
                              "Bald",
                              "Bangs",
                              "Big_Lips",
                              "Big_Nose",
                              "Black_Hair",
                              "Blond_Hair",
                              "Blurry",  # Remove for MobileNet
                              "Brown_Hair",
                              "Bushy_Eyebrows",
                              "Chubby",
                              "Double_Chin",
                              "Eyeglasses",
                              "Goatee",
                              "Gray_Hair",
                              "Heavy_Makeup",
                              "High_Cheekbones",
                              "Male",
                              "Mouth_Slightly_Open",
                              "Mustache",
                              "Narrow_Eyes",
                              "No_Beard",
                              "Oval_Face",
                              "Pale_Skin",  # Remove for MobileNet
                              "Pointy_Nose",
                              "Receding_Hairline",
                              "Rosy_Cheeks",
                              "Sideburns",
                              "Smiling",
                              "Straight_Hair",
                              "Wavy_Hair",
                              "Wearing_Earrings",
                              "Wearing_Hat",
                              "Wearing_Lipstick",
                              "Wearing_Necklace",
                              "Wearing_Necktie",
                              "Young"]

    decode_preds = []
    sorted_preds = np.argsort(np.squeeze(preds))[::-1]
    for pred_idx in sorted_preds[:top]:
        decode_preds.append([preds[0, pred_idx], celeb_attributes_names[pred_idx]])

    for decode in decode_preds:
        print(decode)


def extract_features(features_map, attention_map):
    bz, h, w, nc = features_map.shape
    # New Feature Map F' is computed by F' = (F * AAM) + F
    new_f = (np.squeeze(features_map).reshape((nc, h, w)) * attention_map) + np.squeeze(features_map).reshape(
        (nc, h, w))
    # print("Original Feature Maps shape: ", np.squeeze(features_map).reshape((nc, h, w)).shape)
    # print("Original feature:")
    # print(tf.keras.layers.GlobalAveragePooling2D()(features_map))
    # print("New Feature Map shape ", new_f.shape)
    # print("Discriminative Feature:")
    tf.keras.backend.clear_session()
    input_tensor = tf.convert_to_tensor(new_f.reshape((1, h, w, nc)))
    new_feature = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)[0]
    # print(tf.keras.layers.GlobalAveragePooling2D()(new_f.reshape((1, h, w, nc)))[0].numpy())

    return new_feature.numpy()


def generate_CAM_img(img_path, CAMs, n_cams):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Create superimposed images
    for i in range(0, n_cams):
        # We rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * CAMs[i])

        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.3 + img * 0.5
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        save_path = "cam" + str(i) + ".jpg"
        superimposed_img.save(save_path)

        # Display Grad CAM
        display(Image(save_path))


def generate_aam_img(attention_map, img_path):
    # Load the original image
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * attention_map)

    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.3 + img * 0.5
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    save_path = "Attention Map.jpg"
    superimposed_img.save(save_path)


if __name__ == '__main__':

    # Load model
    #model = keras.models.load_model('models/weights-FC40-VGGFace-weights-celeba-0.86.hdf5')
    model = keras.models.load_model('models/model_VGGFace_BEST/weights-FC40-VGGFace-0.87.hdf5')
    #model = keras.models.load_model('/home/cristianopatricio/PycharmProjects/attention-network/weights-FC40-VGGFace-0.82.hdf5')
    #model = VGGFace(include_top=True, input_shape=(224, 224, 3), pooling="avg")

    model.summary()

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

    """
    # Define layer names
    last_conv_layer_name = "conv5_3"
    classifier_layer_names = [
        "pool5",
        "flatten",
        "fc6",
        "fc6/relu",
        "fc7",
        "fc7/relu",
        "fc8",
        "fc8/softmax"
    ]
    """

    # Preprocess image
    main_folder = "/home/cristianopatricio/Documents/Datasets/LFWA/LFWA+/lfw-deepfunneled/"
    images_df = pd.read_csv("list_eval_partition.csv", sep=",")
    files = images_df["image_id"].tolist()
    img_no = 1543 #10033

    img_size = (224, 224)
    img_array = get_img_array(os.path.join(main_folder, files[img_no]), size=img_size)
    #img_array_preproc = utils.preprocess_input(img_array, version=1)

    # Print what the top predicted class is
    preds = model.predict(img_array)
    print("Predictions: ", my_decode_predictions(preds, top=10))

    # Generate CAMs
    CAMs = [0] * 10
    for idx in range(0, 10):
        CAMs[idx], features_map = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names,
                                                       idx)
    CAMs = np.asarray(CAMs)
    print("CAMs shape: ", np.squeeze(CAMs).shape)

    generate_CAM_img(os.path.join(main_folder, files[img_no]), CAMs, n_cams=10)

    # Attribute Attention Map is generated by maximum operation over the CAMs
    attention_map = np.amax(CAMs, axis=0)
    print("AAM shape ", attention_map.shape)

    generate_aam_img(attention_map, os.path.join(main_folder, files[img_no]))

    feat = extract_features(features_map, attention_map)