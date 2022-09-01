import numpy as np
import pickle
import pandas as pd

# Load each file that contains the custom features
dataset = np.load("/home/cristianopatricio/PycharmProjects/SGAM_LFWA/feats-VGGFace-Weights-CelebA-LFWA-0_13143.npy")

image_files = []
main_folder = "/home/cristianopatricio/Documents/Datasets/LFWA/LFWA+/lfw-deepfunneled/"
images_df = pd.read_csv("list_eval_partition.csv", sep=",")
image_files = images_df["image_id"].tolist()

# add main_folder to path
for i in range(len(image_files)):
    image_files[i] = main_folder + image_files[i]

image_files = np.asarray(image_files)

labels = []
# Get class labels
lfw_identity_file = 'identity_lfw.txt'
lfw_identity_df = pd.read_csv(lfw_identity_file, sep=" ", header=None)
labels = lfw_identity_df.iloc[:, 1]

labels = np.asarray(labels)

print(dataset.shape)
print(image_files.shape)
print(labels.shape)

dict = {
    'features': dataset,
    'image_files': image_files,
    'labels': labels
}

# Sanity check
print(dict["image_files"][0])
print(dict["labels"][0])

# Save dict into a file
with open("vgg_face_lfwa_weights_celeba_2.pickle", "wb") as f:
    pickle.dump(dict, f)
