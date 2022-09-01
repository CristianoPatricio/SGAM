import numpy as np
import math
import re
import pickle
import pandas as pd



#################################################################
# Auxiliary Functions
#################################################################

attr_file = pd.read_csv("list_attr_LFWA.csv", sep=",", header=0)
celeb_attributes_names = attr_file.keys()[1:].to_list()


def image_name2image_id(image_name):
    return int(image_name[:-4])


def load_lfw_identities(lfw_identities_file):
    """"
    input: celeba_identities_file - path to the file containing CELEB-A IDs

        identity_CelebA.txt

        image_name_1 person_id_1
        ...
        image_name_n person_id_n


    output: identity_info - dictionary of the list image names per id

        identity_info[person_id] -> (image_name_1, ..., image_name_n)
        image_info[image_id] -> person_id
    """

    identity_info = dict()
    image_info = dict()
    with open(lfw_identities_file, "r") as identities:
        lines = identities.readlines()
        for identity in lines:
            identity = identity.rstrip().lstrip().split()
            # we have 2 infos per line, image name and identity id
            if len(identity) != 2:
                continue
            image_name = identity[0]
            identity_id = int(identity[1])

            if identity_id not in identity_info:
                identity_info[identity_id] = []
            identity_info[identity_id].append(image_name)
            image_info[image_name] = identity_id

    return identity_info, image_info


def get_index_from_samples(samples):
    list_attr_LFWA_df = pd.read_csv("list_attr_LFWA.csv", sep=",")

    idxs = []
    for s in samples:
        idxs.extend(list_attr_LFWA_df.index[list_attr_LFWA_df['image_id'] == s].tolist())
    idxs = np.asarray(idxs)

    return idxs


def get_attr_per_samples(attr_path, samples):
    attr_list = []
    with open(attr_path, "r") as attr_file:
        lines = attr_file.readlines()
        for line in lines[2:]:
            values = line.rstrip().split()
            if values[0] in samples:
                attr_list.append([int(v) for v in values[1:]])

    attr_list = np.asarray(attr_list)
    return attr_list


def load_lfw_attrs(lfw_attributes_file):
    """"
    input: celeba_attributes_file - path to the file containing CELEB-A attributes

        list_attr_celeba.txt
        N (HEADER)
        attribute names (HEADER)
        image_id_1 att_1_1 att_1_2 ... att_1_40
        ...
        image_id_n att_n_1 att_n_2 ... att_n_40


    output: identity_info - dictionary of the bb names per image name

        att_info[image_id] -> (att_n_1, att_n_2, ..., att_n_40)

    """

    list_attr_LFWA_df = pd.read_csv(lfw_attributes_file, sep=",")

    attributes_numpy_array = pd.DataFrame(list_attr_LFWA_df.iloc[:, 1:]).to_numpy()

    return attributes_numpy_array




##########################################################################
# Preprocessing
##########################################################################

np.random.seed(10)

lfw_identity_file = 'identity_lfw.txt'

# att
identity_info, image_info = load_lfw_identities(lfw_identity_file)

n_images = len(image_info.keys())
n_identities = len(identity_info.keys())
identities_list = list(identity_info.keys())
lfw_test_unseen_classes = np.loadtxt("lfw_test_unseen_classes_500.txt").tolist()
shuffle_identities = np.random.permutation(list(set(identities_list).difference(lfw_test_unseen_classes)))

# Generate random splits
training_classes = shuffle_identities[:math.ceil(n_identities * 0.8)]
validation_classes = shuffle_identities[math.ceil(n_identities * 0.8):math.ceil(n_identities * 1)]
#test_classes = shuffle_identities[math.ceil(n_identities * 0.8):]
test_classes = lfw_test_unseen_classes

print(f"No. of training classes: {len(training_classes)}")
print(f"No. of validation classes: {len(validation_classes)}")
print(f"No. of test classes: {len(test_classes)}")

assert n_identities == int(len(training_classes) + len(validation_classes) + len(test_classes))

# Get training samples
training_samples = []
for i in training_classes:
    samples = identity_info[i]
    training_samples.extend(samples)

print(f"Shape of training samples: {np.asarray(training_samples).shape}")

# Get validation samples
val_samples = []
for i in validation_classes:
    samples = identity_info[i]
    val_samples.extend(samples)

print(f"Shape of validation samples: {np.asarray(val_samples).shape}")

# Get test samples
test_samples = []
for i in test_classes:
    samples = identity_info[i]
    test_samples.extend(samples)

print(f"Shape of test samples: {np.asarray(test_samples).shape}")

assert n_images == int(len(training_samples) + len(val_samples) + len(test_samples))

# train_loc
train_loc = get_index_from_samples(training_samples)

# val_loc
val_loc = get_index_from_samples(val_samples)

# trainval_loc
trainval_loc = list(train_loc) + list(val_loc)
trainval_loc = np.asarray(trainval_loc)

# test
test_unseen_loc = get_index_from_samples(test_samples)

# Collect attributes
attrs_per_sample = load_lfw_attrs('list_attr_LFWA.csv')


print(f"//// RESUME ////")
print(f"att: {attrs_per_sample.shape}")
print(f"train_loc: {train_loc.shape}")
print(f"val_loc: {val_loc.shape}")
print(f"trainval_loc: {trainval_loc.shape}")
print(f"test_unseen_loc: {test_unseen_loc.shape}")

dict = {
    'att' : attrs_per_sample,
    'train_loc' : train_loc,
    'trainval_loc' : trainval_loc,
    'val_loc' : val_loc,
    'test_unseen_loc' : test_unseen_loc,
}

# Save dict into a file
with open("lfwa_att_split_top_500_unseen.pickle", "wb") as f:
    pickle.dump(dict, f)
