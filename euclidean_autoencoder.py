import os
import sys
import numpy as np
from PIL import Image
import glob
from scipy.spatial.distance import euclidean
import cv2
import time

repo_base = "/Users/julianoce/Documents/SFA/"
# repo_base = "C:\\Users\\RodrigodaSilvaFerrei\\Documents\\Repositories\\"

sys.path.append(os.path.abspath(os.path.join(repo_base, 'image-texture-library', 'src')))
sys.path.append(os.path.abspath(os.path.join(repo_base, 'image-experiment-library', 'src')))

import imgtex.feature.lbp

# data_base = "/Users/julianoce/Downloads/chest_xray/"
data_base = "/Users/julianoce/Documents/PUC/autoencoder/"
# data_base = "C:\\Users\\RodrigodaSilvaFerrei\\Documents\\datasets\\"

####DEFINITION####

database = "pertex"
query_dataset = ""
db_dataset = "resized"
tile_size = 64

#################

if database == "pertex":
    class_name_size = 3
    within_dataset_size = 166
elif database == "curet":
    class_name_size = 2
    within_dataset_size = 30
else:
    pass

if db_dataset == "within":
    samples = 8
elif db_dataset == "set":
    samples = 12
else:
    samples = 40

db_path = os.path.join(data_base)
query_path = os.path.join(data_base, query_dataset, '*.jpeg')


def ap(image_list, query_class):
    ap = 0

    for m in range(1, samples):
        ap += (m / rank(image_list, query_class, m))

    ap /= (samples - 1)

    return ap


def rank(image_list, query_class, m):
    hit = 0

    for c in range(len(image_list)):

        db_class = image_list[c]

        if query_class == db_class:

            hit += 1

            if hit >= m:
                return c + 1


def hausdorff_distance(dataA, dataB):
    """
    Computes the hausdorff distance between two feature sets.
    """
    local_distances = []

    for i in range(dataA.shape[0]):
        for j in range(dataB.shape[0]):
            local_distances.append(euclidean(dataA[i, :], dataB[j, :]))

    return local_distances


def distancePixelWise(desc01, desc02):
    # desc01 = imgexp.data.data.normalize(desc01, "minmax")
    # desc02 = imgexp.data.data.normalize(desc02, "minmax")
    return euclidean(desc01.ravel(), desc02.ravel())


def set_distance(dataA, dataB):
    """
    Computes the hausdorff distance between two feature sets.
    """
    local_distancesA = []
    local_distancesB = []

    for i in range(dataA.shape[0]):
        for j in range(dataB.shape[0]):
            local_distancesA.append(distancePixelWise(dataA[i, :], dataB[j, :]))

    for i in range(dataA.shape[0]):
        for j in range(dataB.shape[0]):
            local_distancesB.append(distancePixelWise(dataB[i, :], dataA[j, :]))

    local_distancesA = np.array(local_distancesA)
    local_distancesB = np.array(local_distancesB)
    local_distances = (np.min(local_distancesA) + np.min(local_distancesB)) / 2

    return local_distances


def compute_hausdorff(img_query_path, db_folder_path):
    """
    Computes the Hausdorff distance between a query image and images from a database.

    Parameters
    ----------

    img_query_path: query image path

    db_folder_path: database folder path

    Returns
    -------

    the image with the minimum distance from the database
    """

    image_list = []
    distance_list = []
    trained_data = np.load(db_folder_path + "sub_sample_train_data.npy")

    trained_label = np.load(db_folder_path + "sub_sample_train_label.npy")
    for i in range(trained_data.shape[0]):
        trained = trained_data[i, :, :, :]
        distance_list.append(set_distance(img_query_path, trained))
        image_list.append(trained_label[i])
    distance_list = np.array(distance_list)
    idx = np.argsort(distance_list)
    image_list = np.array(image_list)[idx]

    return image_list, distance_list


def compute_hausdorff_folder(img_query_folder, db_path_folder):
    start_time = time.time()

    hit = 0
    sum_1_by_rank_1 = 0
    sum_ap = 0
    total = 0

    print("")
    print("img_query_folder: " + img_query_folder)
    print("db_path_folder: " + db_path_folder)
    print("samples: " + str(samples))
    print("method: TILE POSITIONAL 64 EUCLIDEAN")
    print("")

    final_distance = []
    test_data = np.load(img_query_folder + "sub_sample_train_data.npy")
    test_label = np.load(img_query_folder + "sub_sample_train_label.npy")
    print(test_data.shape)
    print(test_label.shape)
    for i in range(test_data.shape[0]):
        query_class = test_label[i]
        current_image = test_data[i, :, :, :]
        image_list, distance_list = compute_hausdorff(current_image, img_query_folder)

        if rank(image_list, query_class, 1) == 1:
            hit += 1

        sum_1_by_rank_1 += (1 / rank(image_list, query_class, 1))

        sum_ap += ap(image_list, query_class)

        total += 1
        print(".", end="", flush=True)

    final_distance = np.array(final_distance)
    np.save("distance_matrices/autoencoder_tile_sum_of_min", final_distance)
    print("\nelapsed time: " + str(round(time.time() - start_time)))

    print("\n")

    return {"p@1": hit / total, "mrr": sum_1_by_rank_1 / total, "map": sum_ap / total}

result = compute_hausdorff_folder(db_path, db_path)
print(result)
