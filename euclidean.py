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

data_base = "/Users/julianoce/Downloads/chest_xray/"
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

db_path = os.path.join(data_base, db_dataset, '*.jpeg')
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

        name = os.path.basename(image_list[c])
        db_class = name[0:1]

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
    local_distances = []
    sum = 0
    for i in range(dataA.shape[0]):
        for j in range(dataB.shape[0]):
            sum += distancePixelWise(dataA[i, :], dataB[j, :])
            local_distances.append(distancePixelWise(dataA[i, :], dataB[j, :]))

    return (1/(dataA.shape[0] * dataB.shape[0])) * sum


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
    params = ([8], [1], "default")

    img_query = Image.open(img_query_path)
    img_query = img_query.convert('L')

    query_feat = imgtex.feature.lbp.computePerTile(np.array(img_query), tile_size, 0, None, *params)
    query_feat = np.reshape(query_feat, (-1, query_feat.shape[2]))

    for filename in glob.glob(db_folder_path):

        if ("_label" in filename):
            continue

        if (filename == img_query_path):
            continue

        image_list.append(filename)
        img_db = Image.open(filename)
        img_db = img_db.convert('L')

        db_feat = imgtex.feature.lbp.computePerTile(np.array(img_db), tile_size, 0, None, *params)
        db_feat = np.reshape(db_feat, (-1, db_feat.shape[2]))

        distance_list.append(set_distance(query_feat, db_feat))
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

    file_list = glob.glob(img_query_folder)
    final_distance = []
    for filename in file_list:

        if "_label" in filename:
            continue

        # print(filename)

        name = os.path.basename(filename)
        query_class = name[0:1]

        if db_dataset == "within":
            if int(query_class) > within_dataset_size:
                continue

        image_list, distance_list = compute_hausdorff(filename, db_path_folder)
        final_distance.append(distance_list)
        if rank(image_list, query_class, 1) == 1:
            hit += 1

        sum_1_by_rank_1 += (1 / rank(image_list, query_class, 1))

        sum_ap += ap(image_list, query_class)

        total += 1

        print(".", end="", flush=True)

    final_distance = np.array(final_distance)
    np.save("distance_matrices/euclidean_tile_average_64", final_distance)
    print("\nelapsed time: " + str(round(time.time() - start_time)))

    print("\n")

    return {"p@1": hit / total, "mrr": sum_1_by_rank_1 / total, "map": sum_ap / total}


result = compute_hausdorff_folder(db_path, db_path)
print(result)
