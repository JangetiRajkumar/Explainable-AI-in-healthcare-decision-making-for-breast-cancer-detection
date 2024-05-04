#import libraries
import os
import pandas as pd

# Get current working directory and images path
current_dir = os.getcwd()

data_path = "images"

# Reshape the images to add an extra dimension
# images = images[..., np.newaxis]

def reshape_and_normalize(images):
    # Normalize pixel values
    images = images / 255.0
    return images

#load training data
train_csv = pd.read_csv("train.csv")

print("Unique patient ids")
print(len(set(train_csv.patient_id)))
print("Laterality counts")
train_subset_0 = train_csv[train_csv.cancer == 0]
train_subset_1 = train_csv[train_csv.cancer == 1]
print(train_subset_0.shape, train_subset_1.shape)
print(train_subset_0.laterality.value_counts())
print(train_subset_1.laterality.value_counts())

#taking subset
train_subset_0_L = train_subset_0[train_subset_0.laterality == "L"].iloc[:588,]
train_subset_0_R = train_subset_0[train_subset_0.laterality == "R"].iloc[:570,]
train_subset_main = pd.concat([train_subset_0_L, train_subset_0_R, train_subset_1])
print(train_subset_main)

#copying dataset into input transformed by segregrating them
import shutil
from tqdm import tqdm
p_id = train_subset_main.patient_id
i_id = train_subset_main.image_id
cncr = train_subset_main.cancer
for pp, ii, cc in tqdm(zip(p_id, i_id, cncr)):
    tmpFile = str(pp) + "_" + str(ii) + ".png"
    tmpSrc = "images/" + tmpFile
    tmpDst = "input_transformed/" + str(cc) + "/" + tmpFile
    shutil.copyfile(tmpSrc, tmpDst)