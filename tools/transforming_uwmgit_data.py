import glob
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm


def rle_decode(mask_rle, shape):
    s = np.array(mask_rle.split(), dtype=int)
    starts, lengths = s[0::2] - 1, s[1::2]
    ends = starts + lengths
    h, w = shape
    img = np.zeros((h * w,), dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = 1
    return img.reshape(shape)


df_train = pd.read_csv("./input/uw-madison-gi-tract-image-segmentation/train.csv")
df_train = df_train.sort_values(["id", "class"]).reset_index(drop=True)
df_train["patient"] = df_train.id.apply(lambda x: x.split("_")[0])
df_train["days"] = df_train.id.apply(lambda x: "_".join(x.split("_")[:2]))

all_image_files = sorted(glob.glob("./input/uw-madison-gi-tract-image-segmentation/train/*/*/scans/*.png"),
                         key=lambda x: x.split("/")[3] + "_" + x.split("/")[5])
size_x = [int(os.path.basename(_)[:-4].split("_")[-4]) for _ in all_image_files]
size_y = [int(os.path.basename(_)[:-4].split("_")[-3]) for _ in all_image_files]
spacing_x = [float(os.path.basename(_)[:-4].split("_")[-2]) for _ in all_image_files]
spacing_y = [float(os.path.basename(_)[:-4].split("_")[-1]) for _ in all_image_files]
df_train["image_files"] = np.repeat(all_image_files, 3)
df_train["spacing_x"] = np.repeat(spacing_x, 3)
df_train["spacing_y"] = np.repeat(spacing_y, 3)
df_train["size_x"] = np.repeat(size_x, 3)
df_train["size_y"] = np.repeat(size_y, 3)
df_train["slice"] = np.repeat([int(os.path.basename(_)[:-4].split("_")[-5]) for _ in all_image_files], 3)

for day, group in tqdm(df_train.groupby("days")):
    patient = group.patient.iloc[0]
    imgs = []
    msks = []
    file_names = []
    for file_name in group.image_files.unique():
        img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        segms = group.loc[group.image_files == file_name]
        masks = {}
        for segm, label in zip(segms.segmentation, segms["class"]):
            if not pd.isna(segm):
                mask = rle_decode(segm, img.shape[:2])
                masks[label] = mask
            else:
                masks[label] = np.zeros(img.shape[:2], dtype=np.uint8)
        masks = np.stack([masks[k] for k in sorted(masks)], -1)
        imgs.append(img)
        msks.append(masks)

    imgs = np.stack(imgs, 0)
    msks = np.stack(msks, 0)
    for i in range(msks.shape[0]):
        img = imgs[[max(0, i - 2), i, min(imgs.shape[0] - 1, i + 2)]].transpose(1, 2, 0)  # 2.5d data
        msk = msks[i]
        new_file_name = f"{day}_{i}.png"
        cv2.imwrite(f"./mmseg_train/images/{new_file_name}", img)
        cv2.imwrite(f"./mmseg_train/labels/{new_file_name}", msk)

all_image_files = glob.glob("./mmseg_train/images/*")
patients = [os.path.basename(_).split("_")[0] for _ in all_image_files]

split = list(GroupKFold(5).split(patients, groups=patients))

for fold, (train_idx, valid_idx) in enumerate(split):
    with open(f"./mmseg_train/splits/fold_{fold}.txt", "w") as f:
        for idx in train_idx:
            f.write(os.path.basename(all_image_files[idx])[:-4] + "\n")
    with open(f"./mmseg_train/splits/holdout_{fold}.txt", "w") as f:
        for idx in valid_idx:
            f.write(os.path.basename(all_image_files[idx])[:-4] + "\n")
