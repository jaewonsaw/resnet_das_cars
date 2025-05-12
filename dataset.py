# +
import numpy as np
import pandas as pd
from edge_detect import normalize, detect_lines
import torch
from torch.utils.data import Dataset\

import torch.nn.functional as F
import os
from tqdm import tqdm


# -

def das_to_numpy(data_dir, f):
    new_name = os.path.join(data_dir, "FBE_sensor")
    f = f.replace("sensor", "")
    f = f.replace(".png", "")
    f = f.replace(".h5", "")
    f = f.replace("Start", "")
    f = f.replace("End", "")
    f = f.replace("0.0_30.0", "0s-30s")
    f = f.replace("30.0_60.0", "30s-60s")
    f = new_name + f + "_0.1-10Hz.npy"
    return f

def compile_dataset(excel_path, data_dir, save_path = "data.pt"):
    df = pd.read_excel(excel_path)
    values = ['DAS_File', 'DAS_DateTime', 'DAS_Start_Seconds', 'DAS_End_Seconds',
           'GoPro_Start_Video', 'GoPro_Start_Time', 'GoPro_Start_Seconds',
           'GoPro_Start_Scaled_Time', 'GoPro_Start_Scaled_Seconds',
           'GoPro_End_Video', 'GoPro_End_Time', 'GoPro_End_Seconds',
           'GoPro_End_Scaled_Time', 'GoPro_End_Scaled_Seconds', 'class_0_count',
           'class_0_tracks', 'class_1_count', 'class_1_tracks', 'class_2_count',
           'class_2_tracks', 'class_3_count', 'class_3_tracks', 'class_5_count',
           'class_5_tracks', 'class_6_count', 'class_6_tracks', 'class_7_count',
           'class_7_tracks']
    fill = {}
    for c in values:
        if "count" in c:
            fill[c] = 0
        elif "tracks" in c:
            fill[c] = "[]"
        else:
            fill[c] = ""
    df = df.fillna(value = fill)
    imgs = []
    ids = []
    counts = []
    labels = []
    target_shape = (585, 153)

    for r_id, row in tqdm(df.iterrows(), total = len(labels)):
        npy_path = das_to_numpy(data_dir, row.DAS_File) 
        c = 0
        label = [0 for _ in range(8)]

        for j in range(8):
            if j != 4:
                vehicles = [v for v in eval(str(row[f"class_{j}_tracks"])) if v != 1]
                label[j] = min(len(vehicles), 5)
        c = sum(label)
        if c > 0:
            ids.append(r_id)
            imgs.append(np.load(npy_path)[:585, :153])
            counts.append(c)
            labels.append(label)
    imgs = torch.tensor(imgs)
    ids = torch.tensor(ids)
    counts = torch.tensor(counts)
    labels = torch.tensor(labels)
    torch.save({"imgs": imgs, "ids": ids, "counts": counts, "labels": labels}, save_path)


compile_dataset("../das_gopro_mapping_conf0.4.xlsx", "../DAS_data")
#FBE_sensor_2024-11-23T232121-0800_30s-60s_0.1-10Hz.npy

class DAS_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, counts, labels, ids, transforms = lambda x: x, num_count_classes = 7, num_label_classes = 8):
        self.imgs = imgs
        self.labels = labels
        self.counts = counts
        self.labels = labels
        self.transforms = transforms
        self.ids = ids
        self.num_count_classes = num_count_classes
        self.num_label_classes = num_label_classes

    def __getitem__(self, i, return_id = False, one_hot = True):
        if one_hot:
            counts = F.one_hot(self.counts[i], self.num_count_classes)
        else:
            counts = self.counts[i]
        label = self.labels[i]
        
        if return_id:
            return (self.transforms(self.imgs[i].unsqueeze(0)), label, counts, self.ids[i])
        return (self.transforms(self.imgs[i].unsqueeze(0)), label, counts)

    def __len__(self):
        return len(self.imgs)

