# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random


def get_ind(vid, index, ds):
    preprocess = torch.nn.Sequential(
            transforms.Resize(256),
            transforms.CenterCrop(224),
        )
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}/{index:06}.jpg")
    elif ds == "sthsth":
        return preprocess(torchvision.io.read_image(f'/datasets01/SSV2/frames/{vid}/{vid}_{index:06}.jpg'))
    elif ds == "robonet":
        return vid[index]


## Data Loader for Ego4D
class R3MBuffer(IterableDataset):
    def __init__(self, num_workers, source1, source2, alpha, datasources, doaug = "none", simclr=False):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.simclr = simclr
        self.doaug = doaug

        # Augmentations
        if self.simclr:
            cj = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
            self.aug = torch.nn.Sequential(
                    transforms.RandomResizedCrop(224, scale = (0.08, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply([cj], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(23, sigma=(0.1, 2.0))
                )
        else:
            if doaug in ["rc", "rctraj"]:
                self.aug = torch.nn.Sequential(
                    transforms.RandomResizedCrop(224, scale = (0.2, 1.0)),
                )
            else:
                self.aug = lambda a : a

        # Load Data
        if "ego4d" in self.data_sources:
            print("Ego4D")
            self.manifest = pd.read_csv("/private/home/surajn/data/ego4d/manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)
        if "robonet" in self.data_sources:
            print("RoboNet")
            from robonet.datasets import load_metadata, get_dataset_class
            metadata = load_metadata("/private/home/surajn/data/hdf5")
            hparams = {'T': 15, 'action_mismatch': 3, 'state_mismatch': 3, 
                'splits':[0.8, 0.1, 0.1], 'normalize_images':False, 'img_size': [224, 224]}
            self.robonetdata = get_dataset_class("RoboNet")(metadata, source1, hparams)
            self.robonetlen = self.robonetdata.__len__()
        if "sthsth" in self.data_sources:
            self.sthsthdata = pd.read_csv(f"/datasets01/SSV2/{source1}.csv", sep=" ")
            self.sthsthlabels = {}
            f = open(f"/datasets01/SSV2/something-something-v2-{source2}.json")
            for label in json.load(f):
                self.sthsthlabels[label["id"]] = label["label"]


    def _sample(self):
        t0 = time.time()
        ds = random.choice(self.data_sources)

        if ds == "ego4d":
            vidid = np.random.randint(0, self.ego4dlen)
            m = self.manifest.iloc[vidid]
            vidlen = m["len"]
            txt = m["txt"]
            label = txt[2:]
            vid = m["path"]
        elif ds == "sthsth":
            vid = np.random.choice(self.sthsthdata["original_vido_id"])
            vidlen = self.sthsthdata.loc[self.sthsthdata["original_vido_id"] == vid]["frame_id"].max()
            label = self.sthsthlabels[str(vid)]
        elif ds == "robonet":
            vidid = np.random.randint(0, self.robonetlen)
            images, _, _ = self.robonetdata.__getitem__(vidid)
            vid = torch.FloatTensor(images) * 255
            vidlen = 14 
            label = ""

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1-self.alpha) * vidlen)-1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen+1)

        if self.simclr:
            im = get_ind(vid, s1_ind, ds)
            imts1_v1 = self.aug(im / 255.0) * 255.0
            imts1_v2 = self.aug(im / 255.0) * 255.0
            return (imts1_v1, imts1_v2)
        else:   
            if self.doaug == "rctraj":
                im0 = get_ind(vid, start_ind, ds) 
                img = get_ind(vid, end_ind, ds)
                imts0 = get_ind(vid, s0_ind, ds)
                imts1 = get_ind(vid, s1_ind, ds)
                imts2 = get_ind(vid, s2_ind, ds)
                allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
                allims_aug = self.aug(allims / 255.0) * 255.0

                im0 = allims_aug[0]
                img = allims_aug[1]
                imts0 = allims_aug[2]
                imts1 = allims_aug[3]
                imts2 = allims_aug[4]
            else:
                im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
                img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
                imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
                imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
                imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label)

    def __iter__(self):
        while True:
            yield self._sample()